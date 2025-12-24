"""
RSI/ADX ETH COIN-M (v34 integrado)
- Timeframe operativo: 5m
- Macro filtro: 1h
- Regímenes: TRENDING vs RANGING con histéresis ADX
- Edge: hooks RSI, divergencias, distancia EMA50
- Salida: payload compatible con RiskManager (sl_multiplier, risk_reward, atr, entry_price)
- Robustez: retries HTTP, validación de gaps, deduplicación persistente con checksum, circuit breaker MR
"""
import os
import sys
import json
import time
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Tuple

import requests
import pandas as pd
import numpy as np
from requests import exceptions as req_exc
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from scipy.signal import argrelextrema

# ----------------------------------------
# Métricas simples (stdout JSON)
# ----------------------------------------
class MetricEmitter:
    @staticmethod
    def emit(name, value, labels=None):
        payload = {"type": "metric", "name": name, "value": value}
        if labels:
            payload.update(labels)
        print(json.dumps(payload))


# ----------------------------------------
# Estado persistente con checksum
# ----------------------------------------
class StateStore:
    def __init__(self, symbol: str, version: int = 1):
        self.dir = Path("logs") / symbol.lower()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.file = self.dir / "strategy_state.json"
        self.version = version

    def _checksum(self, data: dict) -> str:
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def load(self) -> dict:
        if not self.file.exists():
            return {}
        try:
            raw = self.file.read_text(encoding="utf-8")
            data = json.loads(raw)
            stored = data.pop("__checksum__", None)
            if stored and stored != self._checksum(data):
                MetricEmitter.emit("state_corruption", 1, {"file": str(self.file)})
                return {}
            if data.get("__version__") != self.version:
                MetricEmitter.emit("state_version_mismatch", 1, {"file": str(self.file)})
                return {}
            data.pop("__version__", None)
            return data
        except Exception as e:
            MetricEmitter.emit("state_load_error", 1, {"error": str(e)})
            return {}

    def save(self, data: dict):
        try:
            data["__version__"] = self.version
            data["__checksum__"] = self._checksum({k: v for k, v in data.items() if not k.startswith("__")})
            tmp = self.file.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            tmp.replace(self.file)
        except Exception as e:
            MetricEmitter.emit("state_save_error", 1, {"error": str(e)})


# ----------------------------------------
# Estrategia
# ----------------------------------------
class RSIStrategy:
    """
    Estrategia RSI/ADX para COIN-M ETH, compatible con el pipeline existente.
    """

    BASE_URL = "https://dapi.binance.com/dapi/v1/klines"  # COIN-M
    SYMBOL = "ETHUSD_PERP"
    TF_TRADE = "5m"
    TF_MACRO = "1h"

    # Parámetros de señal
    RSI_PERIOD = 14
    ATR_PERIOD = 14
    RSI_OB = 70    # sobrecompra
    RSI_OS = 30    # sobreventa
    ADX_PERIOD = 14
    ADX_ENTER = 25
    ADX_EXIT = 20

    # Distancias y filtros
    DIST_MIN_R = 1.0   # distancia mínima a EMA50 en ATR
    CLIMAX_VELAS_R = 1.2
    GAP_MAX_R = 0.8
    WICK_RATIO_MAX = 3.0
    VOL_SPIKE_MULT = 2.0

    # SL/TP dinámico (Multiplicadores para el Risk Manager)
    SL_REVERSION_R = 1.0
    SL_PULLBACK_R = 1.4
    TP_REVERSION_RR = 3.0
    TP_PULLBACK_RR = 1.5

    # Duplicados
    MIN_TIME_BETWEEN_SIGNALS = 1200  # 20 minutos

    def __init__(self):
        self.last_regime = None
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "RSI-ADX-ETH/2.0"})
        # Retries HTTP
        retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 504], allowed_methods=["GET"])
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        # Estado persistente
        self.state = StateStore(self.SYMBOL, version=1)
        persisted = self.state.load()
        self.last_signal_hash = persisted.get("last_signal_hash")
        self.last_close_time_ms = persisted.get("last_close_time_ms")
        self.last_signal_price = persisted.get("last_signal_price")
        last_time_str = persisted.get("last_signal_time")
        self.last_signal_time = datetime.fromisoformat(last_time_str) if last_time_str else None

        # Circuit breaker MR
        self.mr_cooldown_until = None
        self.last_mr_signal_price = None
        self.last_mr_side = None

        # Rate-limit por close_time
        self.last_candle_check_ms = None

        # Buffer
        self.df_5m = None

    # ---------- Utils de mercado ----------
    def fetch_klines(self, interval: str, limit: int = 500) -> pd.DataFrame:
        params = {"symbol": self.SYMBOL, "interval": interval, "limit": limit}
        try:
            resp = self.session.get(self.BASE_URL, params=params, timeout=8)
            resp.raise_for_status()
            data = resp.json()
        except req_exc.RequestException as e:
            MetricEmitter.emit("fetch_error", 1, {"interval": interval, "error": str(e)})
            return pd.DataFrame()

        cols = [
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "q_vol", "trades", "tb_base", "tb_quote", "ignore"
        ]
        df = pd.DataFrame(data, columns=cols)
        if df.empty:
            return df

        float_cols = ["open", "high", "low", "close", "volume"]
        df[float_cols] = df[float_cols].apply(pd.to_numeric, errors="coerce")
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

        # Validación de continuidad (tolerancia 50%)
        if len(df) > 1:
            expected = pd.Timedelta(minutes=5) if interval == "5m" else pd.Timedelta(hours=1)
            if (df["timestamp"].diff().dropna() > expected * 1.5).any():
                MetricEmitter.emit("gap_detected", 1, {"interval": interval})
                return pd.DataFrame()

        return df

    # ---------- Indicadores ----------
    def _atr_wilder(self, df: pd.DataFrame, period: int) -> pd.Series:
        h_l = df["high"] - df["low"]
        h_pc = (df["high"] - df["close"].shift()).abs()
        l_pc = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
        return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    def _rsi_wilder(self, df: pd.DataFrame, period: int) -> pd.Series:
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-6)
        return 100 - (100 / (1 + rs))

    def _ema(self, df: pd.DataFrame, period: int) -> pd.Series:
        return df["close"].ewm(span=period, adjust=False).mean()

    def _adx_wilder(self, df: pd.DataFrame, period: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        high, low, close = df["high"], df["low"], df["close"]
        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = pd.Series(plus_dm, index=df.index)
        minus_dm = pd.Series(minus_dm, index=df.index)

        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

        plus_di = 100 * plus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr.replace(0, 1e-6)
        minus_di = 100 * minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr.replace(0, 1e-6)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-6)
        adx = dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        return adx, plus_di, minus_di

    def _detect_divergence(self, df: pd.DataFrame, order: int = 5, max_bars_ago: int = 5) -> Optional[str]:
        if len(df) < order * 4:
            return None
        recent = df.iloc[-(order * 4):]
        if recent["rsi"].isna().any():
            return None

        highs = recent['high'].values
        lows = recent['low'].values
        rsis = recent['rsi'].values

        # Bearish
        high_peak_idxs = argrelextrema(highs, np.greater, order=order)[0]
        if len(high_peak_idxs) >= 2:
            last_peak_idx = high_peak_idxs[-1]
            prev_peak_idx = high_peak_idxs[-2]
            bars_ago = len(recent) - 1 - last_peak_idx
            if bars_ago <= max_bars_ago:
                if highs[last_peak_idx] > highs[prev_peak_idx] and rsis[last_peak_idx] < rsis[prev_peak_idx] and rsis[last_peak_idx] > 60:
                    return "BEARISH_DIV"

        # Bullish
        low_trough_idxs = argrelextrema(lows, np.less, order=order)[0]
        if len(low_trough_idxs) >= 2:
            last_trough_idx = low_trough_idxs[-1]
            prev_trough_idx = low_trough_idxs[-2]
            bars_ago = len(recent) - 1 - last_trough_idx
            if bars_ago <= max_bars_ago:
                if lows[last_trough_idx] < lows[prev_trough_idx] and rsis[last_trough_idx] > rsis[prev_trough_idx] and rsis[last_trough_idx] < 40:
                    return "BULLISH_DIV"
        return None

    # ---------- Regímenes ----------
    def _regime(self, adx_val: float) -> str:
        if self.last_regime == "TRENDING":
            is_trending = adx_val > self.ADX_EXIT
        else:
            is_trending = adx_val > self.ADX_ENTER
        self.last_regime = "TRENDING" if is_trending else "RANGING"
        return self.last_regime

    # ---------- Filtros ----------
    def _is_climatic(self, row) -> bool:
        return row["body"] > self.CLIMAX_VELAS_R * row["atr"]

    def _has_gap(self, df: pd.DataFrame, idx: int) -> bool:
        if idx < 1:
            return False
        prev_close = df.iloc[idx - 1]["close"]
        current_open = df.iloc[idx]["open"]
        return abs(current_open - prev_close) > self.GAP_MAX_R * df.iloc[idx]["atr"]

    def _has_bad_wicks(self, row) -> bool:
        return row["wick_ratio"] > self.WICK_RATIO_MAX

    def _has_vol_spike(self, row) -> bool:
        return row["vol_spike"]

    def _is_breakout_candle(self, row: pd.Series) -> bool:
        body_size = abs(row['close'] - row['open'])
        is_big = body_size > (2.0 * row['atr'])
        if row['close'] > row['open']:
            strong_close = (row['high'] - row['close']) < (0.2 * body_size)
            return is_big and strong_close
        else:
            strong_close = (row['close'] - row['low']) < (0.2 * body_size)
            return is_big and strong_close

    # ---------- Circuit Breaker ----------
    def _check_circuit_breaker(self, current_price: float, current_time: datetime, atr: float) -> bool:
        if self.mr_cooldown_until:
            if current_time > self.mr_cooldown_until:
                self.mr_cooldown_until = None
                self.last_mr_signal_price = None
                self.last_mr_side = None
            else:
                return True

        if self.last_mr_signal_price and self.last_mr_side:
            threshold = 1.5 * atr
            if self.last_mr_side == "SHORT" and current_price > (self.last_mr_signal_price + threshold):
                self.mr_cooldown_until = current_time + timedelta(hours=4)
                MetricEmitter.emit("mr_circuit_breaker", 1, {"side": "SHORT"})
                return True
            if self.last_mr_side == "LONG" and current_price < (self.last_mr_signal_price - threshold):
                self.mr_cooldown_until = current_time + timedelta(hours=4)
                MetricEmitter.emit("mr_circuit_breaker", 1, {"side": "LONG"})
                return True
        return False

    # ---------- Deduplicación ----------
    def _setup_hash(self, signal_type: str, strategy: str, regime: str, close_val: float, rsi_val: float, adx_val: float, distance: float) -> str:
        s = f"{signal_type}|{strategy}|{regime}|{close_val:.2f}|{rsi_val:.1f}|{adx_val:.1f}|{distance:.2f}"
        return hashlib.md5(s.encode()).hexdigest()[:8]

    def _is_duplicate(self, signal_type: str, strategy: str, regime: str,
                      close_time_ms: int, close_val: float, rsi_val: float,
                      adx_val: float, distance: float) -> bool:
        now = datetime.now(timezone.utc)
        setup_hash = self._setup_hash(signal_type, strategy, regime, close_val, rsi_val, adx_val, distance)

        if setup_hash == self.last_signal_hash:
            return True
        if close_time_ms == self.last_close_time_ms:
            return True

        if self.last_signal_price is not None and self.df_5m is not None and not self.df_5m.empty:
            atr_current = self.df_5m.iloc[-2]["atr"]
            price_move = abs(close_val - self.last_signal_price)
            if price_move < 0.5 * atr_current:
                return True

        if self.last_signal_time and (now - self.last_signal_time).total_seconds() < self.MIN_TIME_BETWEEN_SIGNALS:
            return True

        # No duplicado: persistir
        self.last_signal_hash = setup_hash
        self.last_close_time_ms = close_time_ms
        self.last_signal_price = close_val
        self.last_signal_time = now
        self.state.save({
            "last_signal_hash": self.last_signal_hash,
            "last_close_time_ms": self.last_close_time_ms,
            "last_signal_price": self.last_signal_price,
            "last_signal_time": self.last_signal_time.isoformat(),
        })
        return False

    def _log_signal(self, signal: Dict):
        log_dir = Path("logs") / self.SYMBOL.lower()
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"signals_{datetime.now(timezone.utc).strftime('%Y%m')}.jsonl"
        try:
            log_entry = {
                **signal,
                "timestamp": signal["timestamp"].isoformat() if isinstance(signal["timestamp"], datetime) else signal["timestamp"],
                "candle_close_time": signal["candle_close_time"].isoformat() if isinstance(signal["candle_close_time"], datetime) else signal["candle_close_time"]
            }
            with open(log_file, "a", encoding="utf-8") as f:
                json.dump(log_entry, f, ensure_ascii=False)
                f.write("\n")
        except Exception as e:
            MetricEmitter.emit("signal_log_error", 1, {"error": str(e)})

    def print_signal(self, signal: Dict):
        if not signal:
            return
        print(f"\n{'='*60}")
        print(f"🚨 SEÑAL DE TRADING")
        print(f"{'='*60}")
        print(f"⏰ Timestamp: {signal['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Tipo: {signal['type']}")
        print(f"📈 Estrategia: {signal.get('strategy', 'N/A')}")
        print(f"💰 Precio entrada: ${signal['entry_price']:,.4f}")
        print(f"📊 ATR: ${signal['atr']:.4f}")
        print(f"🎯 TP Ratio: {signal.get('risk_reward', 'N/A')}:1")
        print(f"🛑 SL Ratio: {signal.get('sl_multiplier', 'N/A')} ATR")
        print(f"📝 Razón: {signal['reason']}")
        print(f"{'='*60}\n")

    # ---------- Entrada pública ----------
    def generate_signal(self) -> Optional[Dict]:
        # Rate limit por close_time real
        df_trade = self.fetch_klines(self.TF_TRADE, limit=500)
        if df_trade.empty:
            return None
        last_close = int(df_trade.iloc[-2]["close_time"].timestamp() * 1000)
        if self.last_candle_check_ms == last_close:
            return None
        self.last_candle_check_ms = last_close
        return self.get_signal()

    # ---------- Núcleo de señal ----------
    def get_signal(self) -> Optional[Dict]:
        df_trade = self.fetch_klines(self.TF_TRADE, limit=500)
        df_macro = self.fetch_klines(self.TF_MACRO, limit=400)
        if df_trade.empty or df_macro.empty:
            return None

        self.df_5m = self._compute_indicators(df_trade, ema_periods=(50, 200))
        df_macro = self._compute_indicators(df_macro, ema_periods=(50, 200))

        self.df_5m = self.df_5m.dropna().reset_index(drop=True)
        df_macro = df_macro.dropna().reset_index(drop=True)
        if len(self.df_5m) < 50 or len(df_macro) < 50:
            return None

        curr = self.df_5m.iloc[-2]  # vela cerrada
        prev = self.df_5m.iloc[-3]
        macro_curr = df_macro.iloc[-2]

        atr = curr["atr"]
        adx_macro = macro_curr["adx"]
        regime = self._regime(adx_macro)

        ema200_slope = macro_curr["ema_200"] - df_macro.iloc[-6]["ema_200"]
        bias_bull = (macro_curr["close"] > macro_curr["ema_200"]) and (macro_curr["di_plus"] > macro_curr["di_minus"]) and (ema200_slope > 0)
        bias_bear = (macro_curr["close"] < macro_curr["ema_200"]) and (macro_curr["di_minus"] > macro_curr["di_plus"]) and (ema200_slope < 0)

        # Filtros de vela
        if self._is_climatic(curr):
            return None
        if self._has_gap(self.df_5m, len(self.df_5m) - 2):
            return None
        if self._has_bad_wicks(curr):
            return None
        if self._has_vol_spike(curr):
            return None

        distance_to_ema50 = abs(curr["close"] - curr["ema_50"]) / atr if atr > 0 else 0
        rsi = curr["rsi"]
        rsi_prev = prev["rsi"]
        divergence = self._detect_divergence(self.df_5m.iloc[:-1])

        signal = {
            "timestamp": datetime.now(timezone.utc),
            "candle_close_time": curr["close_time"],
            "candle_close_time_ms": int(curr["close_time"].timestamp() * 1000),
            "type": None,
            "entry_price": curr["close"],
            "close_price": curr["close"],
            "rsi": rsi,
            "atr": atr,
            "adx": adx_macro,
            "regime": regime,
            "strategy": None,
            "risk_reward": None,
            "sl_multiplier": None,
            "reason": "",
        }

        # --------- RANGING: Mean Reversion ---------
        if regime == "RANGING":
            now = datetime.now(timezone.utc)
            if self._check_circuit_breaker(curr["close"], now, curr["atr"]):
                return None

            if self._is_breakout_candle(curr):
                return None

            macro_distance = abs(macro_curr["close"] - macro_curr["ema_200"]) / macro_curr["atr"]
            if macro_distance > 1.0:
                return None

            if ((rsi_prev > self.RSI_OB and rsi < self.RSI_OB) or divergence == "BEARISH_DIV") and distance_to_ema50 > self.DIST_MIN_R:
                signal["type"] = "SHORT"
                signal["strategy"] = "MEAN_REVERSION"
                signal["sl_multiplier"] = self.SL_REVERSION_R
                signal["risk_reward"] = self.TP_REVERSION_RR
                signal["reason"] = f"Rango: hook OB/div bajista, dist {distance_to_ema50:.2f}R"
                self.last_mr_signal_price = curr["close"]
                self.last_mr_side = "SHORT"

            elif ((rsi_prev < self.RSI_OS and rsi > self.RSI_OS) or divergence == "BULLISH_DIV") and distance_to_ema50 > self.DIST_MIN_R:
                signal["type"] = "LONG"
                signal["strategy"] = "MEAN_REVERSION"
                signal["sl_multiplier"] = self.SL_REVERSION_R
                signal["risk_reward"] = self.TP_REVERSION_RR
                signal["reason"] = f"Rango: hook OS/div alcista, dist {distance_to_ema50:.2f}R"
                self.last_mr_signal_price = curr["close"]
                self.last_mr_side = "LONG"

        # --------- TRENDING: Pullbacks ---------
        elif regime == "TRENDING":
            if distance_to_ema50 > 1.5:
                return None
            needs_ema200_5m = adx_macro < 35

            if bias_bull and rsi < 45 and curr["close"] > curr["ema_50"]:
                if needs_ema200_5m and curr["close"] < curr["ema_200"]:
                    return None
                if rsi > rsi_prev:
                    signal["type"] = "LONG"
                    signal["strategy"] = "TREND_PULLBACK"
                    signal["sl_multiplier"] = self.SL_PULLBACK_R
                    signal["risk_reward"] = 2.0 if (adx_macro > 35 and distance_to_ema50 < 0.5) else self.TP_PULLBACK_RR
                    signal["reason"] = f"Trend bull: RSI dip hook up, ADX {adx_macro:.1f}, dist {distance_to_ema50:.2f}R"

            elif bias_bear and rsi > 55 and curr["close"] < curr["ema_50"]:
                if needs_ema200_5m and curr["close"] > curr["ema_200"]:
                    return None
                if rsi < rsi_prev:
                    signal["type"] = "SHORT"
                    signal["strategy"] = "TREND_PULLBACK"
                    signal["sl_multiplier"] = self.SL_PULLBACK_R
                    signal["risk_reward"] = 2.0 if (adx_macro > 35 and distance_to_ema50 < 0.5) else self.TP_PULLBACK_RR
                    signal["reason"] = f"Trend bear: RSI rally hook down, ADX {adx_macro:.1f}, dist {distance_to_ema50:.2f}R"

        if signal["type"] is None:
            return None

        # Deduplicación
        if self._is_duplicate(signal["type"], signal["strategy"], signal["regime"],
                              signal["candle_close_time_ms"], signal["close_price"],
                              signal["rsi"], signal["adx"], distance_to_ema50):
            MetricEmitter.emit("signal_blocked_duplicate", 1, {"type": signal["type"]})
            return None

        self._log_signal(signal)
        MetricEmitter.emit("signal_emitted", 1, {"type": signal["type"], "strategy": signal["strategy"], "regime": signal["regime"]})
        return signal

    # ---------- Compute indicators wrapper ----------
    def _compute_indicators(self, df: pd.DataFrame, ema_periods=(50, 200)) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        required_cols = ["open", "high", "low", "close", "volume"]
        df = df.dropna(subset=required_cols)

        df["rsi"] = self._rsi_wilder(df, self.RSI_PERIOD)
        df["atr"] = self._atr_wilder(df, self.ATR_PERIOD)
        df["ema_50"] = self._ema(df, ema_periods[0])
        df["ema_200"] = self._ema(df, ema_periods[1])
        adx, di_plus, di_minus = self._adx_wilder(df, self.ADX_PERIOD)
        df["adx"], df["di_plus"], df["di_minus"] = adx, di_plus, di_minus

        df["body"] = (df["close"] - df["open"]).abs()
        df["wick_top"] = df["high"] - df[["open", "close"]].max(axis=1)
        df["wick_bot"] = df[["open", "close"]].min(axis=1) - df["low"]
        df["wick_ratio"] = (df["wick_top"] + df["wick_bot"]) / df["body"].replace(0, np.nan)
        df["vol_med_50"] = df["volume"].rolling(50).median()
        df["vol_spike"] = df["volume"] > (df["vol_med_50"] * self.VOL_SPIKE_MULT)
        return df


if __name__ == "__main__":
    s = RSIStrategy()
    print("🔬 Analizando mercado COIN-M ETH (5m / macro 1h)...")
    sig = s.generate_signal()
    if sig:
        s.print_signal(sig)
    else:
        print("💤 Sin setup de alta probabilidad (o rate limit activo)")
