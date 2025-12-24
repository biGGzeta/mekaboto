"""
RSI / ADX ETH COIN-M
OMNI v32 kernel + v33 hardening
- Timeframes: 5m (trade), 1h (macro)
- Señales compatibles con el stack: type, entry_price, atr, sl_multiplier, risk_reward
- Robustez: retries HTTP, validación de gaps, checksum de estado, deduplicación persistente, time sync
- Sin sizing ni órdenes: el sizing se delega al RiskManager (stop-based en COIN-M)
"""

import os
import json
import hashlib
import logging
import math
import time
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from collections import deque
from typing import Optional, Dict

import pandas as pd
import numpy as np
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# -----------------------------------------------------
# Configuración y logging
# -----------------------------------------------------
@dataclass
class BotConfig:
    SYMBOL: str = "ETHUSD_PERP"
    TF_TRADE: str = "5m"
    TF_MACRO: str = "1h"

    # Ventanas / lógica
    WINDOW_RANK: int = 200
    WINDOW_ADX: int = 14
    COOLDOWN_CANDLES: int = 12  # anti-clustering (5m candles)

    # Pesos para scoring
    W_TREND: dict = field(default_factory=lambda: {"trend": 0.50, "mom": 0.30, "level": 0.10, "fund": 0.10})
    W_RANGE: dict = field(default_factory=lambda: {"level": 0.45, "dist": 0.35, "fund": 0.15, "mom": 0.05})

    # Señal → SL/TP sugeridos (multiplicadores ATR)
    SL_TREND: float = 1.4
    SL_RANGE: float = 1.0
    RR_TREND: float = 2.0
    RR_RANGE: float = 2.5

    # Deduplicación
    MIN_TIME_BETWEEN_SIGNALS_SEC: int = 1200  # 20 minutos
    SCHEMA_VERSION: int = 1


class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "lvl": record.levelname,
            "msg": record.getMessage(),
            "module": record.module
        }
        if hasattr(record, 'extra'):  # optional
            log_obj.update(record.extra)
        return json.dumps(log_obj)


handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger("OMNI_v32_hardened")


# -----------------------------------------------------
# Infra: estado con checksum y backup
# -----------------------------------------------------
class StateStore:
    def __init__(self, symbol: str, version: int = 1):
        self.dir = Path("logs") / symbol.lower()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.file = self.dir / "strategy_state.json"
        self.bak = self.dir / "strategy_state.bak"
        self.version = version

    def _checksum(self, data: dict) -> str:
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def load(self) -> dict:
        def _read(path: Path):
            if not path.exists():
                return {}
            return json.loads(path.read_text(encoding="utf-8"))
        try:
            data = _read(self.file)
            if not data:
                return _read(self.bak)
            stored = data.pop("__checksum__", None)
            if stored and stored != self._checksum({k: v for k, v in data.items() if not k.startswith("__")}):
                logger.error("State checksum mismatch, falling back to bak")
                data = _read(self.bak)
            if data.get("__version__") and data["__version__"] != self.version:
                logger.warning(f"State version mismatch ({data.get('__version__')} != {self.version}), ignoring state")
                return {}
            data.pop("__version__", None)
            return data
        except Exception as e:
            logger.error(f"State load error: {e}")
            return {}

    def save(self, data: dict):
        try:
            data["__version__"] = self.version
            data["__checksum__"] = self._checksum({k: v for k, v in data.items() if not k.startswith("__")})
            tmp = self.file.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            if self.file.exists():
                import shutil
                shutil.copy2(self.file, self.bak)
            tmp.replace(self.file)
        except Exception as e:
            logger.error(f"State save error: {e}")


# -----------------------------------------------------
# Cliente HTTP con retries + sync de tiempo
# -----------------------------------------------------
class HardenedClient:
    BASE = "https://dapi.binance.com/dapi/v1"

    def __init__(self):
        self.s = Session()
        retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 504])
        adapter = HTTPAdapter(max_retries=retry)
        self.s.mount("https://", adapter)
        self.s.mount("http://", adapter)
        self.time_offset_ms = 0
        self._sync_time()

    def _sync_time(self):
        try:
            r = self.s.get(f"{self.BASE}/time", timeout=3)
            r.raise_for_status()
            server_time = r.json().get("serverTime")
            if server_time:
                local = int(time.time() * 1000)
                self.time_offset_ms = server_time - local
                logger.info(json.dumps({"event": "time_sync", "offset_ms": self.time_offset_ms}))
        except Exception:
            logger.warning("Time sync failed; assuming 0 offset")

    def server_now(self) -> datetime:
        ts = int(time.time() * 1000) + self.time_offset_ms
        return datetime.fromtimestamp(ts / 1000, tz=timezone.utc)

    def klines(self, symbol: str, interval: str, limit: int):
        try:
            r = self.s.get(f"{self.BASE}/klines", params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=8)
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, list) or len(data) == 0:
                return None
            return data
        except Exception as e:
            logger.error(f"API error {interval}: {e}")
            return None

    def funding(self, symbol: str) -> float:
        try:
            r = self.s.get(f"{self.BASE}/premiumIndex", params={"symbol": symbol}, timeout=5).json()
            obj = r[0] if isinstance(r, list) else r
            return float(obj.get("lastFundingRate", 0.0))
        except Exception:
            return 0.0


# -----------------------------------------------------
# Validación de dataframes
# -----------------------------------------------------
class DataFrameValidator:
    @staticmethod
    def validate(df: pd.DataFrame, tf_minutes: int) -> pd.DataFrame:
        if df.empty:
            raise ValueError("empty df")
        if df.isnull().values.any():
            raise ValueError("NaN in df")
        if not df["ts"].is_monotonic_increasing:
            raise ValueError("non-monotonic ts")
        # tolerancia 50% (más laxa que v33 estricto)
        diffs = df["ts"].diff().dropna().dt.total_seconds()
        expected = tf_minutes * 60
        if (diffs > expected * 1.5).any():
            raise ValueError("gaps detected")
        return df


# -----------------------------------------------------
# Quant kernel (v32)
# -----------------------------------------------------
class QuantLib:
    @staticmethod
    def wilder(series: pd.Series, n: int) -> pd.Series:
        return series.ewm(alpha=1 / n, adjust=False).mean()

    @staticmethod
    def rolling_rank_norm(series: pd.Series, window: int) -> pd.Series:
        if series.count() < window:
            return pd.Series(np.nan, index=series.index)
        return (series.rolling(window).rank(pct=True) - 0.5) * 2.0

    @staticmethod
    def robust_scale(val: float, history: list) -> float:
        n = len(history)
        if n < 5:
            return 0.0
        arr = np.array(history)
        med = np.median(arr)
        mad = np.median(np.abs(arr - med))
        if mad < 1e-9:
            return 0.0
        z = (val - med) / (mad * 1.4826)
        if n < 20:
            z *= (n / 20.0)
        return z


@dataclass
class Features:
    rsi_rank: float
    rsi_mom: float
    ema_slope: float
    dist_mean: float
    funding: float
    trend_strength: float
    atr: float
    atr_pct: float


class FeatureEngineer:
    def __init__(self, config: BotConfig):
        self.c = config

    def compute(self, df_raw: pd.DataFrame, df_macro: pd.DataFrame, funding_hist: list) -> Optional[Features]:
        df = df_raw.copy()

        # --- Micro (5m) ---
        prev_c = df["c"].shift(1)
        tr = pd.concat([df["h"] - df["l"], (df["h"] - prev_c).abs(), (df["l"] - prev_c).abs()], axis=1).max(axis=1)
        df["atr"] = QuantLib.wilder(tr, 14)
        df["atr_pct"] = df["atr"] / df["c"]

        up = df["h"].diff()
        down = -df["l"].diff()
        pdm = np.where((up > down) & (up > 0), up, 0.0)
        mdm = np.where((down > up) & (down > 0), down, 0.0)
        pdi = 100 * QuantLib.wilder(pd.Series(pdm, index=df.index), 14) / df["atr"]
        mdi = 100 * QuantLib.wilder(pd.Series(mdm, index=df.index), 14) / df["atr"]
        dx = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-9)
        df["adx"] = QuantLib.wilder(dx, self.c.WINDOW_ADX)

        d = df["c"].diff()
        rs = QuantLib.wilder(d.clip(lower=0), 14) / QuantLib.wilder(-d.clip(upper=0), 14).replace(0, 1e-9)
        df["rsi"] = 100 - 100 / (1 + rs)
        df["rsi_rank"] = QuantLib.rolling_rank_norm(df["rsi"], self.c.WINDOW_RANK)

        rsi_vol = df["rsi"].rolling(14).std() + 1e-6
        rsi_mom_raw = df["rsi"].diff(3) / rsi_vol
        rsi_mom_clipped = rsi_mom_raw.clip(lower=rsi_mom_raw.quantile(0.01), upper=rsi_mom_raw.quantile(0.99))
        df["rsi_mom_rank"] = QuantLib.rolling_rank_norm(rsi_mom_clipped, self.c.WINDOW_RANK)

        ema20 = df["c"].ewm(span=20).mean()
        ema50 = df["c"].ewm(span=50).mean()
        dist_raw = ((df["c"] - ema50) * 0.7 + (df["c"] - ema20) * 0.3) / df["atr"]
        df["dist_rank"] = QuantLib.rolling_rank_norm(dist_raw, self.c.WINDOW_RANK)

        # --- Macro (1h) ---
        df_macro = df_macro.sort_values("ts")
        prev_c_macro = df_macro["c"].shift(1)
        tr_macro = pd.concat([df_macro["h"] - df_macro["l"], (df_macro["h"] - prev_c_macro).abs(), (df_macro["l"] - prev_c_macro).abs()], axis=1).max(axis=1)
        macro_atr = QuantLib.wilder(tr_macro, 14).replace(0, 1e-9)
        macro_ema = df_macro["c"].ewm(span=200).mean()
        df_macro["slope_raw"] = macro_ema.diff(5) / macro_atr
        merged = pd.merge_asof(df, df_macro[["ts", "slope_raw"]], on="ts", direction="backward", suffixes=("", "_macro"))
        merged["macro_slope_rank"] = QuantLib.rolling_rank_norm(merged["slope_raw"], self.c.WINDOW_RANK)

        curr = merged.iloc[-1]
        if curr[["rsi_rank", "rsi_mom_rank", "dist_rank", "macro_slope_rank"]].isna().any():
            return None

        clean_funding = [x for x in funding_hist if abs(x) > 1e-9]
        f_norm = np.clip(clean_funding[-1], -0.005, 0.005) / 0.005 if clean_funding else 0.0

        adx_norm = np.clip(curr["adx"] / 50.0, 0, 1)
        slope_mag = abs(curr["macro_slope_rank"])
        trend_strength = max(np.sqrt(adx_norm * slope_mag), slope_mag * 0.4)

        return Features(
            rsi_rank=curr["rsi_rank"],
            rsi_mom=curr["rsi_mom_rank"],
            ema_slope=curr["macro_slope_rank"],
            dist_mean=curr["dist_rank"],
            funding=f_norm,
            trend_strength=trend_strength,
            atr=curr["atr"],
            atr_pct=curr["atr_pct"]
        )


# -----------------------------------------------------
# Scoring (v32)
# -----------------------------------------------------
class ScoringEngine:
    def __init__(self, config: BotConfig):
        self.c = config
        self.hist_norm_trend = deque(maxlen=1000)
        self.hist_norm_range = deque(maxlen=1000)
        self.hist_strength = deque(maxlen=2000)
        self.base_trend = deque(maxlen=2000)
        self.base_range = deque(maxlen=2000)
        self.is_trending = False

    def compute(self, f: Features) -> dict:
        # Regime hysteresis
        self.hist_strength.append(f.trend_strength)
        if len(self.hist_strength) > 100:
            up_t = np.percentile(self.hist_strength, 75)
            down_t = np.percentile(self.hist_strength, 40)
            if self.is_trending and f.trend_strength < down_t:
                self.is_trending = False
            elif not self.is_trending and f.trend_strength > up_t:
                self.is_trending = True

        regime = "TREND" if self.is_trending else "RANGE"
        w = self.c.W_TREND if self.is_trending else self.c.W_RANGE

        if self.is_trending:
            raw = (f.ema_slope * w["trend"]) + (f.rsi_mom * w["mom"]) + (-f.rsi_rank * w["level"]) + (-f.funding * w["fund"])
            hist_norm = self.hist_norm_trend
        else:
            raw = (-f.rsi_rank * w["level"]) + (-f.dist_mean * w["dist"]) + (-f.funding * w["fund"]) + (f.rsi_mom * w["mom"])
            hist_norm = self.hist_norm_range

        norm_val = QuantLib.robust_scale(raw, list(hist_norm))
        hist_norm.append(raw)

        final_score = math.tanh(norm_val)
        side = "LONG" if final_score > 0 else "SHORT"

        slope = f.ema_slope
        if (side == "LONG" and slope < -0.2) or (side == "SHORT" and slope > 0.2):
            final_score *= (1.0 - min(abs(slope), 1.0) * 0.4)

        abs_score = abs(final_score)
        base = self.base_trend if self.is_trending else self.base_range
        percentile = (np.array(base) < abs_score).mean() * 100.0 if len(base) > 50 else 0.0
        base.append(abs_score)

        req = 85.0 if self.is_trending else 95.0
        trigger = percentile > req

        return {
            "score": final_score,
            "pct": percentile,
            "trigger": trigger,
            "regime": regime,
            "side": side
        }


# -----------------------------------------------------
# Estrategia (señales) — payload compatible con el bot
# -----------------------------------------------------
class RSIStrategy:
    def __init__(self):
        self.c = BotConfig()
        self.client = HardenedClient()
        self.state = StateStore(self.c.SYMBOL, version=self.c.SCHEMA_VERSION)
        self.fe = FeatureEngineer(self.c)
        self.eng = ScoringEngine(self.c)
        self.funding_hist = deque(maxlen=100)
        self.last_hash = None
        self.last_close_time_ms = None
        self.last_signal_price = None
        self.last_signal_time = None
        self._hydrate()

    # ------------------ Estado ------------------
    def _hydrate(self):
        d = self.state.load()
        if not d:
            return
        self.last_hash = d.get("last_hash")
        self.last_close_time_ms = d.get("last_close_time_ms")
        self.last_signal_price = d.get("last_signal_price")
        lts = d.get("last_signal_time")
        self.last_signal_time = datetime.fromisoformat(lts) if lts else None
        self.funding_hist = deque(d.get("funding_hist", []), maxlen=100)
        # restore eng hist
        self.eng.hist_norm_trend = deque(d.get("hnt", []), maxlen=1000)
        self.eng.hist_norm_range = deque(d.get("hnr", []), maxlen=1000)
        self.eng.base_trend = deque(d.get("bt", []), maxlen=2000)
        self.eng.base_range = deque(d.get("br", []), maxlen=2000)
        self.eng.is_trending = d.get("regime", False)

    def _persist(self):
        self.state.save({
            "last_hash": self.last_hash,
            "last_close_time_ms": self.last_close_time_ms,
            "last_signal_price": self.last_signal_price,
            "last_signal_time": self.last_signal_time.isoformat() if self.last_signal_time else None,
            "funding_hist": list(self.funding_hist),
            "hnt": list(self.eng.hist_norm_trend),
            "hnr": list(self.eng.hist_norm_range),
            "bt": list(self.eng.base_trend),
            "br": list(self.eng.base_range),
            "regime": self.eng.is_trending,
        })

    # ------------------ Utils ------------------
    def _parse(self, raw):
        df = pd.DataFrame(raw, columns=["ts", "o", "h", "l", "c", "v", "ct", "q", "n", "tb", "tq", "x"])
        df[["o", "h", "l", "c", "v"]] = df[["o", "h", "l", "c", "v"]].astype(float)
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df

    def _closed_candle(self, df: pd.DataFrame) -> pd.Series:
        # usa server time sync para decidir si la última vela está cerrada
        last = df.iloc[-1]
        server_now = self.client.server_now()
        close_time = last["ts"] + timedelta(minutes=5)
        if server_now >= close_time:
            return last
        return df.iloc[-2]

    def _is_duplicate(self, close_time_ms: int, price: float, side: str, regime: str, pct: float) -> bool:
        now = datetime.now(timezone.utc)
        payload = f"{self.c.SYMBOL}|{close_time_ms}|{side}|{regime}|{pct:.1f}"
        h = hashlib.md5(payload.encode()).hexdigest()[:10]
        if h == self.last_hash:
            return True
        if self.last_close_time_ms == close_time_ms:
            return True
        if self.last_signal_price is not None and self.last_signal_time:
            # cooldown temporal
            if (now - self.last_signal_time).total_seconds() < self.c.MIN_TIME_BETWEEN_SIGNALS_SEC:
                return True
        # no duplicado → persistir
        self.last_hash = h
        self.last_close_time_ms = close_time_ms
        self.last_signal_price = price
        self.last_signal_time = now
        self._persist()
        return False

    # ------------------ API pública ------------------
    def generate_signal(self) -> Optional[Dict]:
        # 1) Fetch
        raw5 = self.client.klines(self.c.SYMBOL, self.c.TF_TRADE, 500)
        raw1h = self.client.klines(self.c.SYMBOL, self.c.TF_MACRO, 300)
        if not raw5 or not raw1h:
            return None
        df5 = self._parse(raw5)
        df1h = self._parse(raw1h)
        try:
            df5 = DataFrameValidator.validate(df5, 5)
            df1h = DataFrameValidator.validate(df1h, 60)
        except Exception as e:
            logger.error(f"Data validation: {e}")
            return None

        # 2) Funding
        fr = self.client.funding(self.c.SYMBOL)
        self.funding_hist.append(fr)

        # 3) Selección de vela cerrada
        curr_candle = self._closed_candle(df5)
        close_time_ms = int(curr_candle["ts"].timestamp() * 1000)

        # 4) Features
        feats = self.fe.compute(df5, df1h, list(self.funding_hist))
        if feats is None:
            return None

        # 5) Scoring
        res = self.eng.compute(feats)
        if not res["trigger"]:
            self._persist()
            return None

        side = res["side"]
        regime = res["regime"]
        pct = res["pct"]

        # Deduplicación
        if self._is_duplicate(close_time_ms, curr_candle["c"], side, regime, pct):
            logger.info(json.dumps({"event": "dup_block", "side": side, "regime": regime, "pct": pct}))
            return None

        # 6) Mapear a payload para RiskManager
        if regime == "TREND":
            sl_mult = self.c.SL_TREND
            rr = self.c.RR_TREND
        else:
            sl_mult = self.c.SL_RANGE
            rr = self.c.RR_RANGE

        signal = {
            "timestamp": datetime.now(timezone.utc),
            "candle_close_time": curr_candle["ts"],
            "candle_close_time_ms": close_time_ms,
            "type": side,
            "entry_price": float(curr_candle["c"]),
            "atr": float(feats.atr),
            "sl_multiplier": sl_mult,
            "risk_reward": rr,
            "strategy": "OMNI_v32_hardened",
            "regime": regime,
            "reason": f"score={res['score']:.3f}, pct={pct:.1f}, regime={regime}"
        }

        logger.info(json.dumps({"event": "signal", "side": side, "regime": regime, "pct": pct, "score": res["score"]}))
        return signal

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
        print(f"🛑 SL: {signal['sl_multiplier']} ATR | 🎯 RR: {signal['risk_reward']}:1")
        print(f"📝 Razón: {signal['reason']}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    s = RSIStrategy()
    sig = s.generate_signal()
    if sig:
        s.print_signal(sig)
    else:
        print("💤 Sin señal (o bloqueada por validación/dedup)")
