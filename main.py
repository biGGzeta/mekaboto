"""
Bot principal para ETH COIN-M
Monitorea señales, ejecuta trades y maneja trailing stop
"""
import sys
import os
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, root_dir)

import time
from datetime import datetime, timedelta
from typing import Optional, Dict
import signal as signal_module
import json

from coinm.eth.config import ETHConfig
from coinm.eth.strategy import RSIStrategy
from coinm.eth.risk_manager import RiskManager
from coinm.eth.executor import OrderExecutor
from coinm.trailing_manager import TrailingStopManager
from utils.helpers import ColorPrinter

class ETHBot:
    """Bot de trading para ETH COIN-M"""
    
    def __init__(self, dry_run=True, use_signal_handler=True):
        """
        Args:
            dry_run: Si True, no ejecuta órdenes reales
            use_signal_handler: Si True, registra handler para Ctrl+C
        """
        self.config = ETHConfig
        self.strategy = RSIStrategy()
        self.risk_mgr = RiskManager()
        self.executor = OrderExecutor()
        self.trailing_mgr = TrailingStopManager(ETHConfig, 'ETHUSD_PERP')
        
        self.dry_run = dry_run
        self.use_signal_handler = use_signal_handler
        self.running = False
        self.current_position = None
        self.current_trade_plan = None
        self.last_check_time = None
        self.orchestrator = None  # Se asigna desde orchestrator
        self.bot_id = 'eth'  # ID para orchestrator
        
        # Stats
        self.signals_generated = 0
        self.trades_executed = 0
        self.trades_won = 0
        self.trades_lost = 0
        
        # MAE/MFE tracking
        self.trade_entry_time = None
        self.trade_bars_held = 0
        self.mae_price = None  # Peor precio durante el trade
        self.mfe_price = None  # Mejor precio durante el trade
        
        # Archivo para persistir trade plan
        self.trade_plan_file = os.path.join(root_dir, 'logs', 'ethusd_perp', 'current_trade_plan.json')
    
    def save_trade_plan(self):
        """Guarda el trade plan actual en archivo"""
        if self.current_trade_plan:
            try:
                os.makedirs(os.path.dirname(self.trade_plan_file), exist_ok=True)
                with open(self.trade_plan_file, 'w') as f:
                    json.dump(self.current_trade_plan, f, indent=2)
            except Exception as e:
                print(f"   ⚠️  Error guardando trade plan: {e}")
    
    def load_trade_plan(self) -> Optional[Dict]:
        """Carga el trade plan desde archivo"""
        try:
            if os.path.exists(self.trade_plan_file):
                with open(self.trade_plan_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"   ⚠️  Error cargando trade plan: {e}")
        return None
    
    def clear_trade_plan(self):
        """Elimina el archivo de trade plan"""
        try:
            if os.path.exists(self.trade_plan_file):
                os.remove(self.trade_plan_file)
        except Exception as e:
            print(f"   ⚠️  Error eliminando trade plan: {e}")
    
    def start(self):
        """Inicia el bot"""
        self.running = True
        
        print(f"\n{'#'*60}")
        print(f"{'#'*60}")
        print(f"🤖 ETH COIN-M BOT INICIADO")
        print(f"{'#'*60}")
        print(f"{'#'*60}\n")
        
        print(f"⏰ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Symbol: {self.config.SYMBOL}")
        print(f"⏱️  Timeframe: {self.config.TIMEFRAME}")
        print(f"🎯 Estrategia: v33 OMNI (Multi-factor, Regime-adaptive)")
        print(f"💰 Riesgo: {self.config.RISK_PER_TRADE}% por trade")
        
        if self.dry_run:
            ColorPrinter.warning("\n⚠️  MODO DRY RUN - No se ejecutarán órdenes reales")
        else:
            ColorPrinter.error("\n🔴 MODO LIVE - Las órdenes se ejecutarán en la cuenta real")
        
        print(f"\n{'='*60}\n")
        
        # Configurar handler para Ctrl+C solo si es thread principal
        if self.use_signal_handler:
            signal_module.signal(signal_module.SIGINT, self._signal_handler)
        
        try:
            self.run_loop()
        except KeyboardInterrupt:
            self.stop()
    
    def _signal_handler(self, signum, frame):
        """Handler para detener el bot con Ctrl+C"""
        print("\n\n⚠️  Señal de interrupción recibida...")
        self.stop()
    
    def stop(self):
        """Detiene el bot"""
        self.running = False
        print("\n🛑 Deteniendo bot...")
        self.print_stats()
        print("\n✅ Bot detenido\n")
    
    def run_loop(self):
        """Loop principal del bot"""
        check_interval = 60  # Chequear cada 60 segundos (1min candles)
        
        while self.running:
            try:
                current_time = datetime.now()
                
                # Mostrar heartbeat cada minuto
                if self.last_check_time is None or \
                   (current_time - self.last_check_time).total_seconds() >= check_interval:
                    
                    print(f"\n💓 [{current_time.strftime('%H:%M:%S')}] Chequeando mercado...")
                    
                    # 1. Verificar posición actual
                    self.check_position_status()
                    
                    # 2. Si no tenemos posición, buscar señales (solo en intervalos de 15min)
                    if self.current_position is None:
                        # Solo buscar señales cuando cierra vela de 15min
                        if current_time.minute % 15 == 0:
                            self.check_for_signals()
                        else:
                            print(f"   ⏰ Esperando cierre de vela 15min (próximo: {15 - (current_time.minute % 15)}min)")
                    else:
                        # 3. Si tenemos posición, gestionar trailing
                        self.manage_trailing_stop()
                    
                    self.last_check_time = current_time
                
                # Dormir un poco
                time.sleep(1)
                
            except Exception as e:
                ColorPrinter.error(f"❌ Error en loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(5)
    
    def check_position_status(self):
        """Verifica si tenemos posición abierta"""
        previous_position = self.current_position
        self.current_position = self.executor.get_current_position()
        
        # Si tenemos posición pero no trade plan, intentar recuperar
        if self.current_position and not self.current_trade_plan:
            self.current_trade_plan = self.load_trade_plan()
            if self.current_trade_plan:
                print(f"   ✅ Trade plan recuperado desde archivo")
            else:
                print(f"   ⚠️  Posición detectada pero sin trade plan guardado")
        
        # Detectar cierre de posición
        if previous_position and not self.current_position:
            # La posición se cerró
            
            # Guardar MAE/MFE antes de resetear
            if self.current_trade_plan and self.mae_price and self.mfe_price:
                entry_price = self.current_trade_plan['entry_price']
                initial_sl = self.current_trade_plan['stop_loss']
                side = self.current_trade_plan['side']
                
                # Calcular MAE y MFE en R
                if side == 'LONG':
                    risk = entry_price - initial_sl
                    mae_r = (self.mae_price - entry_price) / risk if risk > 0 else 0
                    mfe_r = (self.mfe_price - entry_price) / risk if risk > 0 else 0
                else:  # SHORT
                    risk = initial_sl - entry_price
                    mae_r = (entry_price - self.mae_price) / risk if risk > 0 else 0
                    mfe_r = (entry_price - self.mfe_price) / risk if risk > 0 else 0
                
                # Log MAE/MFE
                self.executor.logger.log_mae_mfe(
                    mae_r=mae_r,
                    mfe_r=mfe_r,
                    bars_held=self.trade_bars_held,
                    mae_price=self.mae_price,
                    mfe_price=self.mfe_price
                )
            
            # Resetear todo
            self.current_trade_plan = None
            self.trailing_mgr.reset()
            self.mae_price = None
            self.mfe_price = None
            self.trade_bars_held = 0
            self.trade_entry_time = None
            
            # Limpiar archivo de trade plan
            self.clear_trade_plan()
            
            if self.orchestrator:
                self.orchestrator.register_position_closed(self.bot_id)
            print(f"   ✅ Posición cerrada")
        
        if self.current_position:
            pos = self.current_position
            print(f"   📊 Posición activa: {pos['symbol']}")
            print(f"      Size: {pos['positionAmt']}")
            print(f"      Entry: ${pos['entryPrice']:,.2f}")
            print(f"      Mark: ${pos['markPrice']:,.2f}")
            print(f"      PnL: {pos['unRealizedProfit']:.4f} {self.config.MARGIN_ASSET}")
        else:
            print(f"   ℹ️  Sin posiciones abiertas")
    
    def check_for_signals(self):
        """Busca señales de trading"""
        print(f"   🔍 Buscando señales...")
        
        # Obtener contexto
        context = self.strategy.get_market_context()
        print(f"      RSI: {context.get('rsi', 0):.2f}")
        print(f"      Price: ${context.get('current_price', 0):,.2f}")
        
        # Generar señal
        signal = self.strategy.generate_signal()
        
        if signal:
            # Verificar si el orchestrator permite abrir posición (con el lado específico)
            if self.orchestrator:
                can_trade, reason = self.orchestrator.can_open_position(self.bot_id, signal['type'])
                if not can_trade:
                    ColorPrinter.warning(f"   ⏸️  No se puede abrir posición: {reason}")
                    return
            
            self.signals_generated += 1
            ColorPrinter.success(f"\n   🚨 SEÑAL DETECTADA!")
            self.strategy.print_signal(signal)
            
            # Registrar señal en log
            self.executor.logger.log_signal(signal)
            
            # Procesar señal
            self.process_signal(signal)
        else:
            print(f"      Sin señales")
    
    def process_signal(self, signal: Dict):
        """
        Procesa una señal de trading
        
        Args:
            signal: Señal generada por la estrategia
        """
        print(f"\n{'='*60}")
        print(f"⚙️  PROCESANDO SEÑAL")
        print(f"{'='*60}\n")
        
        # 1. Calcular position sizing CON SL DINÁMICO
        print("📊 Calculando tamaño de posición...")
        
        # Obtener SL multiplier de la señal (si existe), sino usar default
        sl_multiplier = signal.get('sl_multiplier', 2.0)  # Default 2.0 ATR
        print(f"   📏 SL dinámico: {sl_multiplier} ATR (estrategia: {signal.get('strategy', 'N/A')})")
        
        plan = self.risk_mgr.calculate_position_size(signal, sl_multiplier=sl_multiplier)
        
        # 2. Validar plan
        if not plan:
            ColorPrinter.error("❌ Error calculando plan, señal descartada")
            return
        
        if not plan.get('valid', False):
            ColorPrinter.error("❌ Plan inválido, señal descartada")
            self.risk_mgr.print_position_plan(plan)
            return
        
        # 3. Mostrar plan
        self.risk_mgr.print_position_plan(plan)
        
        # 4. Ejecutar trade (si no es dry run)
        if self.dry_run:
            ColorPrinter.warning("⚠️  DRY RUN - Trade NO ejecutado")
            print("   En modo live, aquí se ejecutaría el trade\n")
        else:
            # Ejecutar directamente sin confirmación
            if not plan:
                ColorPrinter.error("❌ Plan inválido, no se puede ejecutar trade")
                return
            
            results = self.executor.execute_full_trade(plan)
            self.executor.print_execution_summary(results)
            
            if results['entry'].get('success'):
                self.trades_executed += 1
                self.current_trade_plan = plan
                
                # Inicializar tracking MAE/MFE
                self.trade_entry_time = datetime.now()
                self.trade_bars_held = 0
                self.mae_price = plan['entry_price']  # Inicializar con entry
                self.mfe_price = plan['entry_price']
                
                # Registrar posición abierta en orchestrator
                if self.orchestrator and plan:
                    self.orchestrator.register_position_opened(self.bot_id, plan['side'])
                
                # Guardar order IDs para trailing
                if self.current_trade_plan:
                    if results.get('sl') and results['sl'].get('order'):
                        self.current_trade_plan['sl_order_id'] = results['sl']['order']['orderId']
                    if results.get('tp') and results['tp'].get('order'):
                        self.current_trade_plan['tp_order_id'] = results['tp']['order']['orderId']
                
                # Guardar trade plan en archivo
                self.save_trade_plan()
                
                # Resetear trailing manager
                self.trailing_mgr.reset()
                ColorPrinter.success("✅ Trade ejecutado exitosamente")
            else:
                ColorPrinter.error("❌ Error ejecutando trade")
    
    def manage_trailing_stop(self):
        """Gestiona trailing stop progresivo para posición activa"""
        if not self.current_position or not self.current_trade_plan:
            return
        
        try:
            # Obtener precio actual
            df = self.strategy._fetch_klines(self.strategy.TF_TRADE, limit=10)
            if df is None or df.empty:
                print(f"   ⚠️  No se pudieron obtener klines para trailing")
                return
            
            current_price = df['close'].iloc[-1]
            
            pos = self.current_position
            entry_price = float(pos['entryPrice'])
            position_side = self.current_trade_plan['side']
            initial_sl = self.current_trade_plan['stop_loss']
            
            # Actualizar MAE/MFE
            if position_side == 'LONG':
                # MAE = peor precio (más bajo)
                if self.mae_price is None or current_price < self.mae_price:
                    self.mae_price = current_price
                # MFE = mejor precio (más alto)
                if self.mfe_price is None or current_price > self.mfe_price:
                    self.mfe_price = current_price
            else:  # SHORT
                # MAE = peor precio (más alto)
                if self.mae_price is None or current_price > self.mae_price:
                    self.mae_price = current_price
                # MFE = mejor precio (más bajo)
                if self.mfe_price is None or current_price < self.mfe_price:
                    self.mfe_price = current_price
            
            # Incrementar contador de barras
            self.trade_bars_held += 1
            
            # Calcular R actual para debug
            if position_side == 'LONG':
                risk = entry_price - initial_sl
                profit = current_price - entry_price
            else:
                risk = initial_sl - entry_price
                profit = entry_price - current_price
            
            r_current = profit / risk if risk > 0 else 0
            
            print(f"   📊 Trailing Check: Price ${current_price:.2f} | R={r_current:.2f} | Threshold=+1.35R")
            
            # Calcular trailing
            trailing_info = self.trailing_mgr.calculate_trailing_stop(
                entry_price=entry_price,
                current_price=current_price,
                initial_sl=initial_sl,
                side=position_side
            )
            
            if trailing_info is None:
                # Aún no alcanzamos +1.35R
                print(f"   ⏸️  Trailing inactivo (necesita +1.35R, actual: {r_current:.2f}R)")
                return
            
            # Si es la primera vez que activamos trailing, cancelar TP
            if trailing_info['state_changed'] and self.trailing_mgr.should_cancel_tp():
                if 'tp_order_id' in self.current_trade_plan and self.current_trade_plan['tp_order_id']:
                    print(f"   🎯 Cancelando TP para dejar correr el profit...")
                    self.executor.cancel_order(self.current_trade_plan['tp_order_id'])
                    self.current_trade_plan['tp_order_id'] = None
            
            # Obtener SL actual
            current_sl = self.current_trade_plan.get('trailing_sl', initial_sl)
            new_sl = trailing_info['new_sl']
            
            # Solo actualizar si el nuevo SL es mejor
            should_update = False
            if position_side == 'LONG':
                should_update = new_sl > current_sl
            else:  # SHORT
                should_update = new_sl < current_sl
            
            if should_update:
                # Actualizar SL en Binance
                result = self.executor.update_stop_loss(self.current_trade_plan, new_sl)
                
                if result and result['success']:
                    self.current_trade_plan['trailing_sl'] = new_sl
                    self.current_trade_plan['sl_order_id'] = result['order_id']
                    
                    print(f"   📊 Trailing Update:")
                    print(f"      Stage: {trailing_info['stage']}")
                    print(f"      R: {trailing_info['r_multiple']:.2f}")
                    print(f"      ATR: {trailing_info['atr_timeframe']} = ${trailing_info['atr_value']:.4f}")
                    print(f"      Extreme: ${trailing_info['extreme_price']:.2f}")
                    print(f"      SL: ${current_sl:.2f} → ${new_sl:.2f}")
                    
                    # Log en archivo
                    self.executor.logger.log_position_update(
                        contracts=int(float(pos['positionAmt'])),
                        entry_price=entry_price,
                        mark_price=current_price,
                        pnl=float(pos['unRealizedProfit']),
                        pnl_usd=float(pos['unRealizedProfit']) * current_price * 10
                    )
            
        except Exception as e:
            print(f"   ⚠️  Error en trailing: {e}")
    
    def print_stats(self):
        """Imprime estadísticas del bot"""
        print(f"\n{'='*60}")
        print(f"📊 ESTADÍSTICAS DEL BOT")
        print(f"{'='*60}")
        print(f"Señales generadas: {self.signals_generated}")
        print(f"Trades ejecutados: {self.trades_executed}")
        print(f"Trades ganadores: {self.trades_won}")
        print(f"Trades perdedores: {self.trades_lost}")
        if self.trades_executed > 0:
            win_rate = (self.trades_won / self.trades_executed) * 100
            print(f"Win Rate: {win_rate:.1f}%")
        print(f"{'='*60}")

def main():
    """Función principal"""
    
    # Mostrar configuración
    ETHConfig.print_config()
    
    # Preguntar modo
    print(f"\n{'='*60}")
    print("SELECCIONAR MODO:")
    print("1. DRY RUN (prueba, sin ejecutar órdenes)")
    print("2. LIVE (ejecuta órdenes reales)")
    print(f"{'='*60}\n")
    
    mode = input("Selecciona modo (1 o 2): ")
    
    dry_run = mode != '2'
    
    if not dry_run:
        ColorPrinter.error("\n⚠️⚠️⚠️  ADVERTENCIA  ⚠️⚠️⚠️")
        ColorPrinter.error("Vas a ejecutar el bot en MODO LIVE")
        ColorPrinter.error("Se ejecutarán órdenes REALES en tu cuenta")
        confirm = input("\nEscribe 'ENTIENDO' para continuar: ")
        
        if confirm != 'ENTIENDO':
            print("\n❌ Operación cancelada\n")
            return
    
    # Iniciar bot
    bot = ETHBot(dry_run=dry_run)
    bot.start()

if __name__ == '__main__':
    main()
