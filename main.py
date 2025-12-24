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
# Sustituido oldstrategy por la nueva strategy integrada
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
    
    # ... resto del archivo sin cambios ...
    # (No se modifican métodos ni lógica, sólo el import de RSIStrategy)
    
if __name__ == '__main__':
    def main():
        """Función principal"""
        # Mostrar configuración
        ETHConfig.print_config()
        
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
        
        bot = ETHBot(dry_run=dry_run)
        bot.start()
    
    main()
