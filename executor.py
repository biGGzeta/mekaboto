"""
Executor para ejecutar órdenes en COIN-M ETH
Maneja entrada, TP parcial y trailing stop
"""
import sys
import os
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, root_dir)

from typing import Dict, Optional
import time
from datetime import datetime
from coinm.eth.config import ETHConfig
from rest_api.coinm_balance import CoinMBalanceManager
from coinm.trade_logger import TradeLogger

class OrderExecutor:
    """Ejecuta órdenes en COIN-M"""
    
    def __init__(self):
        self.config = ETHConfig
        self.balance_mgr = CoinMBalanceManager()
        self.active_positions = {}
        self.active_orders = {}
        self.leverage_verified = False
        self.logger = TradeLogger(symbol=ETHConfig.SYMBOL)
    
    def verify_and_set_leverage(self):
        """
        Verifica y establece el leverage correcto para el símbolo
        
        Returns:
            bool: True si está correcto o se estableció exitosamente
        """
        if self.leverage_verified:
            return True
        
        try:
            # Obtener leverage actual (ya no usamos BASE_URL directamente)
            response = self.balance_mgr._send_request('GET', '/dapi/v1/positionSide/dual', {})
            
            # Establecer leverage
            leverage_params = {
                'symbol': self.config.SYMBOL,
                'leverage': self.config.MAX_LEVERAGE,
                'timestamp': int(time.time() * 1000),
                'recvWindow': 5000
            }
            
            print(f"\n⚙️  Configurando leverage...")
            response = self.balance_mgr._send_request('POST', '/dapi/v1/leverage', leverage_params)
            
            if 'leverage' in response:
                current_leverage = response['leverage']
                print(f"   ✅ Leverage establecido: {current_leverage}x")
                self.leverage_verified = True
                return True
            else:
                print(f"   ⚠️  Respuesta inesperada: {response}")
                return False
                
        except Exception as e:
            error_str = str(e)
            # Si el error es que ya está en el leverage correcto, está bien
            if 'No need to change leverage' in error_str or '-4028' in error_str:
                print(f"   ℹ️  Leverage ya configurado correctamente")
                self.leverage_verified = True
                return True
            else:
                print(f"   ⚠️  Error configurando leverage: {e}")
                return False
    
    def execute_entry(self, plan: Dict) -> Dict:
        """
        Ejecuta orden de entrada
        
        Args:
            plan: Plan de posición del RiskManager
            
        Returns:
            Dict con resultado de la ejecución
        """
        if not plan.get('valid'):
            return {'error': 'Plan inválido'}
        
        try:
            side = 'BUY' if plan['side'] == 'LONG' else 'SELL'
            
            # Preparar orden MARKET
            position_side = 'LONG' if plan['side'] == 'LONG' else 'SHORT'
            order_params = {
                'symbol': self.config.SYMBOL,
                'side': side,
                'type': 'MARKET',
                'positionSide': position_side,
                'quantity': int(plan['contracts']),
                'timestamp': int(time.time() * 1000),
                'recvWindow': 5000
            }
            
            print(f"\n{'='*60}")
            print(f"📤 EJECUTANDO ORDEN DE ENTRADA")
            print(f"{'='*60}")
            print(f"   Symbol: {order_params['symbol']}")
            print(f"   Side: {side}")
            print(f"   Quantity: {order_params['quantity']} contratos")
            print(f"   Tipo: MARKET")
            print(f"   Valor: ~${plan['position_value_usd']:,.2f}")
            
            # Ejecutar orden real
            result = self.balance_mgr._send_request('POST', '/dapi/v1/order', order_params)
            
            # # Para testing, simular resultado
            # result = {
            #     'orderId': int(time.time()),
            #     'symbol': self.config.SYMBOL,
            #     'status': 'FILLED',
            #     'side': side,
            #     'type': 'MARKET',
            #     'origQty': str(order_params['quantity']),
            #     'executedQty': str(order_params['quantity']),
            #     'avgPrice': str(plan['entry_price']),
            #     'updateTime': int(time.time() * 1000)
            # }
            
            print(f"\n✅ Orden ejecutada:")
            print(f"   Order ID: {result['orderId']}")
            print(f"   Status: {result['status']}")
            print(f"   Avg Price: ${float(result['avgPrice']):,.2f}")
            print(f"{'='*60}\n")
            
            # Registrar trade en log
            self.logger.log_trade_entry(
                side=plan['side'],
                contracts=int(plan['contracts']),
                price=float(result['avgPrice']),
                stop_loss=plan['stop_loss'],
                take_profit=plan['take_profit'],
                leverage=self.config.MAX_LEVERAGE,
                order_id=result['orderId']
            )
            
            return {
                'success': True,
                'order': result,
                'plan': plan
            }
            
        except Exception as e:
            print(f"❌ Error ejecutando entrada: {e}")
            self.logger.log_error('ORDER', f"Error ejecutando entrada: {e}", {'plan': plan})
            return {'error': str(e)}
    
    def place_stop_loss(self, plan: Dict, entry_result: Dict) -> Dict:
        """
        Coloca Stop Loss para toda la posición
        
        Args:
            plan: Plan de posición
            entry_result: Resultado de orden de entrada
            
        Returns:
            Dict con resultado
        """
        try:
            # Lado opuesto para cerrar
            side = 'SELL' if plan['side'] == 'LONG' else 'BUY'
            stop_side = 'LONG' if plan['side'] == 'LONG' else 'SHORT'
            
            order_params = {
                'symbol': self.config.SYMBOL,
                'side': side,
                'type': 'STOP_MARKET',
                'positionSide': stop_side,
                'stopPrice': plan['stop_loss'],
                'quantity': int(plan['contracts']),
                'closePosition': True,      # cierra toda la posición
                'reduceOnly': True,         # evita invertir/sobreapilar
                'timestamp': int(time.time() * 1000),
                'recvWindow': 5000
            }
            
            print(f"📤 Colocando Stop Loss:")
            print(f"   Price: ${plan['stop_loss']:,.2f}")
            print(f"   Quantity: {plan['contracts']} contratos")
            
            # Ejecutar orden real
            result = self.balance_mgr._send_request('POST', '/dapi/v1/order', order_params)
            
            # result = {
            #     'orderId': int(time.time()) + 1,
            #     'symbol': self.config.SYMBOL,
            #     'status': 'NEW',
            #     'type': 'STOP_MARKET',
            #     'stopPrice': str(plan['stop_loss'])
            # }
            
            print(f"   ✅ Stop Loss colocado (ID: {result['orderId']})\n")
            
            return {'success': True, 'order': result}
            
        except Exception as e:
            print(f"❌ Error colocando SL: {e}")
            return {'error': str(e)}
    
    def place_take_profit_partial(self, plan: Dict) -> Dict:
        """
        Coloca Take Profit para el 50% de la posición
        
        Args:
            plan: Plan de posición
            
        Returns:
            Dict con resultado
        """
        try:
            side = 'SELL' if plan['side'] == 'LONG' else 'BUY'
            position_side = 'LONG' if plan['side'] == 'LONG' else 'SHORT'
            
            order_params = {
                'symbol': self.config.SYMBOL,
                'side': side,
                'type': 'TAKE_PROFIT_MARKET',
                'positionSide': position_side,
                'stopPrice': plan['take_profit'],
                'quantity': int(plan['contracts_partial_tp']),
                'closePosition': False,
                'reduceOnly': True,         # evita abrir/invertir
                'timestamp': int(time.time() * 1000),
                'recvWindow': 5000
            }
            
            print(f"📤 Colocando Take Profit parcial ({self.config.PARTIAL_TP_PERCENT}%):")
            print(f"   Price: ${plan['take_profit']:,.2f}")
            print(f"   Quantity: {plan['contracts_partial_tp']} contratos")
            
            # Ejecutar orden real
            result = self.balance_mgr._send_request('POST', '/dapi/v1/order', order_params)
            
            # result = {
            #     'orderId': int(time.time()) + 2,
            #     'symbol': self.config.SYMBOL,
            #     'status': 'NEW',
            #     'type': 'TAKE_PROFIT_MARKET',
            #     'stopPrice': str(plan['take_profit'])
            # }
            
            print(f"   ✅ TP parcial colocado (ID: {result['orderId']})\n")
            
            return {'success': True, 'order': result}
            
        except Exception as e:
            print(f"❌ Error colocando TP parcial: {e}")
            return {'error': str(e)}
    
    def execute_full_trade(self, plan: Dict) -> Dict:
        """
        Ejecuta trade completo: entrada + SL + TP parcial
        
        Args:
            plan: Plan de posición del RiskManager
            
        Returns:
            Dict con todos los resultados
        """
        print(f"\n{'#'*60}")
        print(f"{'#'*60}")
        print(f"🚀 EJECUTANDO TRADE COMPLETO")
        print(f"{'#'*60}")
        print(f"{'#'*60}\n")
        
        # Verificar y establecer leverage antes de ejecutar
        if not self.verify_and_set_leverage():
            print("⚠️  Advertencia: No se pudo verificar leverage")
        
        results = {
            'timestamp': datetime.now(),
            'plan': plan
        }
        
        # 1. Ejecutar entrada
        entry_result = self.execute_entry(plan)
        results['entry'] = entry_result
        
        if not entry_result.get('success'):
            print("❌ Error en entrada, abortando trade")
            return results
        
        # Esperar un momento
        time.sleep(0.5)
        
        # 2. Colocar Stop Loss
        sl_result = self.place_stop_loss(plan, entry_result)
        results['stop_loss'] = sl_result
        
        if not sl_result.get('success'):
            print("⚠️  Error colocando SL, pero entrada ejecutada!")
        
        # Esperar un momento
        time.sleep(0.5)
        
        # 3. Colocar Take Profit parcial
        tp_result = self.place_take_profit_partial(plan)
        results['take_profit_partial'] = tp_result
        
        if not tp_result.get('success'):
            print("⚠️  Error colocando TP parcial")
        
        print(f"\n{'#'*60}")
        print(f"✅ TRADE EJECUTADO COMPLETAMENTE")
        print(f"{'#'*60}")
        print(f"\n📝 Resumen:")
        print(f"   Entrada: {'✅' if entry_result.get('success') else '❌'}")
        print(f"   Stop Loss: {'✅' if sl_result.get('success') else '❌'}")
        print(f"   TP Parcial: {'✅' if tp_result.get('success') else '❌'}")
        print(f"\n⚠️  NOTA: {plan['contracts_trailing']} contratos quedan para trailing stop")
        print(f"   Trailing será manejado por el monitor de posiciones")
        print(f"{'#'*60}\n")
        
        return results
    
    def get_current_position(self) -> Optional[Dict]:
        """
        Obtiene posición actual en el símbolo
        
        Returns:
            Dict con posición o None
        """
        try:
            positions = self.balance_mgr.get_open_positions_coinm()
            
            for pos in positions:
                if pos['symbol'] == self.config.SYMBOL:
                    return pos
            
            return None
            
        except Exception as e:
            print(f"Error obteniendo posición: {e}")
            return None
    
    def print_execution_summary(self, results: Dict):
        """Imprime resumen de ejecución"""
        print(f"\n{'='*60}")
        print(f"📊 RESUMEN DE EJECUCIÓN")
        print(f"{'='*60}")
        print(f"Timestamp: {results['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        if results['entry'].get('success'):
            order = results['entry']['order']
            plan = results['plan']
            
            print(f"\n✅ Entrada Ejecutada:")
            print(f"   Order ID: {order['orderId']}")
            print(f"   Side: {plan['side']}")
            print(f"   Contracts: {plan['contracts']}")
            print(f"   Price: ${plan['entry_price']:,.2f}")
            print(f"   Value: ${plan['position_value_usd']:,.2f}")
            
            if results['stop_loss'].get('success'):
                print(f"\n✅ Stop Loss:")
                print(f"   Order ID: {results['stop_loss']['order']['orderId']}")
                print(f"   Price: ${plan['stop_loss']:,.2f}")
                print(f"   Contracts: {plan['contracts']}")
            
            if results['take_profit_partial'].get('success'):
                print(f"\n✅ Take Profit Parcial:")
                print(f"   Order ID: {results['take_profit_partial']['order']['orderId']}")
                print(f"   Price: ${plan['take_profit']:,.2f}")
                print(f"   Contracts: {plan['contracts_partial_tp']}")
            
            print(f"\n📊 Riesgo/Reward:")
            print(f"   Riesgo: ${plan['risk_amount_usd']:.2f} ({plan['risk_percentage']}%)")
            print(f"   Reward esperado: ${plan['net_profit']:.2f}")
            print(f"   R:R: 1:{plan['net_profit']/plan['risk_amount_usd']:.2f}")
        
        print(f"{'='*60}\n")


    def cancel_order(self, order_id: int) -> bool:
        """
        Cancela una orden específica
        
        Args:
            order_id: ID de la orden a cancelar
            
        Returns:
            True si se canceló correctamente
        """
        try:
            params = {
                'symbol': self.config.SYMBOL,
                'orderId': order_id,
                'timestamp': int(time.time() * 1000),
                'recvWindow': 5000
            }
            
            result = self.balance_mgr._send_request('DELETE', '/dapi/v1/order', params)
            
            print(f"   ✅ Orden {order_id} cancelada")
            return True
            
        except Exception as e:
            print(f"   ⚠️  Error cancelando orden {order_id}: {e}")
            return False
    
    def update_stop_loss(self, plan: Dict, new_sl_price: float) -> Optional[Dict]:
        """
        Actualiza el stop loss de una posición activa
        
        Args:
            plan: Plan de posición actual
            new_sl_price: Nuevo precio de SL
            
        Returns:
            Dict con resultado o None
        """
        try:
            # Primero cancelar SL actual si existe
            if 'sl_order_id' in plan and plan['sl_order_id']:
                self.cancel_order(plan['sl_order_id'])
            
            # Crear nuevo SL
            side = 'BUY' if plan['side'] == 'SHORT' else 'SELL'
            position_side = plan['side']
            
            order_params = {
                'symbol': self.config.SYMBOL,
                'side': side,
                'type': 'STOP_MARKET',
                'positionSide': position_side,
                'stopPrice': new_sl_price,
                'closePosition': True,      # cierra toda la posición
                'reduceOnly': True,         # evita invertir
                'timestamp': int(time.time() * 1000),
                'recvWindow': 5000
            }
            
            result = self.balance_mgr._send_request('POST', '/dapi/v1/order', order_params)
            
            print(f"   ✅ SL actualizado a ${new_sl_price:.4f} (Order ID: {result['orderId']})")
            
            return {
                'success': True,
                'order_id': result['orderId'],
                'new_sl': new_sl_price
            }
            
        except Exception as e:
            print(f"   ❌ Error actualizando SL: {e}")
            return None


if __name__ == '__main__':
    from utils.helpers import ColorPrinter
    from coinm.eth.oldstrategy import RSIStrategy
    from risk_manager import RiskManager
    
    ColorPrinter.header("TEST EXECUTOR ETH")
    
    # Generar señal
    strategy = RSIStrategy()
    signal = strategy.generate_signal()
    
    if not signal:
        ColorPrinter.warning("⚠️  Sin señal, creando señal de prueba")
        current_price = ETHConfig.get_current_price()
        signal = {
            'type': 'LONG',
            'entry_price': current_price,
            'atr': 30.0,
            'rsi': 32.0
        }
    
    # Calcular risk
    risk_mgr = RiskManager()
    plan = risk_mgr.calculate_position_size(signal)
    
    if not plan or not plan.get('valid', False):
        ColorPrinter.error("❌ Plan inválido")
        if plan:
            risk_mgr.print_position_plan(plan)
    else:
        # Mostrar plan
        risk_mgr.print_position_plan(plan)
        
        # Preguntar confirmación
        print("\n⚠️  MODO TEST - Las órdenes NO se ejecutarán realmente")
        confirm = input("\n¿Ejecutar trade de prueba? (s/n): ")
        
        if confirm.lower() == 's':
            executor = OrderExecutor()
            results = executor.execute_full_trade(plan)
            
            executor.print_execution_summary(results)
            ColorPrinter.success("✅ Test completado")
        else:
            ColorPrinter.info("ℹ️  Ejecución cancelada")
