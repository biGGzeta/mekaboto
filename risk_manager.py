"""
Risk Manager para ETH COIN-M
Calcula tamaño de posición, TP/SL, considera fees y balance
"""
import sys
import os
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, root_dir)

from typing import Dict, Optional
from coinm.eth.config import ETHConfig
from rest_api.coinm_balance import CoinMBalanceManager

class RiskManager:
    """Gestión de riesgo para ETH COIN-M"""
    
    def __init__(self):
        self.config = ETHConfig
        self.balance_mgr = CoinMBalanceManager()
    
    def get_available_balance(self) -> float:
        """
        Obtiene balance disponible en ETH
        
        Returns:
            Balance disponible en ETH
        """
        try:
            balance_info = self.balance_mgr.get_asset_balance(self.config.MARGIN_ASSET)
            
            if balance_info:
                return balance_info['availableBalance']
            
            return 0.0
            
        except Exception as e:
            print(f"Error obteniendo balance: {e}")
            return 0.0
    
    def get_balance_in_usd(self, eth_balance: float, eth_price: float) -> float:
        """
        Convierte balance ETH a USD
        
        Args:
            eth_balance: Balance en ETH
            eth_price: Precio de ETH en USD
            
        Returns:
            Balance en USD
        """
        return eth_balance * eth_price
    
    def calculate_position_size(self, signal: Dict, sl_multiplier: float = 2.0) -> Optional[Dict]:
        """
        Calcula tamaño de posición basado en riesgo
        
        Args:
            signal: Señal de trading con ATR y precio
            sl_multiplier: Multiplicador de ATR para SL (default 2.0)
            
        Returns:
            Dict con sizing o None si no se puede operar
        """
        # Obtener balance disponible
        eth_balance = self.get_available_balance()
        
        print(f"   💰 Balance disponible: {eth_balance:.8f} ETH")
        
        if eth_balance <= 0:
            return {
                'error': 'Balance insuficiente',
                'eth_balance': eth_balance
            }
        
        entry_price = signal['entry_price']
        atr = signal['atr']
        side = signal['type']
        
        # Risk-Reward ratio variable según estrategia
        rr_ratio = signal.get('risk_reward', 2.0)  # Default 2:1 si no está definido
        
        # Balance en USD
        balance_usd = self.get_balance_in_usd(eth_balance, entry_price)
        
        # Cantidad a arriesgar (2% del balance)
        risk_amount_usd = balance_usd * (self.config.RISK_PER_TRADE / 100)
        
        # Stop Loss dinámico usando sl_multiplier de la señal
        stop_distance = atr * sl_multiplier
        
        print(f"   📊 Balance USD: ${balance_usd:.2f}")
        print(f"   💵 Riesgo: ${risk_amount_usd:.2f} ({self.config.RISK_PER_TRADE}%)")
        print(f"   📏 ATR: ${atr:.2f}")
        print(f"   🛑 Stop distance: ${stop_distance:.2f} ({sl_multiplier} ATR)")
        
        # Porcentaje de stop en relación al entry
        stop_percentage = stop_distance / entry_price
        
        # Calcular posición basada en risk
        # En COIN-M: cada contrato = $10 USD notional
        # Position Value (USD) = Contracts × Contract Size
        # Risk (USD) = Position Value × Stop %
        # Contracts = Risk / (Contract Size × Stop %)
        
        params = self.config.get_params()
        if params is None:
            return {'error': 'Parámetros del símbolo no inicializados', 'valid': False}
        
        contract_size = params['contractSize']
        contracts_raw = risk_amount_usd / (contract_size * stop_percentage)
        
        # Ajustar a límites configurados
        max_contracts_by_limit = self.config.MAX_POSITION_SIZE_USD / contract_size
        
        # IMPORTANTE: Limitar por balance disponible con leverage
        # Usar 95% del balance (5% margen para fees, redondeos y seguridad)
        # El riesgo real está controlado por RISK_PER_TRADE = 5%
        max_position_by_balance = (balance_usd * 0.95) * self.config.MAX_LEVERAGE
        max_contracts_by_balance = max_position_by_balance / contract_size
        
        # Usar el menor de los dos límites
        max_contracts = min(max_contracts_by_limit, max_contracts_by_balance)
        
        if contracts_raw > max_contracts:
            contracts_raw = max_contracts
            print(f"   ⚠️  Posición limitada a {max_contracts} contratos (balance disponible)")
        
        # Convertir a contratos válidos
        contracts, error = self.config.validate_quantity(contracts_raw)
        
        if error:
            return {'error': error, 'valid': False}
        
        params = self.config.get_params()
        if contracts is None or (params and contracts < params['minQty']):
            return {
                'error': 'Posición muy pequeña',
                'valid': False,
                'calculated_contracts': contracts_raw,
                'min_required': params['minQty'] if params else 1
            }
        
        # Calcular valores finales
        position_value_usd = self.config.contracts_to_usd(contracts, entry_price)
        
        # Stop Loss
        if side == 'LONG':
            stop_loss = entry_price - stop_distance
        else:  # SHORT
            stop_loss = entry_price + stop_distance
        
        stop_loss = self.config.validate_price(stop_loss)
        
        # Take Profit con RR variable según estrategia
        profit_distance = stop_distance * rr_ratio
        
        if side == 'LONG':
            take_profit = entry_price + profit_distance
        else:  # SHORT
            take_profit = entry_price - profit_distance
        
        take_profit = self.config.validate_price(take_profit)
        
        # Calcular fees
        entry_fee = self.config.calculate_fees(position_value_usd, is_maker=False)
        exit_fee = self.config.calculate_fees(position_value_usd, is_maker=True)
        total_fees = entry_fee + exit_fee
        
        # Profit neto esperado
        gross_profit = (profit_distance / entry_price) * position_value_usd
        net_profit = gross_profit - total_fees
        
        # Validar profit mínimo
        if net_profit < self.config.MIN_PROFIT_USD:
            return {
                'error': f'Profit neto ${net_profit:.2f} menor al mínimo ${self.config.MIN_PROFIT_USD}',
                'calculated_contracts': contracts,
                'net_profit': net_profit
            }
        
        # Margen requerido
        margin_required_usd = position_value_usd / self.config.MAX_LEVERAGE
        margin_required_eth = margin_required_usd / entry_price
        
        print(f"   💼 Posición: ${position_value_usd:.2f}")
        print(f"   🔧 Leverage: {self.config.MAX_LEVERAGE}x")
        print(f"   💰 Margen requerido: {margin_required_eth:.6f} ETH (${margin_required_usd:.2f})")
        print(f"   ✓ Balance disponible: {eth_balance:.6f} ETH")
        
        # Verificar que tenemos suficiente margen
        if margin_required_eth > eth_balance:
            return {
                'error': 'Margen insuficiente',
                'margin_required_eth': margin_required_eth,
                'available_eth': eth_balance
            }
        
        # Calcular trailing stop para 50% de la posición
        contracts_partial = contracts * (self.config.PARTIAL_TP_PERCENT / 100)
        contracts_trailing = contracts - contracts_partial
        
        # Ajustar ambos a step size
        contracts_partial, _ = self.config.validate_quantity(contracts_partial)
        contracts_trailing, _ = self.config.validate_quantity(contracts_trailing)
        
        return {
            'valid': True,
            'side': side,
            'entry_price': entry_price,
            'contracts': contracts,
            'contracts_partial_tp': contracts_partial,
            'contracts_trailing': contracts_trailing,
            'position_value_usd': position_value_usd,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'stop_distance': stop_distance,
            'profit_distance': profit_distance,
            'atr': atr,
            'risk_reward': rr_ratio,
            'strategy': signal.get('strategy', 'UNKNOWN'),
            'risk_amount_usd': risk_amount_usd,
            'risk_percentage': self.config.RISK_PER_TRADE,
            'leverage': self.config.MAX_LEVERAGE,
            'margin_required_eth': margin_required_eth,
            'margin_required_usd': margin_required_usd,
            'balance_eth': eth_balance,
            'balance_usd': balance_usd,
            'entry_fee': entry_fee,
            'exit_fee': exit_fee,
            'total_fees': total_fees,
            'gross_profit': gross_profit,
            'net_profit': net_profit,
            'trailing_atr_multiplier': self.config.ATR_MULTIPLIER
        }
    
    def print_position_plan(self, plan: Dict):
        """Imprime plan de posición formateado"""
        if 'error' in plan:
            print(f"\n❌ ERROR: {plan['error']}")
            return
        
        print(f"\n{'='*60}")
        print(f"📋 PLAN DE POSICIÓN")
        print(f"{'='*60}")
        
        # Mostrar estrategia y RR
        strategy_name = plan.get('strategy', 'UNKNOWN')
        rr_ratio = plan.get('risk_reward', 2.0)
        print(f"\n🎯 Estrategia: {strategy_name}")
        print(f"   Risk:Reward = 1:{rr_ratio}")
        
        print(f"\n💼 Balance:")
        print(f"   Disponible: {plan['balance_eth']:.4f} ETH (${plan['balance_usd']:,.2f})")
        print(f"   Riesgo: {plan['risk_percentage']}% = ${plan['risk_amount_usd']:.2f}")
        
        print(f"\n📊 Posición:")
        print(f"   Lado: {plan['side']}")
        print(f"   Contratos: {plan['contracts']}")
        print(f"   Valor: ${plan['position_value_usd']:,.2f}")
        print(f"   Leverage: {plan['leverage']}x")
        print(f"   Margen: {plan['margin_required_eth']:.4f} ETH (${plan['margin_required_usd']:.2f})")
        
        print(f"\n💰 Precios:")
        print(f"   Entry: ${plan['entry_price']:,.2f}")
        print(f"   Stop Loss: ${plan['stop_loss']:,.2f} (${plan['stop_distance']:.2f})")
        print(f"   Take Profit: ${plan['take_profit']:,.2f} (${plan['profit_distance']:.2f})")
        
        print(f"\n📈 Gestión de Salida:")
        print(f"   {self.config.PARTIAL_TP_PERCENT}% ({plan['contracts_partial_tp']} contratos) → TP fijo ${plan['take_profit']:,.2f}")
        print(f"   {self.config.TRAILING_PERCENT}% ({plan['contracts_trailing']} contratos) → Trailing (ATR x {plan['trailing_atr_multiplier']})")
        
        print(f"\n💵 Fees y Profit:")
        print(f"   Entry Fee: ${plan['entry_fee']:.2f}")
        print(f"   Exit Fee: ${plan['exit_fee']:.2f}")
        print(f"   Total Fees: ${plan['total_fees']:.2f}")
        print(f"   Profit Bruto: ${plan['gross_profit']:.2f}")
        print(f"   Profit Neto: ${plan['net_profit']:.2f}")
        
        print(f"\n📊 ATR: ${plan['atr']:.2f}")
        print(f"{'='*60}\n")

if __name__ == '__main__':
    from utils.helpers import ColorPrinter
    from coinm.eth.oldstrategy import RSIStrategy
    
    ColorPrinter.header("TEST RISK MANAGER ETH")
    
    # Primero generar una señal
    strategy = RSIStrategy()
    
    print("\n🔍 Generando señal de trading...")
    signal = strategy.generate_signal()
    
    if not signal:
        # Crear señal de prueba
        ColorPrinter.warning("⚠️  No hay señal real, usando señal de prueba")
        current_price = ETHConfig.get_current_price()
        signal = {
            'type': 'LONG',
            'entry_price': current_price,
            'atr': 30.0,  # ATR de prueba
            'rsi': 32.0
        }
    else:
        ColorPrinter.success("✅ Señal generada")
        strategy.print_signal(signal)
    
    # Calcular position sizing
    print("\n📊 Calculando tamaño de posición...")
    risk_mgr = RiskManager()
    plan = risk_mgr.calculate_position_size(signal)
    
    if not plan:
        ColorPrinter.error("❌ Error calculando plan")
        sys.exit(1)
    
    # Mostrar plan
    risk_mgr.print_position_plan(plan)
    
    if plan.get('valid', False):
        ColorPrinter.success("✅ Plan de posición válido")
    else:
        ColorPrinter.error("❌ Plan de posición inválido")
