"""
Configuración específica para ETH COIN-M Futures
"""
import sys
import os
# Agregar el directorio raíz al path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, root_dir)

import requests

class ETHConfig:
    """Configuración para operar ETHUSD_PERP en COIN-M"""
    
    # Símbolo
    SYMBOL = 'ETHUSD_PERP'
    MARGIN_ASSET = 'ETH'  # Margen en ETH
    
    # Timeframe
    TIMEFRAME = '1m'  # Velas de 1 minuto
    
    # Estrategia RSI(15m) + EMA + ADX
    RSI_PERIOD = 14      # Período RSI (14 es estándar)
    RSI_OVERBOUGHT = 60  # SHORT cuando RSI(15m) > 60
    RSI_OVERSOLD = 40    # LONG cuando RSI(15m) < 40
    
    # Risk Management
    RISK_PER_TRADE = 2.5  # 2.5% del balance por operación
    MAX_LEVERAGE = 10      # Leverage máximo para ETH
    
    # TP/SL Ratio
    TP_RATIO = 2.0  # Risk:Reward 1:2 (en lugar de 1:1.5)
    PARTIAL_TP_PERCENT = 100  # 100% de la posición con TP fijo
    TRAILING_PERCENT = 0      # Sin trailing stop
    
    # ATR para trailing
    ATR_PERIOD = 14
    ATR_MULTIPLIER = 4.0  # Trailing stop = ATR * 4 (en lugar de 2)
    
    # Fees (COIN-M)
    MAKER_FEE = 0.0002   # 0.02% maker
    TAKER_FEE = 0.0004   # 0.04% taker
    
    # Control de posición
    MAX_POSITION_SIZE_USD = 10000  # Máximo $10k por posición
    MIN_PROFIT_USD = 0.10  # Profit mínimo en USD ($0.10)
    
    # Parámetros del símbolo (se obtienen dinámicamente)
    _symbol_params = None
    _exchange_info = None
    
    @classmethod
    def initialize(cls):
        """Inicializa parámetros del símbolo desde Binance"""
        if cls._symbol_params is None:
            # Obtener parámetros de COIN-M
            try:
                url = "https://dapi.binance.com/dapi/v1/exchangeInfo"
                response = requests.get(url, timeout=10)
                data = response.json()
                
                # Buscar ETHUSD_PERP
                for symbol_info in data['symbols']:
                    if symbol_info['symbol'] == cls.SYMBOL:
                        cls._symbol_params = cls._parse_symbol_info(symbol_info)
                        break
                
                if cls._symbol_params is None:
                    raise ValueError(f"Símbolo {cls.SYMBOL} no encontrado")
                    
            except Exception as e:
                print(f"Error obteniendo parámetros: {e}")
                # Valores por defecto
                cls._symbol_params = {
                    'pricePrecision': 2,
                    'quantityPrecision': 0,
                    'minQty': 1,
                    'maxQty': 100000,
                    'stepSize': 1,
                    'minNotional': 0,  # COIN-M no tiene min notional en USD
                    'tickSize': 0.1,
                    'contractSize': 10,  # Cada contrato = 10 USD
                }
    
    @classmethod
    def _parse_symbol_info(cls, symbol_info):
        """Parse información del símbolo"""
        params = {
            'pricePrecision': symbol_info.get('pricePrecision', 2),
            'quantityPrecision': symbol_info.get('quantityPrecision', 0),
            'contractSize': symbol_info.get('contractSize', 10),
        }
        
        # Parsear filtros
        for filter_obj in symbol_info.get('filters', []):
            filter_type = filter_obj['filterType']
            
            if filter_type == 'LOT_SIZE':
                params['minQty'] = float(filter_obj['minQty'])
                params['maxQty'] = float(filter_obj['maxQty'])
                params['stepSize'] = float(filter_obj['stepSize'])
            
            elif filter_type == 'PRICE_FILTER':
                params['minPrice'] = float(filter_obj['minPrice'])
                params['maxPrice'] = float(filter_obj['maxPrice'])
                params['tickSize'] = float(filter_obj['tickSize'])
            
            elif filter_type == 'MIN_NOTIONAL':
                params['minNotional'] = float(filter_obj.get('notional', 0))
        
        return params
    
    @classmethod
    def get_params(cls):
        """Obtiene parámetros del símbolo"""
        if cls._symbol_params is None:
            cls.initialize()
        return cls._symbol_params
    
    @classmethod
    def validate_quantity(cls, quantity):
        """Valida y ajusta cantidad según step size"""
        params = cls.get_params()
        if params is None:
            return None, "Parámetros del símbolo no inicializados"
        
        # Ajustar a step size
        from decimal import Decimal, ROUND_DOWN
        qty_decimal = Decimal(str(quantity))
        step_decimal = Decimal(str(params['stepSize']))
        
        adjusted = float((qty_decimal / step_decimal).quantize(Decimal('1'), rounding=ROUND_DOWN) * step_decimal)
        
        # Verificar límites
        if adjusted < params['minQty']:
            return None, f"Cantidad {adjusted} menor al mínimo {params['minQty']}"
        
        if adjusted > params['maxQty']:
            return None, f"Cantidad {adjusted} mayor al máximo {params['maxQty']}"
        
        return adjusted, None
    
    @classmethod
    def validate_price(cls, price):
        """Valida y ajusta precio según tick size"""
        params = cls.get_params()
        if params is None:
            return price
        
        from decimal import Decimal, ROUND_DOWN
        price_decimal = Decimal(str(price))
        tick_decimal = Decimal(str(params['tickSize']))
        
        adjusted = float((price_decimal / tick_decimal).quantize(Decimal('1'), rounding=ROUND_DOWN) * tick_decimal)
        
        return adjusted
    
    @classmethod
    def contracts_to_usd(cls, contracts, price):
        """Convierte contratos a valor USD"""
        params = cls.get_params()
        if params is None:
            return 0
        return contracts * params['contractSize']
    
    @classmethod
    def usd_to_contracts(cls, usd_value, price):
        """Convierte USD a número de contratos"""
        params = cls.get_params()
        if params is None:
            return None, "Parámetros del símbolo no inicializados"
        contracts = usd_value / params['contractSize']
        
        # Ajustar a step size
        adjusted, error = cls.validate_quantity(contracts)
        return adjusted, error
    
    @classmethod
    def get_current_price(cls):
        """Obtiene precio actual de mercado"""
        try:
            url = "https://dapi.binance.com/dapi/v1/ticker/price"
            response = requests.get(url, params={'symbol': cls.SYMBOL}, timeout=10)
            data = response.json()
            # COIN-M puede devolver lista o dict
            if isinstance(data, list):
                data = data[0]
            return float(data['price'])
        except Exception as e:
            print(f"Error obteniendo precio: {e}")
            return None
    
    @classmethod
    def calculate_fees(cls, position_value_usd, is_maker=False):
        """Calcula fees en USD"""
        fee_rate = cls.MAKER_FEE if is_maker else cls.TAKER_FEE
        return position_value_usd * fee_rate
    
    @classmethod
    def print_config(cls):
        """Imprime configuración actual"""
        cls.initialize()
        
        print(f"\n{'='*60}")
        print(f"Configuración ETH COIN-M Futures")
        print(f"{'='*60}")
        print(f"\n📊 Símbolo: {cls.SYMBOL}")
        print(f"   Margen: {cls.MARGIN_ASSET}")
        if cls._symbol_params:
            print(f"   Contract Size: {cls._symbol_params['contractSize']} USD")
        else:
            print(f"   Contract Size: N/A (no inicializado)")
        
        print(f"\n⏱️  Estrategia:")
        print(f"   Timeframe: {cls.TIMEFRAME}")
        print(f"   RSI Period: {cls.RSI_PERIOD}")
        print(f"   RSI Overbought: >{cls.RSI_OVERBOUGHT} (SHORT)")
        print(f"   RSI Oversold: <{cls.RSI_OVERSOLD} (LONG)")
        
        print(f"\n💰 Risk Management:")
        print(f"   Riesgo por trade: {cls.RISK_PER_TRADE}%")
        print(f"   Max Leverage: {cls.MAX_LEVERAGE}x")
        print(f"   TP Ratio: 1:{cls.TP_RATIO}")
        
        print(f"\n📈 Take Profit:")
        print(f"   {cls.PARTIAL_TP_PERCENT}% con TP fijo")
        print(f"   {cls.TRAILING_PERCENT}% con trailing (ATR x {cls.ATR_MULTIPLIER})")
        
        print(f"\n💵 Fees:")
        print(f"   Maker: {cls.MAKER_FEE*100}%")
        print(f"   Taker: {cls.TAKER_FEE*100}%")
        
        print(f"\n📏 Parámetros del Símbolo:")
        if cls._symbol_params:
            print(f"   Precio: {cls._symbol_params['pricePrecision']} decimales")
            print(f"   Cantidad: {cls._symbol_params['quantityPrecision']} decimales")
            print(f"   Min Qty: {cls._symbol_params['minQty']}")
            print(f"   Step Size: {cls._symbol_params['stepSize']}")
            print(f"   Tick Size: {cls._symbol_params['tickSize']}")
        else:
            print(f"   ⚠️  Parámetros no inicializados")
        
        # Precio actual
        current_price = cls.get_current_price()
        if current_price:
            print(f"\n💲 Precio Actual: ${current_price:,.2f}")

# Inicializar al importar
ETHConfig.initialize()

if __name__ == '__main__':
    ETHConfig.print_config()
