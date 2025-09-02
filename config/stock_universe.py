# config/stock_universe.py
"""
Stock Universe Configuration
===========================
Curated list of stocks optimized for Wyckoff trading methodology
Focus on liquid, volatile stocks with clear institutional patterns
"""

from typing import Dict, List, Set
from dataclasses import dataclass

@dataclass
class StockInfo:
    """Information about a tradeable stock"""
    symbol: str
    name: str
    sector: str
    market_cap: str  # Large, Mid, Small
    avg_volume: int  # Average daily volume
    volatility: str  # High, Medium, Low
    wyckoff_friendly: bool  # Good for Wyckoff analysis
    institutional_activity: str  # High, Medium, Low

class StockUniverse:
    """
    Curated stock universe for Wyckoff trading
    Stocks selected based on:
    - High liquidity (>1M daily volume)
    - Clear institutional patterns
    - Good price/volume relationships
    - Diverse sector representation
    """
    
    # Core Technology Stocks - High institutional activity
    TECHNOLOGY = {
        'AAPL': StockInfo('AAPL', 'Apple Inc.', 'Technology', 'Large', 50000000, 'Medium', True, 'High'),
        'MSFT': StockInfo('MSFT', 'Microsoft Corporation', 'Technology', 'Large', 25000000, 'Medium', True, 'High'),
        'GOOGL': StockInfo('GOOGL', 'Alphabet Inc.', 'Technology', 'Large', 20000000, 'Medium', True, 'High'),
        'META': StockInfo('META', 'Meta Platforms Inc.', 'Technology', 'Large', 15000000, 'High', True, 'High'),
        'NVDA': StockInfo('NVDA', 'NVIDIA Corporation', 'Technology', 'Large', 30000000, 'High', True, 'High'),
        'AMZN': StockInfo('AMZN', 'Amazon.com Inc.', 'Technology', 'Large', 25000000, 'Medium', True, 'High'),
        'NFLX': StockInfo('NFLX', 'Netflix Inc.', 'Technology', 'Large', 5000000, 'High', True, 'High'),
        'CRM': StockInfo('CRM', 'Salesforce Inc.', 'Technology', 'Large', 3000000, 'High', True, 'Medium'),
        'ORCL': StockInfo('ORCL', 'Oracle Corporation', 'Technology', 'Large', 15000000, 'Medium', True, 'Medium'),
        'AMD': StockInfo('AMD', 'Advanced Micro Devices', 'Technology', 'Large', 40000000, 'High', True, 'High'),
    }
    
    # Financial Sector - Interest rate sensitive, institutional heavy
    FINANCIAL = {
        'JPM': StockInfo('JPM', 'JPMorgan Chase & Co.', 'Financial', 'Large', 12000000, 'Medium', True, 'High'),
        'BAC': StockInfo('BAC', 'Bank of America Corp.', 'Financial', 'Large', 40000000, 'Medium', True, 'High'),
        'WFC': StockInfo('WFC', 'Wells Fargo & Company', 'Financial', 'Large', 25000000, 'Medium', True, 'High'),
        'GS': StockInfo('GS', 'Goldman Sachs Group', 'Financial', 'Large', 2000000, 'High', True, 'High'),
        'MS': StockInfo('MS', 'Morgan Stanley', 'Financial', 'Large', 8000000, 'High', True, 'High'),
        'C': StockInfo('C', 'Citigroup Inc.', 'Financial', 'Large', 15000000, 'High', True, 'Medium'),
        'BRK-B': StockInfo('BRK-B', 'Berkshire Hathaway Inc.', 'Financial', 'Large', 3000000, 'Low', True, 'High'),
        'V': StockInfo('V', 'Visa Inc.', 'Financial', 'Large', 6000000, 'Medium', True, 'High'),
        'MA': StockInfo('MA', 'Mastercard Inc.', 'Financial', 'Large', 3000000, 'Medium', True, 'High'),
        'AXP': StockInfo('AXP', 'American Express Company', 'Financial', 'Large', 2500000, 'Medium', True, 'Medium'),
    }
    
    # Healthcare - Defensive with growth potential
    HEALTHCARE = {
        'JNJ': StockInfo('JNJ', 'Johnson & Johnson', 'Healthcare', 'Large', 7000000, 'Low', True, 'Medium'),
        'PFE': StockInfo('PFE', 'Pfizer Inc.', 'Healthcare', 'Large', 25000000, 'Medium', True, 'Medium'),
        'UNH': StockInfo('UNH', 'UnitedHealth Group Inc.', 'Healthcare', 'Large', 3000000, 'Medium', True, 'High'),
        'ABBV': StockInfo('ABBV', 'AbbVie Inc.', 'Healthcare', 'Large', 6000000, 'Medium', True, 'Medium'),
        'TMO': StockInfo('TMO', 'Thermo Fisher Scientific', 'Healthcare', 'Large', 1500000, 'Medium', True, 'Medium'),
        'ABT': StockInfo('ABT', 'Abbott Laboratories', 'Healthcare', 'Large', 5000000, 'Medium', True, 'Medium'),
        'LLY': StockInfo('LLY', 'Eli Lilly and Company', 'Healthcare', 'Large', 2500000, 'Medium', True, 'High'),
        'MRK': StockInfo('MRK', 'Merck & Co. Inc.', 'Healthcare', 'Large', 10000000, 'Medium', True, 'Medium'),
        'MDT': StockInfo('MDT', 'Medtronic plc', 'Healthcare', 'Large', 4000000, 'Medium', True, 'Low'),
        'GILD': StockInfo('GILD', 'Gilead Sciences Inc.', 'Healthcare', 'Large', 6000000, 'Medium', True, 'Low'),
    }
    
    # Consumer Discretionary - Economic cycle sensitive
    CONSUMER_DISCRETIONARY = {
        'TSLA': StockInfo('TSLA', 'Tesla Inc.', 'Consumer Discretionary', 'Large', 75000000, 'High', True, 'High'),
        'HD': StockInfo('HD', 'Home Depot Inc.', 'Consumer Discretionary', 'Large', 3000000, 'Medium', True, 'Medium'),
        'MCD': StockInfo('MCD', 'McDonald\'s Corporation', 'Consumer Discretionary', 'Large', 2500000, 'Low', True, 'Low'),
        'NKE': StockInfo('NKE', 'NIKE Inc.', 'Consumer Discretionary', 'Large', 5000000, 'Medium', True, 'Medium'),
        'SBUX': StockInfo('SBUX', 'Starbucks Corporation', 'Consumer Discretionary', 'Large', 5000000, 'Medium', True, 'Medium'),
        'TGT': StockInfo('TGT', 'Target Corporation', 'Consumer Discretionary', 'Large', 3000000, 'Medium', True, 'Medium'),
        'LOW': StockInfo('LOW', 'Lowe\'s Companies Inc.', 'Consumer Discretionary', 'Large', 3500000, 'Medium', True, 'Medium'),
        'DIS': StockInfo('DIS', 'Walt Disney Company', 'Consumer Discretionary', 'Large', 8000000, 'Medium', True, 'Medium'),
        'BKNG': StockInfo('BKNG', 'Booking Holdings Inc.', 'Consumer Discretionary', 'Large', 300000, 'High', True, 'Medium'),
        'AMGN': StockInfo('AMGN', 'Amgen Inc.', 'Consumer Discretionary', 'Large', 2500000, 'Medium', True, 'Low'),
    }
    
    # Energy - Commodity driven, institutional heavy
    ENERGY = {
        'XOM': StockInfo('XOM', 'Exxon Mobil Corporation', 'Energy', 'Large', 20000000, 'High', True, 'High'),
        'CVX': StockInfo('CVX', 'Chevron Corporation', 'Energy', 'Large', 12000000, 'Medium', True, 'High'),
        'COP': StockInfo('COP', 'ConocoPhillips', 'Energy', 'Large', 7000000, 'High', True, 'Medium'),
        'SLB': StockInfo('SLB', 'Schlumberger NV', 'Energy', 'Large', 12000000, 'High', True, 'Medium'),
        'EOG': StockInfo('EOG', 'EOG Resources Inc.', 'Energy', 'Large', 4000000, 'High', True, 'Medium'),
        'DVN': StockInfo('DVN', 'Devon Energy Corporation', 'Energy', 'Large', 10000000, 'High', True, 'Medium'),
        'KMI': StockInfo('KMI', 'Kinder Morgan Inc.', 'Energy', 'Large', 15000000, 'Medium', True, 'Low'),
        'OXY': StockInfo('OXY', 'Occidental Petroleum', 'Energy', 'Large', 20000000, 'High', True, 'Medium'),
        'HAL': StockInfo('HAL', 'Halliburton Company', 'Energy', 'Large', 15000000, 'High', True, 'Medium'),
        'MPC': StockInfo('MPC', 'Marathon Petroleum Corp', 'Energy', 'Large', 4000000, 'High', True, 'Medium'),
    }
    
    # Industrial - Economic cycle indicators
    INDUSTRIAL = {
        'BA': StockInfo('BA', 'Boeing Company', 'Industrial', 'Large', 8000000, 'High', True, 'Medium'),
        'CAT': StockInfo('CAT', 'Caterpillar Inc.', 'Industrial', 'Large', 3000000, 'Medium', True, 'Medium'),
        'GE': StockInfo('GE', 'General Electric Company', 'Industrial', 'Large', 45000000, 'High', True, 'Medium'),
        'HON': StockInfo('HON', 'Honeywell International', 'Industrial', 'Large', 2000000, 'Medium', True, 'Low'),
        'UPS': StockInfo('UPS', 'United Parcel Service', 'Industrial', 'Large', 2500000, 'Medium', True, 'Low'),
        'RTX': StockInfo('RTX', 'Raytheon Technologies', 'Industrial', 'Large', 4000000, 'Medium', True, 'Low'),
        'DE': StockInfo('DE', 'Deere & Company', 'Industrial', 'Large', 1500000, 'Medium', True, 'Low'),
        'MMM': StockInfo('MMM', '3M Company', 'Industrial', 'Large', 2000000, 'Medium', True, 'Low'),
        'FDX': StockInfo('FDX', 'FedEx Corporation', 'Industrial', 'Large', 2000000, 'Medium', True, 'Low'),
        'LMT': StockInfo('LMT', 'Lockheed Martin Corp', 'Industrial', 'Large', 1000000, 'Medium', True, 'Low'),
    }
    
    # Communication Services
    COMMUNICATION = {
        'GOOGL': StockInfo('GOOGL', 'Alphabet Inc.', 'Communication', 'Large', 20000000, 'Medium', True, 'High'),
        'META': StockInfo('META', 'Meta Platforms Inc.', 'Communication', 'Large', 15000000, 'High', True, 'High'),
        'NFLX': StockInfo('NFLX', 'Netflix Inc.', 'Communication', 'Large', 5000000, 'High', True, 'High'),
        'DIS': StockInfo('DIS', 'Walt Disney Company', 'Communication', 'Large', 8000000, 'Medium', True, 'Medium'),
        'CMCSA': StockInfo('CMCSA', 'Comcast Corporation', 'Communication', 'Large', 15000000, 'Medium', True, 'Low'),
        'VZ': StockInfo('VZ', 'Verizon Communications', 'Communication', 'Large', 15000000, 'Low', True, 'Low'),
        'T': StockInfo('T', 'AT&T Inc.', 'Communication', 'Large', 35000000, 'Medium', True, 'Low'),
        'CHTR': StockInfo('CHTR', 'Charter Communications', 'Communication', 'Large', 1200000, 'High', True, 'Low'),
        'TMUS': StockInfo('TMUS', 'T-Mobile US Inc.', 'Communication', 'Large', 4000000, 'Medium', True, 'Medium'),
        'DISH': StockInfo('DISH', 'DISH Network Corporation', 'Communication', 'Mid', 3000000, 'High', False, 'Low'),
    }
    
    # Utilities - Defensive, dividend focused
    UTILITIES = {
        'NEE': StockInfo('NEE', 'NextEra Energy Inc.', 'Utilities', 'Large', 7000000, 'Low', True, 'Low'),
        'DUK': StockInfo('DUK', 'Duke Energy Corporation', 'Utilities', 'Large', 2500000, 'Low', True, 'Low'),
        'SO': StockInfo('SO', 'Southern Company', 'Utilities', 'Large', 4000000, 'Low', True, 'Low'),
        'D': StockInfo('D', 'Dominion Energy Inc.', 'Utilities', 'Large', 3000000, 'Low', True, 'Low'),
        'EXC': StockInfo('EXC', 'Exelon Corporation', 'Utilities', 'Large', 5000000, 'Low', True, 'Low'),
    }
    
    @classmethod
    def get_all_stocks(cls) -> Dict[str, StockInfo]:
        """Get all stocks from all sectors"""
        all_stocks = {}
        all_stocks.update(cls.TECHNOLOGY)
        all_stocks.update(cls.FINANCIAL)
        all_stocks.update(cls.HEALTHCARE)
        all_stocks.update(cls.CONSUMER_DISCRETIONARY)
        all_stocks.update(cls.ENERGY)
        all_stocks.update(cls.INDUSTRIAL)
        all_stocks.update(cls.COMMUNICATION)
        all_stocks.update(cls.UTILITIES)
        return all_stocks
    
    @classmethod
    def get_wyckoff_optimized_list(cls) -> List[str]:
        """Get symbols optimized for Wyckoff analysis"""
        all_stocks = cls.get_all_stocks()
        return [symbol for symbol, info in all_stocks.items() if info.wyckoff_friendly]
    
    @classmethod
    def get_high_volume_stocks(cls, min_volume: int = 10000000) -> List[str]:
        """Get stocks with high daily volume"""
        all_stocks = cls.get_all_stocks()
        return [symbol for symbol, info in all_stocks.items() if info.avg_volume >= min_volume]
    
    @classmethod
    def get_sector_stocks(cls, sector: str) -> List[str]:
        """Get all stocks from a specific sector"""
        all_stocks = cls.get_all_stocks()
        return [symbol for symbol, info in all_stocks.items() if info.sector == sector]
    
    @classmethod
    def get_sector_mapping(cls) -> Dict[str, str]:
        """Get symbol to sector mapping for risk management"""
        all_stocks = cls.get_all_stocks()
        return {symbol: info.sector for symbol, info in all_stocks.items()}
    
    @classmethod
    def get_high_institutional_activity(cls) -> List[str]:
        """Get stocks with high institutional activity"""
        all_stocks = cls.get_all_stocks()
        return [symbol for symbol, info in all_stocks.items() if info.institutional_activity == 'High']
    
    @classmethod
    def get_recommended_watchlist(cls, max_symbols: int = 20) -> List[str]:
        """
        Get recommended watchlist for Wyckoff trading
        Prioritizes: High institutional activity, Wyckoff-friendly, High volume
        """
        all_stocks = cls.get_all_stocks()
        
        # Score each stock
        scored_stocks = []
        for symbol, info in all_stocks.items():
            score = 0
            
            # Wyckoff friendly bonus
            if info.wyckoff_friendly:
                score += 10
            
            # Institutional activity bonus
            if info.institutional_activity == 'High':
                score += 8
            elif info.institutional_activity == 'Medium':
                score += 4
            
            # Volume bonus
            if info.avg_volume >= 20000000:
                score += 6
            elif info.avg_volume >= 10000000:
                score += 4
            elif info.avg_volume >= 5000000:
                score += 2
            
            # Volatility bonus (medium is preferred for Wyckoff)
            if info.volatility == 'Medium':
                score += 5
            elif info.volatility == 'High':
                score += 3
            
            # Market cap bonus (large cap preferred)
            if info.market_cap == 'Large':
                score += 3
            
            scored_stocks.append((symbol, score))
        
        # Sort by score and return top symbols
        scored_stocks.sort(key=lambda x: x[1], reverse=True)
        return [symbol for symbol, score in scored_stocks[:max_symbols]]
    
    @classmethod
    def get_sector_diversified_watchlist(cls, symbols_per_sector: int = 3) -> List[str]:
        """Get sector-diversified watchlist"""
        sectors = ['Technology', 'Financial', 'Healthcare', 'Consumer Discretionary', 
                  'Energy', 'Industrial', 'Communication']
        
        diversified_list = []
        all_stocks = cls.get_all_stocks()
        
        for sector in sectors:
            sector_stocks = [(symbol, info) for symbol, info in all_stocks.items() 
                           if info.sector == sector and info.wyckoff_friendly]
            
            # Sort by institutional activity and volume
            sector_stocks.sort(key=lambda x: (
                x[1].institutional_activity == 'High',
                x[1].avg_volume
            ), reverse=True)
            
            # Add top stocks from this sector
            for symbol, info in sector_stocks[:symbols_per_sector]:
                diversified_list.append(symbol)
        
        return diversified_list