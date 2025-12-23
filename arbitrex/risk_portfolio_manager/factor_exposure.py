"""
Factor & Sector Exposure Tracking - Enterprise Grade

Implements sophisticated exposure analytics:
1. Equity factor exposure (beta, momentum, value, size, volatility)
2. Sector/industry concentration limits
3. Macro theme exposure (rates, commodities, risk-on/off)
4. Factor contribution to portfolio risk
5. Cross-currency (FX) exposure decomposition

Prevents concentration risk at factor and sector levels.

Version: 2.0.0 (Enterprise)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime
from enum import Enum
from collections import defaultdict


class EquityFactor(Enum):
    """Standard equity risk factors"""
    MARKET_BETA = "market_beta"  # Systematic market risk
    MOMENTUM = "momentum"  # Price momentum
    VALUE = "value"  # Value vs growth
    SIZE = "size"  # Market cap (small vs large)
    VOLATILITY = "volatility"  # Low-vol anomaly
    QUALITY = "quality"  # Profitability, stability


class Sector(Enum):
    """GICS Sectors"""
    ENERGY = "energy"
    MATERIALS = "materials"
    INDUSTRIALS = "industrials"
    CONSUMER_DISCRETIONARY = "consumer_discretionary"
    CONSUMER_STAPLES = "consumer_staples"
    HEALTHCARE = "healthcare"
    FINANCIALS = "financials"
    INFORMATION_TECHNOLOGY = "information_technology"
    COMMUNICATION_SERVICES = "communication_services"
    UTILITIES = "utilities"
    REAL_ESTATE = "real_estate"
    COMMODITIES = "commodities"
    CURRENCIES = "currencies"
    FIXED_INCOME = "fixed_income"
    CRYPTO = "crypto"


class MacroTheme(Enum):
    """Macro regime themes"""
    RISK_ON = "risk_on"  # Growth, equities, EM, high-yield
    RISK_OFF = "risk_off"  # Safety, bonds, gold, JPY
    RATES_SENSITIVE = "rates_sensitive"  # Duration, financials
    COMMODITY_SENSITIVE = "commodity_sensitive"  # Energy, materials, inflation
    USD_STRENGTH = "usd_strength"  # Dollar moves
    VOLATILITY_LONG = "volatility_long"  # Long vol (options, VIX)


@dataclass
class AssetFactorProfile:
    """Factor exposures for a single asset"""
    symbol: str
    sector: Sector
    market_beta: float = 1.0
    momentum: float = 0.0  # Z-score vs market
    value: float = 0.0  # Z-score vs market
    size: float = 0.0  # Z-score (negative = large cap)
    volatility: float = 1.0  # Relative to market
    quality: float = 0.0  # Z-score
    
    # Macro theme exposures (0.0 to 1.0)
    risk_on_exposure: float = 0.5
    rates_sensitivity: float = 0.0  # Duration-like measure
    commodity_sensitivity: float = 0.0
    usd_sensitivity: float = 0.0
    
    # Currency
    base_currency: str = "USD"
    quote_currency: Optional[str] = None  # For FX pairs
    
    def get_factor_vector(self) -> Dict[EquityFactor, float]:
        """Return factor exposures as dict"""
        return {
            EquityFactor.MARKET_BETA: self.market_beta,
            EquityFactor.MOMENTUM: self.momentum,
            EquityFactor.VALUE: self.value,
            EquityFactor.SIZE: self.size,
            EquityFactor.VOLATILITY: self.volatility,
            EquityFactor.QUALITY: self.quality
        }


@dataclass
class PortfolioFactorExposure:
    """Aggregate factor exposures for portfolio"""
    total_market_beta: float
    total_momentum: float
    total_value: float
    total_size: float
    total_volatility: float
    total_quality: float
    
    # Sector concentration
    sector_exposures: Dict[Sector, float]  # % of capital
    max_sector_exposure: Tuple[Sector, float]  # (sector, %)
    
    # Macro themes
    risk_on_exposure: float  # 0.0 = risk-off, 1.0 = risk-on
    rates_sensitivity: float
    commodity_sensitivity: float
    usd_net_exposure: float  # Net USD exposure
    
    # FX breakdown
    currency_exposures: Dict[str, float]  # {currency: net_exposure_pct}
    
    # Risk contribution
    factor_risk_contributions: Dict[EquityFactor, float]  # % of portfolio variance
    
    timestamp: datetime
    
    def to_dict(self) -> dict:
        return {
            'market_beta': self.total_market_beta,
            'momentum': self.total_momentum,
            'value': self.total_value,
            'size': self.total_size,
            'volatility': self.total_volatility,
            'quality': self.total_quality,
            'sector_exposures': {s.value: v for s, v in self.sector_exposures.items()},
            'max_sector': self.max_sector_exposure[0].value if self.max_sector_exposure else None,
            'max_sector_pct': self.max_sector_exposure[1] if self.max_sector_exposure else 0.0,
            'risk_on_exposure': self.risk_on_exposure,
            'rates_sensitivity': self.rates_sensitivity,
            'commodity_sensitivity': self.commodity_sensitivity,
            'usd_net_exposure': self.usd_net_exposure,
            'currency_exposures': self.currency_exposures,
            'factor_risk_contributions': {f.value: v for f, v in self.factor_risk_contributions.items()},
            'timestamp': self.timestamp.isoformat()
        }


class AssetFactorDatabase:
    """
    Database of asset factor profiles.
    
    In production, this would load from:
    - Bloomberg/Refinitiv factor models
    - Internal quantitative models
    - Risk vendor (MSCI Barra, Axioma)
    
    For now, implements configurable defaults and manual overrides.
    """
    
    def __init__(self):
        # Asset profiles: {symbol: AssetFactorProfile}
        self.profiles: Dict[str, AssetFactorProfile] = {}
        
        # Default sector mappings (FX symbols)
        self._initialize_fx_profiles()
    
    def _initialize_fx_profiles(self):
        """Initialize common FX pair profiles"""
        # Major pairs
        self.register_asset(AssetFactorProfile(
            symbol="EURUSD",
            sector=Sector.CURRENCIES,
            market_beta=0.0,
            base_currency="EUR",
            quote_currency="USD",
            usd_sensitivity=-1.0,  # Inverse USD
            risk_on_exposure=0.6
        ))
        
        self.register_asset(AssetFactorProfile(
            symbol="GBPUSD",
            sector=Sector.CURRENCIES,
            market_beta=0.0,
            base_currency="GBP",
            quote_currency="USD",
            usd_sensitivity=-1.0,
            risk_on_exposure=0.7
        ))
        
        self.register_asset(AssetFactorProfile(
            symbol="USDJPY",
            sector=Sector.CURRENCIES,
            market_beta=0.0,
            base_currency="USD",
            quote_currency="JPY",
            usd_sensitivity=1.0,
            risk_on_exposure=0.8  # JPY is risk-off
        ))
        
        self.register_asset(AssetFactorProfile(
            symbol="AUDUSD",
            sector=Sector.CURRENCIES,
            market_beta=0.0,
            base_currency="AUD",
            quote_currency="USD",
            usd_sensitivity=-1.0,
            risk_on_exposure=0.9,  # AUD is risk-on
            commodity_sensitivity=0.8  # Commodity currency
        ))
        
        # Commodity FX
        self.register_asset(AssetFactorProfile(
            symbol="USDCAD",
            sector=Sector.CURRENCIES,
            market_beta=0.0,
            base_currency="USD",
            quote_currency="CAD",
            usd_sensitivity=1.0,
            commodity_sensitivity=0.7  # CAD sensitive to oil
        ))
        
        # Safe havens
        self.register_asset(AssetFactorProfile(
            symbol="XAUUSD",  # Gold
            sector=Sector.COMMODITIES,
            market_beta=-0.3,  # Negative correlation with stocks
            risk_on_exposure=0.2,  # Risk-off asset
            usd_sensitivity=-0.5,
            commodity_sensitivity=1.0,
            base_currency="USD"
        ))
    
    def register_asset(self, profile: AssetFactorProfile):
        """Register or update asset factor profile"""
        self.profiles[profile.symbol] = profile
    
    def get_profile(self, symbol: str) -> AssetFactorProfile:
        """
        Get factor profile for asset.
        
        If not found, returns generic profile based on symbol heuristics.
        """
        if symbol in self.profiles:
            return self.profiles[symbol]
        
        # Heuristic defaults
        if any(curr in symbol for curr in ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD']):
            sector = Sector.CURRENCIES
            beta = 0.0
        elif symbol in ['XAUUSD', 'XAGUSD', 'OIL', 'COPPER']:
            sector = Sector.COMMODITIES
            beta = 0.3
        elif 'BTC' in symbol or 'ETH' in symbol:
            sector = Sector.CRYPTO
            beta = 1.5  # High beta
        else:
            sector = Sector.INFORMATION_TECHNOLOGY  # Default
            beta = 1.0
        
        # Create default profile
        return AssetFactorProfile(
            symbol=symbol,
            sector=sector,
            market_beta=beta,
            base_currency="USD"
        )
    
    def update_factor(self, symbol: str, factor: EquityFactor, value: float):
        """Update single factor for asset"""
        if symbol not in self.profiles:
            self.profiles[symbol] = self.get_profile(symbol)
        
        profile = self.profiles[symbol]
        
        if factor == EquityFactor.MARKET_BETA:
            profile.market_beta = value
        elif factor == EquityFactor.MOMENTUM:
            profile.momentum = value
        elif factor == EquityFactor.VALUE:
            profile.value = value
        elif factor == EquityFactor.SIZE:
            profile.size = value
        elif factor == EquityFactor.VOLATILITY:
            profile.volatility = value
        elif factor == EquityFactor.QUALITY:
            profile.quality = value


class FactorExposureCalculator:
    """
    Calculates portfolio-level factor exposures.
    
    Aggregates individual asset exposures weighted by position size.
    """
    
    def __init__(
        self,
        max_sector_pct: float = 0.40,  # 40% max in any sector
        max_beta: float = 2.0,  # Max portfolio beta
        max_factor_exposure: float = 3.0  # Max sum of abs(factor exposures)
    ):
        self.max_sector_pct = max_sector_pct
        self.max_beta = max_beta
        self.max_factor_exposure = max_factor_exposure
        
        self.factor_db = AssetFactorDatabase()
    
    def calculate_portfolio_exposure(
        self,
        positions: Dict[str, float],  # {symbol: units}
        prices: Dict[str, float],  # {symbol: price}
        total_capital: float
    ) -> PortfolioFactorExposure:
        """
        Calculate comprehensive factor exposures.
        
        Returns:
            PortfolioFactorExposure with all metrics
        """
        if not positions:
            return self._empty_exposure()
        
        # Calculate position values and weights
        position_values = {s: positions[s] * prices[s] for s in positions}
        weights = {s: position_values[s] / total_capital for s in positions}
        
        # Aggregate factor exposures (weighted sum)
        total_beta = sum(weights[s] * self.factor_db.get_profile(s).market_beta for s in positions)
        total_momentum = sum(weights[s] * self.factor_db.get_profile(s).momentum for s in positions)
        total_value = sum(weights[s] * self.factor_db.get_profile(s).value for s in positions)
        total_size = sum(weights[s] * self.factor_db.get_profile(s).size for s in positions)
        total_volatility = sum(weights[s] * self.factor_db.get_profile(s).volatility for s in positions)
        total_quality = sum(weights[s] * self.factor_db.get_profile(s).quality for s in positions)
        
        # Sector exposures
        sector_exposures = defaultdict(float)
        for symbol in positions:
            sector = self.factor_db.get_profile(symbol).sector
            sector_exposures[sector] += abs(weights[symbol])
        
        max_sector = max(sector_exposures.items(), key=lambda x: x[1]) if sector_exposures else (Sector.CURRENCIES, 0.0)
        
        # Macro themes
        risk_on = sum(weights[s] * self.factor_db.get_profile(s).risk_on_exposure for s in positions)
        rates_sens = sum(weights[s] * self.factor_db.get_profile(s).rates_sensitivity for s in positions)
        commodity_sens = sum(weights[s] * self.factor_db.get_profile(s).commodity_sensitivity for s in positions)
        usd_sens = sum(weights[s] * self.factor_db.get_profile(s).usd_sensitivity for s in positions)
        
        # Currency exposures
        currency_exposures = self._calculate_fx_exposures(positions, prices, total_capital)
        
        # Factor risk contributions (simplified - would need covariance in production)
        factor_risk_contributions = self._estimate_factor_risk_contributions(
            total_beta, total_momentum, total_value, total_size, total_volatility, total_quality
        )
        
        return PortfolioFactorExposure(
            total_market_beta=total_beta,
            total_momentum=total_momentum,
            total_value=total_value,
            total_size=total_size,
            total_volatility=total_volatility,
            total_quality=total_quality,
            sector_exposures=dict(sector_exposures),
            max_sector_exposure=max_sector,
            risk_on_exposure=risk_on,
            rates_sensitivity=rates_sens,
            commodity_sensitivity=commodity_sens,
            usd_net_exposure=usd_sens,
            currency_exposures=currency_exposures,
            factor_risk_contributions=factor_risk_contributions,
            timestamp=datetime.utcnow()
        )
    
    def _calculate_fx_exposures(
        self,
        positions: Dict[str, float],
        prices: Dict[str, float],
        total_capital: float
    ) -> Dict[str, float]:
        """
        Decompose FX exposures by currency.
        
        For FX pair XXXYYY:
        - Long position: +XXX, -YYY
        - Short position: -XXX, +YYY
        
        Returns:
            {currency: net_exposure_pct}
        """
        currency_exposures = defaultdict(float)
        
        for symbol in positions:
            profile = self.factor_db.get_profile(symbol)
            position_value = positions[symbol] * prices[symbol]
            weight = position_value / total_capital
            
            if profile.quote_currency:
                # FX pair
                base_sign = np.sign(positions[symbol])
                currency_exposures[profile.base_currency] += base_sign * abs(weight)
                currency_exposures[profile.quote_currency] -= base_sign * abs(weight)
            else:
                # Single currency instrument
                currency_exposures[profile.base_currency] += weight
        
        return dict(currency_exposures)
    
    def _estimate_factor_risk_contributions(
        self,
        beta: float,
        momentum: float,
        value: float,
        size: float,
        volatility: float,
        quality: float
    ) -> Dict[EquityFactor, float]:
        """
        Estimate each factor's contribution to portfolio variance.
        
        Simplified model: risk ∝ exposure² (ignores covariances)
        In production, use factor covariance matrix.
        """
        factor_variances = {
            EquityFactor.MARKET_BETA: beta**2 * 0.15**2,  # Market vol ~15%
            EquityFactor.MOMENTUM: momentum**2 * 0.10**2,  # Momentum vol ~10%
            EquityFactor.VALUE: value**2 * 0.08**2,
            EquityFactor.SIZE: size**2 * 0.06**2,
            EquityFactor.VOLATILITY: volatility**2 * 0.05**2,
            EquityFactor.QUALITY: quality**2 * 0.04**2
        }
        
        total_variance = sum(factor_variances.values())
        
        if total_variance == 0:
            return {f: 0.0 for f in EquityFactor}
        
        # Normalize to percentages
        contributions = {f: v / total_variance for f, v in factor_variances.items()}
        
        return contributions
    
    def _empty_exposure(self) -> PortfolioFactorExposure:
        """Return zero exposure"""
        return PortfolioFactorExposure(
            total_market_beta=0.0,
            total_momentum=0.0,
            total_value=0.0,
            total_size=0.0,
            total_volatility=0.0,
            total_quality=0.0,
            sector_exposures={},
            max_sector_exposure=(Sector.CURRENCIES, 0.0),
            risk_on_exposure=0.5,
            rates_sensitivity=0.0,
            commodity_sensitivity=0.0,
            usd_net_exposure=0.0,
            currency_exposures={},
            factor_risk_contributions={f: 0.0 for f in EquityFactor},
            timestamp=datetime.utcnow()
        )
    
    def check_exposure_limits(
        self,
        exposure: PortfolioFactorExposure
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if exposures breach limits.
        
        Returns:
            (is_acceptable, rejection_reason)
        """
        # Check sector concentration
        if exposure.max_sector_exposure[1] > self.max_sector_pct:
            return False, f"Sector {exposure.max_sector_exposure[0].value} concentration {exposure.max_sector_exposure[1]*100:.1f}% exceeds limit {self.max_sector_pct*100:.1f}%"
        
        # Check market beta
        if abs(exposure.total_market_beta) > self.max_beta:
            return False, f"Portfolio beta {exposure.total_market_beta:.2f} exceeds limit {self.max_beta:.2f}"
        
        # Check total factor exposure (sum of abs values)
        total_factor_exposure = sum([
            abs(exposure.total_momentum),
            abs(exposure.total_value),
            abs(exposure.total_size),
            abs(exposure.total_quality)
        ])
        
        if total_factor_exposure > self.max_factor_exposure:
            return False, f"Total factor exposure {total_factor_exposure:.2f} exceeds limit {self.max_factor_exposure:.2f}"
        
        return True, None
    
    def get_factor_diversification_score(
        self,
        exposure: PortfolioFactorExposure
    ) -> float:
        """
        Calculate factor diversification score (0.0 to 1.0).
        
        Higher = more diversified across factors.
        Uses entropy of factor risk contributions.
        """
        contributions = list(exposure.factor_risk_contributions.values())
        
        if sum(contributions) == 0:
            return 0.0
        
        # Normalize
        contributions = np.array(contributions) / sum(contributions)
        
        # Shannon entropy
        entropy = -sum(c * np.log(c + 1e-10) for c in contributions if c > 0)
        
        # Normalize to [0, 1] (max entropy = log(6) for 6 factors)
        max_entropy = np.log(len(EquityFactor))
        diversification_score = entropy / max_entropy
        
        return diversification_score
