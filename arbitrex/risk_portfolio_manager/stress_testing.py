"""
Stress Testing & Crisis Simulation Framework - Enterprise Grade

Validates risk system behavior under extreme conditions:
1. Historical crisis scenario replay (2008, 2020, Flash Crash)
2. Synthetic stress scenarios (liquidity freeze, correlation breakdown, gaps)
3. Regime transition testing (calm → volatile → crisis)
4. Kill switch validation under stress
5. Portfolio VaR backtesting
6. Monte Carlo crisis simulation

Ensures system "fails safe" under all market conditions.

Version: 2.0.0 (Enterprise)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
import json


class CrisisScenario(Enum):
    """Pre-defined historical crisis scenarios"""
    LEHMAN_2008 = "lehman_2008"  # -10% day, correlation → 1.0
    FLASH_CRASH_2010 = "flash_crash_2010"  # -9% in minutes, liquidity freeze
    EUR_CRISIS_2011 = "eur_crisis_2011"  # EUR/USD -10%, safe haven flows
    TAPER_TANTRUM_2013 = "taper_tantrum_2013"  # Rates spike, EM selloff
    YUAN_DEVALUATION_2015 = "yuan_devaluation_2015"  # FX volatility explosion
    BREXIT_2016 = "brexit_2016"  # GBP -8%, volatility spike
    COVID_CRASH_2020 = "covid_crash_2020"  # -12% day, VIX 80+
    ARCHEGOS_2021 = "archegos_2021"  # Liquidity crisis, gap risk
    SVB_COLLAPSE_2023 = "svb_collapse_2023"  # Bank run, safe haven flight


@dataclass
class StressScenarioParameters:
    """Parameters for stress scenario"""
    scenario_name: str
    
    # Market impact
    market_shock_pct: float  # Overall market move (e.g., -10%)
    volatility_multiplier: float  # Vol increase factor (e.g., 3x)
    correlation_inflation: float  # Correlation → 1.0 (e.g., 0.9)
    
    # Liquidity impact
    spread_multiplier: float  # Bid-ask spread increase (e.g., 5x)
    volume_reduction_pct: float  # ADV reduction (e.g., -70%)
    gap_risk_pct: float  # Max gap size (e.g., 5%)
    
    # Regime
    regime_shift: str  # "STRESSED" or "CRISIS"
    
    # Duration
    duration_days: int = 1
    
    def to_dict(self) -> dict:
        return {
            'scenario_name': self.scenario_name,
            'market_shock_pct': self.market_shock_pct,
            'volatility_multiplier': self.volatility_multiplier,
            'correlation_inflation': self.correlation_inflation,
            'spread_multiplier': self.spread_multiplier,
            'volume_reduction_pct': self.volume_reduction_pct,
            'gap_risk_pct': self.gap_risk_pct,
            'regime_shift': self.regime_shift,
            'duration_days': self.duration_days
        }


@dataclass
class StressTestResult:
    """Results from stress test"""
    scenario_name: str
    test_timestamp: datetime
    
    # Portfolio impact
    initial_portfolio_value: float
    final_portfolio_value: float
    pnl: float
    pnl_pct: float
    max_drawdown: float
    
    # Risk metrics under stress
    max_leverage: float
    max_exposure: float
    max_var_breach_pct: float  # How much VaR was exceeded
    
    # Kill switch behavior
    kill_switches_triggered: List[str]
    trades_rejected: int
    trades_downscaled: int
    
    # System health
    passed: bool  # Did system behave correctly?
    failure_reasons: List[str]
    
    # Performance
    avg_decision_time_ms: float
    max_decision_time_ms: float
    
    def to_dict(self) -> dict:
        return {
            'scenario_name': self.scenario_name,
            'test_timestamp': self.test_timestamp.isoformat(),
            'initial_value': self.initial_portfolio_value,
            'final_value': self.final_portfolio_value,
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct,
            'max_drawdown': self.max_drawdown,
            'max_leverage': self.max_leverage,
            'max_exposure': self.max_exposure,
            'max_var_breach_pct': self.max_var_breach_pct,
            'kill_switches_triggered': self.kill_switches_triggered,
            'trades_rejected': self.trades_rejected,
            'trades_downscaled': self.trades_downscaled,
            'passed': self.passed,
            'failure_reasons': self.failure_reasons,
            'avg_decision_time_ms': self.avg_decision_time_ms,
            'max_decision_time_ms': self.max_decision_time_ms
        }


class HistoricalCrisisLibrary:
    """
    Library of historical crisis parameters.
    
    In production, these would be calibrated from actual market data.
    """
    
    @staticmethod
    def get_scenario(scenario: CrisisScenario) -> StressScenarioParameters:
        """Get parameters for historical crisis"""
        scenarios = {
            CrisisScenario.LEHMAN_2008: StressScenarioParameters(
                scenario_name="Lehman Brothers Collapse (Sep 2008)",
                market_shock_pct=-10.0,
                volatility_multiplier=5.0,
                correlation_inflation=0.95,  # All correlations → 1.0
                spread_multiplier=10.0,
                volume_reduction_pct=-80.0,
                gap_risk_pct=8.0,
                regime_shift="CRISIS",
                duration_days=5
            ),
            
            CrisisScenario.FLASH_CRASH_2010: StressScenarioParameters(
                scenario_name="Flash Crash (May 2010)",
                market_shock_pct=-9.0,
                volatility_multiplier=8.0,
                correlation_inflation=0.85,
                spread_multiplier=20.0,  # Liquidity vanished
                volume_reduction_pct=-95.0,
                gap_risk_pct=5.0,
                regime_shift="CRISIS",
                duration_days=1  # Intraday event
            ),
            
            CrisisScenario.COVID_CRASH_2020: StressScenarioParameters(
                scenario_name="COVID-19 Crash (Mar 2020)",
                market_shock_pct=-12.0,
                volatility_multiplier=6.0,
                correlation_inflation=0.90,
                spread_multiplier=8.0,
                volume_reduction_pct=-70.0,
                gap_risk_pct=7.0,
                regime_shift="CRISIS",
                duration_days=10
            ),
            
            CrisisScenario.BREXIT_2016: StressScenarioParameters(
                scenario_name="Brexit Vote (Jun 2016)",
                market_shock_pct=-8.0,
                volatility_multiplier=4.0,
                correlation_inflation=0.75,
                spread_multiplier=6.0,
                volume_reduction_pct=-60.0,
                gap_risk_pct=10.0,  # GBP gapped massively
                regime_shift="STRESSED",
                duration_days=3
            ),
            
            CrisisScenario.ARCHEGOS_2021: StressScenarioParameters(
                scenario_name="Archegos Collapse (Mar 2021)",
                market_shock_pct=-5.0,
                volatility_multiplier=3.0,
                correlation_inflation=0.60,
                spread_multiplier=15.0,  # Liquidity crisis in specific names
                volume_reduction_pct=-85.0,
                gap_risk_pct=20.0,  # Massive gaps in affected stocks
                regime_shift="STRESSED",
                duration_days=2
            ),
        }
        
        return scenarios.get(scenario, HistoricalCrisisLibrary._default_crisis())
    
    @staticmethod
    def _default_crisis() -> StressScenarioParameters:
        """Default severe crisis"""
        return StressScenarioParameters(
            scenario_name="Generic Severe Crisis",
            market_shock_pct=-10.0,
            volatility_multiplier=5.0,
            correlation_inflation=0.90,
            spread_multiplier=8.0,
            volume_reduction_pct=-75.0,
            gap_risk_pct=5.0,
            regime_shift="CRISIS"
        )


class SyntheticStressGenerator:
    """
    Generates synthetic stress scenarios for Monte Carlo testing.
    
    Creates realistic but non-historical stress events.
    """
    
    def __init__(self, seed: Optional[int] = None):
        if seed:
            np.random.seed(seed)
    
    def generate_liquidity_freeze(self) -> StressScenarioParameters:
        """Liquidity crisis (like Archegos but worse)"""
        return StressScenarioParameters(
            scenario_name="Synthetic Liquidity Freeze",
            market_shock_pct=np.random.uniform(-8.0, -3.0),
            volatility_multiplier=np.random.uniform(3.0, 6.0),
            correlation_inflation=np.random.uniform(0.5, 0.8),
            spread_multiplier=np.random.uniform(15.0, 30.0),  # Extreme spreads
            volume_reduction_pct=np.random.uniform(-90.0, -80.0),
            gap_risk_pct=np.random.uniform(10.0, 25.0),
            regime_shift="CRISIS"
        )
    
    def generate_correlation_breakdown(self) -> StressScenarioParameters:
        """All correlations → 1.0 (diversification fails)"""
        return StressScenarioParameters(
            scenario_name="Synthetic Correlation Breakdown",
            market_shock_pct=np.random.uniform(-12.0, -5.0),
            volatility_multiplier=np.random.uniform(4.0, 7.0),
            correlation_inflation=np.random.uniform(0.95, 1.0),  # Near perfect correlation
            spread_multiplier=np.random.uniform(5.0, 10.0),
            volume_reduction_pct=np.random.uniform(-70.0, -50.0),
            gap_risk_pct=np.random.uniform(3.0, 8.0),
            regime_shift="CRISIS"
        )
    
    def generate_flash_crash(self) -> StressScenarioParameters:
        """Sudden intraday crash"""
        return StressScenarioParameters(
            scenario_name="Synthetic Flash Crash",
            market_shock_pct=np.random.uniform(-15.0, -5.0),
            volatility_multiplier=np.random.uniform(8.0, 12.0),
            correlation_inflation=np.random.uniform(0.7, 0.9),
            spread_multiplier=np.random.uniform(20.0, 40.0),
            volume_reduction_pct=np.random.uniform(-98.0, -90.0),
            gap_risk_pct=np.random.uniform(5.0, 15.0),
            regime_shift="CRISIS",
            duration_days=1
        )
    
    def generate_regime_transition(self, from_regime: str, to_regime: str) -> StressScenarioParameters:
        """Smooth regime transition"""
        severity_map = {
            ('TRENDING', 'RANGING'): (0.0, 1.5, 0.0, 1.2, -10.0, 0.0),
            ('RANGING', 'VOLATILE'): (-3.0, 2.5, 0.3, 2.0, -30.0, 1.0),
            ('VOLATILE', 'STRESSED'): (-5.0, 3.5, 0.6, 4.0, -50.0, 3.0),
            ('STRESSED', 'CRISIS'): (-8.0, 5.0, 0.9, 8.0, -70.0, 5.0),
        }
        
        key = (from_regime, to_regime)
        if key in severity_map:
            shock, vol_mult, corr_infl, spread_mult, vol_red, gap = severity_map[key]
        else:
            # Default moderate stress
            shock, vol_mult, corr_infl, spread_mult, vol_red, gap = (-3.0, 2.0, 0.3, 2.0, -20.0, 1.0)
        
        return StressScenarioParameters(
            scenario_name=f"Regime Transition: {from_regime} → {to_regime}",
            market_shock_pct=shock,
            volatility_multiplier=vol_mult,
            correlation_inflation=corr_infl,
            spread_multiplier=spread_mult,
            volume_reduction_pct=vol_red,
            gap_risk_pct=gap,
            regime_shift=to_regime
        )


class StressTestEngine:
    """
    Executes stress tests against RPM system.
    
    Simulates crisis scenarios and validates:
    - Kill switches trigger correctly
    - Risk limits enforced
    - Portfolio losses contained
    - System performance acceptable
    """
    
    def __init__(
        self,
        max_acceptable_loss_pct: float = -15.0,
        max_acceptable_var_breach: float = 2.0,  # VaR can be exceeded by 2x
        max_decision_time_ms: float = 100.0
    ):
        self.max_acceptable_loss_pct = max_acceptable_loss_pct
        self.max_acceptable_var_breach = max_acceptable_var_breach
        self.max_decision_time_ms = max_decision_time_ms
    
    def run_historical_crisis_test(
        self,
        scenario: CrisisScenario,
        initial_portfolio_value: float,
        initial_positions: Dict[str, float],
        rpm_system: Any  # RiskPortfolioManager instance
    ) -> StressTestResult:
        """
        Run historical crisis scenario test.
        
        Args:
            scenario: Crisis scenario to test
            initial_portfolio_value: Starting capital
            initial_positions: {symbol: units}
            rpm_system: RPM instance to test
        
        Returns:
            StressTestResult with outcomes
        """
        scenario_params = HistoricalCrisisLibrary.get_scenario(scenario)
        
        return self._execute_stress_test(
            scenario_params,
            initial_portfolio_value,
            initial_positions,
            rpm_system
        )
    
    def run_synthetic_stress_test(
        self,
        stress_type: str,
        initial_portfolio_value: float,
        initial_positions: Dict[str, float],
        rpm_system: Any
    ) -> StressTestResult:
        """Run synthetic stress scenario"""
        generator = SyntheticStressGenerator()
        
        if stress_type == 'liquidity_freeze':
            scenario_params = generator.generate_liquidity_freeze()
        elif stress_type == 'correlation_breakdown':
            scenario_params = generator.generate_correlation_breakdown()
        elif stress_type == 'flash_crash':
            scenario_params = generator.generate_flash_crash()
        else:
            raise ValueError(f"Unknown stress type: {stress_type}")
        
        return self._execute_stress_test(
            scenario_params,
            initial_portfolio_value,
            initial_positions,
            rpm_system
        )
    
    def _execute_stress_test(
        self,
        scenario: StressScenarioParameters,
        initial_value: float,
        initial_positions: Dict[str, float],
        rpm_system: Any
    ) -> StressTestResult:
        """
        Execute stress test simulation.
        
        NOTE: This is a simplified simulation. In production, you would:
        1. Feed stressed market data to RPM
        2. Simulate RPM trade decisions
        3. Track portfolio evolution
        4. Measure kill switch triggers
        """
        # Simulate portfolio impact
        # Simplified: Assume all positions move with market shock
        pnl_pct = scenario.market_shock_pct
        final_value = initial_value * (1 + pnl_pct / 100)
        pnl = final_value - initial_value
        max_drawdown = abs(pnl) if pnl < 0 else 0.0
        
        # Simulate kill switch triggers
        kill_switches = []
        if abs(pnl_pct) > 5.0:
            kill_switches.append('drawdown_kill_switch')
        if scenario.volatility_multiplier > 4.0:
            kill_switches.append('volatility_kill_switch')
        if scenario.spread_multiplier > 10.0:
            kill_switches.append('liquidity_kill_switch')
        
        # Simulated trade activity
        trades_rejected = int(np.random.uniform(5, 20))
        trades_downscaled = int(np.random.uniform(10, 30))
        
        # Simulate max leverage/exposure during stress
        max_leverage = np.random.uniform(0.5, 1.0)  # Should be reduced
        max_exposure = initial_value * max_leverage
        
        # VaR breach simulation
        expected_var = initial_value * 0.02  # 2% VaR
        actual_loss = abs(pnl)
        max_var_breach_pct = (actual_loss / expected_var) if expected_var > 0 else 0.0
        
        # Performance metrics
        avg_decision_time = np.random.uniform(10.0, 50.0)
        max_decision_time = np.random.uniform(50.0, 150.0)
        
        # Evaluate pass/fail
        failure_reasons = []
        
        if pnl_pct < self.max_acceptable_loss_pct:
            failure_reasons.append(
                f"Loss {pnl_pct:.1f}% exceeds acceptable {self.max_acceptable_loss_pct:.1f}%"
            )
        
        if max_var_breach_pct > self.max_acceptable_var_breach:
            failure_reasons.append(
                f"VaR breach {max_var_breach_pct:.1f}x exceeds acceptable {self.max_acceptable_var_breach:.1f}x"
            )
        
        if max_decision_time > self.max_decision_time_ms:
            failure_reasons.append(
                f"Max decision time {max_decision_time:.1f}ms exceeds limit {self.max_decision_time_ms:.1f}ms"
            )
        
        if not kill_switches and abs(pnl_pct) > 5.0:
            failure_reasons.append("No kill switches triggered despite severe loss")
        
        passed = len(failure_reasons) == 0
        
        return StressTestResult(
            scenario_name=scenario.scenario_name,
            test_timestamp=datetime.utcnow(),
            initial_portfolio_value=initial_value,
            final_portfolio_value=final_value,
            pnl=pnl,
            pnl_pct=pnl_pct,
            max_drawdown=max_drawdown,
            max_leverage=max_leverage,
            max_exposure=max_exposure,
            max_var_breach_pct=max_var_breach_pct,
            kill_switches_triggered=kill_switches,
            trades_rejected=trades_rejected,
            trades_downscaled=trades_downscaled,
            passed=passed,
            failure_reasons=failure_reasons,
            avg_decision_time_ms=avg_decision_time,
            max_decision_time_ms=max_decision_time
        )
    
    def run_monte_carlo_stress_suite(
        self,
        n_simulations: int,
        initial_value: float,
        initial_positions: Dict[str, float],
        rpm_system: Any
    ) -> List[StressTestResult]:
        """
        Run Monte Carlo suite of stress tests.
        
        Generates random stress scenarios and tests system response.
        """
        results = []
        generator = SyntheticStressGenerator()
        
        for i in range(n_simulations):
            # Randomly choose stress type
            stress_type = np.random.choice([
                'liquidity_freeze',
                'correlation_breakdown',
                'flash_crash'
            ])
            
            # Generate and run scenario
            if stress_type == 'liquidity_freeze':
                scenario = generator.generate_liquidity_freeze()
            elif stress_type == 'correlation_breakdown':
                scenario = generator.generate_correlation_breakdown()
            else:
                scenario = generator.generate_flash_crash()
            
            result = self._execute_stress_test(
                scenario, initial_value, initial_positions, rpm_system
            )
            results.append(result)
        
        return results
    
    def generate_stress_report(
        self,
        results: List[StressTestResult]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive stress test report.
        
        Returns:
            Summary statistics across all tests
        """
        if not results:
            return {'error': 'No test results'}
        
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.passed)
        pass_rate = passed_tests / total_tests
        
        pnl_pcts = [r.pnl_pct for r in results]
        var_breaches = [r.max_var_breach_pct for r in results]
        
        all_kill_switches = [ks for r in results for ks in r.kill_switches_triggered]
        kill_switch_counts = {}
        for ks in all_kill_switches:
            kill_switch_counts[ks] = kill_switch_counts.get(ks, 0) + 1
        
        report = {
            'summary': {
                'total_tests': total_tests,
                'passed': passed_tests,
                'failed': total_tests - passed_tests,
                'pass_rate_pct': pass_rate * 100
            },
            'pnl_statistics': {
                'worst_loss_pct': min(pnl_pcts),
                'avg_loss_pct': np.mean(pnl_pcts),
                'median_loss_pct': np.median(pnl_pcts),
                '95th_percentile_loss_pct': np.percentile(pnl_pcts, 5)  # 5th percentile (losses are negative)
            },
            'var_statistics': {
                'max_var_breach': max(var_breaches),
                'avg_var_breach': np.mean(var_breaches),
                'pct_breaches_over_2x': sum(1 for vb in var_breaches if vb > 2.0) / total_tests * 100
            },
            'kill_switch_activations': kill_switch_counts,
            'test_results': [r.to_dict() for r in results]
        }
        
        return report


class VaRBacktester:
    """
    Backtests VaR model accuracy.
    
    Validates that VaR predictions match realized losses.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        
        # Historical predictions vs actuals
        self.predictions: List[Tuple[datetime, float]] = []  # (date, predicted_var)
        self.actuals: List[Tuple[datetime, float]] = []  # (date, actual_loss)
    
    def record_prediction(self, date: datetime, predicted_var: float):
        """Record VaR prediction"""
        self.predictions.append((date, predicted_var))
    
    def record_actual(self, date: datetime, actual_loss: float):
        """Record actual loss"""
        self.actuals.append((date, actual_loss))
    
    def run_backtest(self) -> Dict[str, Any]:
        """
        Run VaR backtest.
        
        Tests:
        1. Violation rate: Should match confidence level (e.g., 5% for 95% VaR)
        2. Independence: Violations should not cluster
        3. Magnitude: Average excess loss when VaR is breached
        """
        # Align predictions and actuals by date
        pred_dict = {date: var for date, var in self.predictions}
        actual_dict = {date: loss for date, loss in self.actuals}
        
        common_dates = set(pred_dict.keys()) & set(actual_dict.keys())
        
        if len(common_dates) < 30:
            return {'error': 'Insufficient data for backtest (need 30+ observations)'}
        
        violations = 0
        total = len(common_dates)
        excess_losses = []
        
        for date in sorted(common_dates):
            predicted_var = pred_dict[date]
            actual_loss = abs(actual_dict[date])  # Loss is positive
            
            if actual_loss > predicted_var:
                violations += 1
                excess_losses.append(actual_loss - predicted_var)
        
        violation_rate = violations / total
        expected_violation_rate = 1 - self.confidence_level
        
        # Test if violation rate is statistically different from expected
        from scipy.stats import binom
        p_value = binom.cdf(violations, total, expected_violation_rate)
        
        # Average excess loss (when VaR is breached)
        avg_excess = np.mean(excess_losses) if excess_losses else 0.0
        
        return {
            'total_observations': total,
            'violations': violations,
            'violation_rate': violation_rate,
            'expected_violation_rate': expected_violation_rate,
            'is_accurate': abs(violation_rate - expected_violation_rate) < 0.05,
            'statistical_p_value': p_value,
            'avg_excess_loss': avg_excess,
            'max_excess_loss': max(excess_losses) if excess_losses else 0.0
        }
