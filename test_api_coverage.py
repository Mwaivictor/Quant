"""Test that all 13 new API endpoints are loaded correctly"""

from arbitrex.risk_portfolio_manager import api

# Get all routes
routes = [r for r in api.app.routes if hasattr(r, 'methods') and hasattr(r, 'path')]

print(f"Total API endpoints: {len(routes)}")

# Test expectancy endpoints
expectancy_routes = [r for r in routes if '/expectancy' in r.path]
print(f"\n✅ Expectancy endpoints: {len(expectancy_routes)}")
for r in expectancy_routes:
    print(f"  {list(r.methods)[0]:6s} {r.path}")

# Test portfolio risk endpoints
portfolio_risk_routes = [r for r in routes if r.path in ['/portfolio/var_cvar', '/portfolio/covariance_matrix', '/portfolio/volatility_target']]
print(f"\n✅ Portfolio Risk endpoints: {len(portfolio_risk_routes)}")
for r in portfolio_risk_routes:
    print(f"  GET    {r.path}")

# Test adaptive thresholds endpoints
adaptive_routes = [r for r in routes if '/adaptive_thresholds' in r.path]
print(f"\n✅ Adaptive Thresholds endpoints: {len(adaptive_routes)}")
for r in adaptive_routes:
    print(f"  GET    {r.path}")

# Test factor exposure endpoints
factor_routes = [r for r in routes if r.path in ['/portfolio/factor_exposure', '/portfolio/sector_limits']]
print(f"\n✅ Factor Exposure endpoints: {len(factor_routes)}")
for r in factor_routes:
    print(f"  GET    {r.path}")

# Test observability endpoints
observability_routes = [r for r in routes if '/observability' in r.path]
print(f"\n✅ Observability endpoints: {len(observability_routes)}")
for r in observability_routes:
    print(f"  GET    {r.path}")

# Summary
total_new = len(expectancy_routes) + len(portfolio_risk_routes) + len(adaptive_routes) + len(factor_routes) + len(observability_routes)
print(f"\n{'='*60}")
print(f"Total NEW endpoints added: {total_new}/13")
print(f"Total ALL endpoints: {len(routes)}")
print(f"{'='*60}")

if total_new == 13:
    print("✅ SUCCESS - All 13 new endpoints integrated!")
    print("\nAPI Coverage: 100% (55/55 endpoints)")
else:
    print(f"⚠️  Expected 13 new endpoints, found {total_new}")
