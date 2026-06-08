import qlib
import os

_provider = "/Users/zhaowei.yu/code/AIAgentStock/quantitative_factors/data/qlib/cn_data"
print(f"Qlib data dir: {_provider}")
qlib.init(provider_uri=_provider)

from qlib.data import D

# Test instruments() - it returns a dict spec
print("\n=== Testing instruments() ===")
instruments = D.instruments()
print(f"Type: {type(instruments)}")
print(f"Content: {instruments}")

# CORRECT USAGE: Pass the spec dict directly to D.features()
print("\n=== Testing D.features() with spec ===")
try:
    # This is the CORRECT way to use Qlib API
    data = D.features(instruments, ["$close"], start_time="2024-05-01", end_time="2024-05-10", freq="day")
    print(f"✓ Success! Shape: {data.shape}")
    print(f"First 10 rows:")
    print(data.head(10))
    
    # Get actual instrument list
    actual_instruments = data.index.get_level_values(0).unique()
    print(f"\n✓ Total instruments: {len(actual_instruments)}")
    print(f"First 20: {list(actual_instruments[:20])}")
except Exception as e:
    print(f"✗ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

