#!/usr/bin/env python3
"""Minimal test to debug segfault"""
import sys
sys.path.insert(0, 'src/python')

print("Step 1: Importing module...")
try:
    import entropy_cpp
    print("✓ Import successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

print("\nStep 2: Creating engine...")
try:
    engine = entropy_cpp.EntropyEngine()
    print("✓ Engine created")
except Exception as e:
    print(f"✗ Engine creation failed: {e}")
    sys.exit(1)

print("\nStep 3: Testing numpy import...")
try:
    import numpy as np
    print("✓ Numpy imported")
except Exception as e:
    print(f"✗ Numpy import failed: {e}")
    sys.exit(1)

print("\nStep 4: Creating small test data...")
try:
    x = np.array([0.0, 1.0, -1.0, 0.5, -0.5], dtype=np.float64)
    y = np.array([0.0, 0.9, -0.9, 0.4, -0.6], dtype=np.float64)
    print(f"✓ Data created: x shape={x.shape}, y shape={y.shape}")
    print(f"  x dtype={x.dtype}, y dtype={y.dtype}")
except Exception as e:
    print(f"✗ Data creation failed: {e}")
    sys.exit(1)

print("\nStep 5: Testing optimize_bandwidths...")
try:
    h1, h2 = engine.optimize_bandwidths(x, y)
    print(f"✓ Bandwidths: h1={h1:.6f}, h2={h2:.6f}")
except Exception as e:
    print(f"✗ Bandwidth optimization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nStep 6: Testing calculate_metrics...")
try:
    s_rho, down_asy = engine.calculate_metrics(x, y, 0.0)
    print(f"✓ Metrics calculated: S_rho={s_rho:.6f}, DOWN_ASY={down_asy:.6f}")
except Exception as e:
    print(f"✗ Metric calculation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ All steps passed!")
