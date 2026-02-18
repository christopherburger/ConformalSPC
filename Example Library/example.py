import numpy as np
import pandas as pd
import conformalSPC

# --- 1. Simulation Setup (Data with Volatility Shift) ---
n_calib = 300
n_test = 200
total_n = n_calib + n_test

# Generate a sine wave
time = np.arange(total_n)
signal = np.sin(time * 0.1) * 5

# Add Noise: Low noise first, then High noise (The Shift)
noise = np.concatenate([
    np.random.normal(0, 0.5, n_calib),  # Calibration Phase
    np.random.normal(0, 2.5, n_test)    # Test Phase
])
data = signal + noise

# --- 2. Modeling & Calibration ---
# We use pandas to get a rolling mean and rolling std dev (volatility)
series = pd.Series(data)
predictions = series.rolling(window=10).mean().bfill() 
local_vol = series.rolling(window=10).std().bfill()

# Split data
calib_data = data[:n_calib]
calib_preds = predictions[:n_calib]
calib_vol = local_vol[:n_calib]

# Calculate "Studentized" Scores for Calibration
# Score = |Error| / Volatility
calib_scores = conformalSPC.studentized_score(calib_data, calib_preds, calib_vol)

# Get the threshold (q) for 95% confidence
q = conformalSPC.get_conformal_threshold(calib_scores, alpha=0.05)
print(f"Adaptive Threshold (q): {q:.4f}")

# --- 3. Monitoring (Intervals) ---
test_preds = predictions[n_calib:]
test_vol = local_vol[n_calib:]

# Create Dynamic Intervals: Pred +/- (q * Volatility)
lower, upper = conformalSPC.get_adaptive_interval(test_preds, q, test_vol)

# Fix for Visualization: Concatenate to plot the continuous timeline
full_predictions = np.concatenate([calib_preds, test_preds])
# For calibration, the interval is just q * calib_vol
calib_lower = calib_preds - (q * calib_vol)
calib_upper = calib_preds + (q * calib_vol)
full_lower = np.concatenate([calib_lower, lower])
full_upper = np.concatenate([calib_upper, upper])

# --- 4. Monitoring (P-Values) ---
# Calculate scores for the test data
test_scores = conformalSPC.studentized_score(data[n_calib:], test_preds, test_vol)

# Calculate p-values
p_values = [conformalSPC.compute_p_value(s, calib_scores) for s in test_scores]

# --- 5. Visualizations ---
print("\nPlotting Adaptive Intervals...")
conformalSPC.plot_adaptive_intervals(
    data, 
    full_predictions, 
    full_lower, 
    full_upper, 
    shift_point=n_calib
)

print("Plotting P-Values...")
conformalSPC.plot_p_values(
    p_values, 
    alpha=0.05, 
    shift_point=0 # Relative to the start of the p-value array
)
