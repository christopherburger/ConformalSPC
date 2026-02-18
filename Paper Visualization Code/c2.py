import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_data_with_volatility_shift(n_points, shift_point, initial_noise_std, increased_noise_std):
    """
    Generates time-series data with a sudden increase in volatility.
    """
    time = np.arange(n_points)
    baseline = np.sin(time * 0.1) * 5 
    
    noise_before = np.random.normal(0, initial_noise_std, shift_point)
    noise_after = np.random.normal(0, increased_noise_std, n_points - shift_point)
    noise = np.concatenate([noise_before, noise_after])
    
    return baseline + noise

def plot_adaptive_conformal_chart(test_data, predictions_mean, lower_bounds, upper_bounds, shift_point):
    """
    Plots the adaptive conformal prediction intervals and the uncertainty spike chart.
    """
    title_fontsize = 20
    label_fontsize = 16
    tick_fontsize = 14
    legend_fontsize = 12
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    time_steps = np.arange(len(test_data))

    # --- Plot 1: Conformal Prediction Intervals ---
    ax1.plot(time_steps, test_data, 'k-', label='Actual Process Data', alpha=0.6)
    ax1.plot(time_steps, predictions_mean, 'b--', label='Point Prediction (Model Mean)')
    ax1.fill_between(time_steps, lower_bounds, upper_bounds, color='blue', alpha=0.2, label='95% Conformal Interval')
    
    ax1.axvline(x=shift_point, color='red', linestyle=':', linewidth=2, label='Volatility Shift')
    ax1.set_title('Conformal Prediction Intervals Over Time', fontsize=title_fontsize)
    ax1.set_ylabel('Observation Value', fontsize=label_fontsize)
    ax1.legend(loc='upper left', fontsize=legend_fontsize)
    ax1.grid(True)
    ax1.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    # --- Plot 2: Uncertainty Spike Chart ---
    interval_widths = upper_bounds - lower_bounds
    ax2.plot(time_steps, interval_widths, 'r-', label='Interval Width')
    
    ax2.axvline(x=shift_point, color='red', linestyle=':', linewidth=2, label='Volatility Shift')
    ax2.set_title('Uncertainty Spike Chart', fontsize=title_fontsize)
    ax2.set_xlabel('Time / Sample Number', fontsize=label_fontsize)
    ax2.set_ylabel('Prediction Interval Width', fontsize=label_fontsize)
    ax2.legend(loc='upper left', fontsize=legend_fontsize)
    ax2.grid(True)
    ax2.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # --- Simulation Parameters ---
    N_TOTAL = 500
    N_CALIB = 300
    N_TEST = N_TOTAL - N_CALIB
    VOLATILITY_SHIFT_POINT = 400
    ALPHA = 0.05
    INITIAL_NOISE = 0.5
    INCREASED_NOISE = 2.5
    MOVING_AVG_WINDOW = 10

    # Generate data with a shift in volatility
    full_data = generate_data_with_volatility_shift(
        n_points=N_TOTAL,
        shift_point=VOLATILITY_SHIFT_POINT,
        initial_noise_std=INITIAL_NOISE,
        increased_noise_std=INCREASED_NOISE
    )

    # --- Correctly split data into calibration and test sets ---
    calib_data = full_data[:N_CALIB]
    test_data = full_data[N_CALIB:]

    # --- Phase I: Model Training and Calibration on calib_data ---
    calib_series = pd.Series(calib_data)
    calib_pred_mean = calib_series.rolling(window=MOVING_AVG_WINDOW).mean().bfill()
    calib_residuals = np.abs(calib_data - calib_pred_mean.values)

    calib_residuals_series = pd.Series(calib_residuals)
    calib_pred_std = calib_residuals_series.rolling(window=MOVING_AVG_WINDOW).mean().bfill()
    calib_pred_std[calib_pred_std < 1e-6] = 1e-6

    valid_calib_indices = range(MOVING_AVG_WINDOW, N_CALIB)
    non_conformity_scores = calib_residuals[valid_calib_indices] / calib_pred_std.values[valid_calib_indices]

    q = np.quantile(non_conformity_scores, 1 - ALPHA)
    print(f"Calibrated Quantile (q): {q:.4f}")

    # --- Phase II: Real-Time Monitoring on test_data ---
    warm_up_data = calib_data
    extended_data_for_testing = np.concatenate([warm_up_data, test_data])

    extended_series = pd.Series(extended_data_for_testing)
    pred_mean_full = extended_series.rolling(window=MOVING_AVG_WINDOW).mean()

    pred_residuals_full = np.abs(extended_data_for_testing - pred_mean_full.bfill().values)
    pred_residuals_series = pd.Series(pred_residuals_full)
    pred_std_full = pred_residuals_series.rolling(window=MOVING_AVG_WINDOW).mean()
    pred_std_full[pred_std_full < 1e-6] = 1e-6

    # --- Trim to match test_data length ---
    test_pred_mean = pred_mean_full.values[-N_TEST:]
    test_pred_std = pred_std_full.values[-N_TEST:]

    interval_half_width = q * test_pred_std
    lower_bounds = test_pred_mean - interval_half_width
    upper_bounds = test_pred_mean + interval_half_width

    # --- Optional Debug: Assert matching dimensions ---
    assert test_data.shape[0] == test_pred_mean.shape[0] == lower_bounds.shape[0] == upper_bounds.shape[0], \
        "Mismatch in dimensions of plotting arrays!"

    # --- Plot ---
    plot_adaptive_conformal_chart(
        test_data=test_data,
        predictions_mean=test_pred_mean,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        shift_point=(VOLATILITY_SHIFT_POINT - N_CALIB)
    )
