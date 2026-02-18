import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, expon

def generate_data(distribution_type, n_calib, n_test, shift_point=None, shift_magnitude=0):
    """
    Generates calibration and test data from a specified distribution.
    
    Args:
        distribution_type (str): 'normal', 't', or 'exponential'.
        n_calib (int): Number of calibration samples.
        n_test (int): Number of test samples.
        shift_point (int, optional): Index at which to introduce a process shift. Defaults to None.
        shift_magnitude (float, optional): The magnitude of the process shift. Defaults to 0.
        
    Returns:
        tuple: A tuple containing calibration_data and test_data.
    """
    # Generate calibration data (Phase I)
    if distribution_type == 'normal':
        calib_data = np.random.normal(0, 1, n_calib)
    elif distribution_type == 't':
        # Using 10 degrees of freedom for a distribution similar to normal but with heavier tails
        calib_data = t.rvs(df=10, size=n_calib)
    elif distribution_type == 'exponential':
        calib_data = expon.rvs(size=n_calib)
    else:
        raise ValueError("Unsupported distribution type")

    # Generate test data (Phase II)
    if distribution_type == 'normal':
        test_data = np.random.normal(0, 1, n_test)
    elif distribution_type == 't':
        test_data = t.rvs(df=10, size=n_test)
    elif distribution_type == 'exponential':
        test_data = expon.rvs(size=n_test)

    # Introduce a shift in the process if specified
    if shift_point is not None and 0 <= shift_point < n_test:
        test_data[shift_point:] += shift_magnitude
        
    return calib_data, test_data

def calculate_shewhart_limits(data):
    """Calculates traditional Shewhart control limits (mean +/- 3*std)."""
    mean = np.mean(data)
    std_dev = np.std(data)
    ucl = mean + 3 * std_dev
    lcl = mean - 3 * std_dev
    return mean, ucl, lcl

def calculate_conformal_limit(calib_data, alpha):
    """Calculates the conformal control limit based on a non-conformity score."""
    # Use a robust non-conformity score: absolute deviation from the median
    median = np.median(calib_data)
    scores = np.abs(calib_data - median)
    
    # The control limit is the (1-alpha) quantile of the calibration scores
    # We add 1 to n_calib for the finite sample correction
    quantile_level = (1 - alpha) * (len(calib_data) + 1) / len(calib_data)
    if quantile_level > 1: # Ensure quantile level does not exceed 1
        quantile_level = 1.0
        
    conformal_limit = np.quantile(scores, quantile_level)
    
    return median, conformal_limit

def plot_comparison(test_data, shewhart_mean, shewhart_ucl, shewhart_lcl, conformal_median, conformal_limit, distribution_name, shift_point=None):
    """
    Plots a comparison of the Shewhart and Conformal control charts with enhanced font sizes and fixed legend positions.
    """
    # --- Font size definitions for better visibility ---
    title_fontsize = 20
    label_fontsize = 16
    tick_fontsize = 14
    legend_fontsize = 11
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    time_steps = np.arange(1, len(test_data) + 1)

    # --- Shewhart Control Chart ---
    ax1.plot(time_steps, test_data, 'ko-', label='Test Data')
    ax1.axhline(shewhart_mean, color='blue', linestyle='--', label='Center Line (Mean)')
    ax1.axhline(shewhart_ucl, color='red', linestyle='--', label='UCL')
    ax1.axhline(shewhart_lcl, color='red', linestyle='--', label='LCL')
    
    # Find and count out-of-control points
    out_of_control_shewhart_indices = np.where((test_data > shewhart_ucl) | (test_data < shewhart_lcl))
    num_ooc_shewhart = len(out_of_control_shewhart_indices)
    ax1.plot(time_steps[out_of_control_shewhart_indices], test_data[out_of_control_shewhart_indices], 'ro', markersize=10, label='Out of Control')
    
    ax1.set_title(f'Traditional Shewhart Chart ({distribution_name} Data)', fontsize=title_fontsize)
    ax1.set_ylabel('Observation Value', fontsize=label_fontsize)
    ax1.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax1.grid(True)

    # --- Conformal-Enhanced Control Chart ---
    # Calculate non-conformity scores for the test data
    test_scores = np.abs(test_data - conformal_median)
    
    ax2.plot(time_steps, test_scores, 'ko-', label='Non-Conformity Scores')
    ax2.axhline(conformal_limit, color='red', linestyle='--', label='Conformal Limit (q)')
    
    # Find and count out-of-control points
    out_of_control_conformal_indices = np.where(test_scores > conformal_limit)
    num_ooc_conformal = len(out_of_control_conformal_indices)
    ax2.plot(time_steps[out_of_control_conformal_indices], test_scores[out_of_control_conformal_indices], 'ro', markersize=10, label='Out of Control')

    ax2.set_title(f'Conformal-Enhanced Chart ({distribution_name} Data)', fontsize=title_fontsize)
    ax2.set_xlabel('Time / Sample Number', fontsize=label_fontsize)
    ax2.set_ylabel('Non-Conformity Score', fontsize=label_fontsize)
    ax2.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax2.grid(True)

    # Add a line indicating the process shift and place legends
    if shift_point is not None:
        ax1.axvline(x=shift_point, color='green', linestyle=':', linewidth=2, label='Process Shift')
        ax2.axvline(x=shift_point, color='green', linestyle=':', linewidth=2, label='Process Shift')

    ax1.legend(loc='upper left', fontsize=legend_fontsize)
    ax2.legend(loc='upper left', fontsize=legend_fontsize)

    plt.tight_layout()
    plt.show()
    
    return num_ooc_shewhart, num_ooc_conformal

if __name__ == '__main__':
    # --- Simulation Parameters ---
    N_CALIBRATION = 200  # Number of data points for Phase I
    N_TEST = 100         # Number of data points for Phase II monitoring
    ALPHA = 0.05         # Desired false alarm rate (1 - confidence level)
    SHIFT_POINT = 50     # Point in time where the process shifts
    SHIFT_MAGNITUDE = 2.5 # How much the process mean shifts
    
    # Distributions to test
    distributions = ['normal', 't', 'exponential']
    
    for dist in distributions:
        print(f"--- Generating comparison for {dist.capitalize()} Distribution ---")
        
        # Generate data
        calib_data, test_data = generate_data(
            distribution_type=dist,
            n_calib=N_CALIBRATION,
            n_test=N_TEST,
            shift_point=SHIFT_POINT,
            shift_magnitude=SHIFT_MAGNITUDE
        )
        
        # Calculate limits for the Shewhart chart
        shewhart_mean, shewhart_ucl, shewhart_lcl = calculate_shewhart_limits(calib_data)
        
        # Calculate limit for the Conformal chart
        conformal_median, conformal_limit = calculate_conformal_limit(calib_data, ALPHA)
        
        # Plot the comparison and get the counts
        num_ooc_shewhart, num_ooc_conformal = plot_comparison(
            test_data=test_data,
            shewhart_mean=shewhart_mean,
            shewhart_ucl=shewhart_ucl,
            shewhart_lcl=shewhart_lcl,
            conformal_median=conformal_median,
            conformal_limit=conformal_limit,
            distribution_name=dist.capitalize(),
            shift_point=SHIFT_POINT
        )
        
        print(f"Results for {dist.capitalize()} Distribution:")
        print(f"  Shewhart Out-of-Control Points: {num_ooc_shewhart}")
        print(f"  Conformal Out-of-Control Points: {num_ooc_conformal}\n")