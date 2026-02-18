import numpy as np
import matplotlib.pyplot as plt

#Functions for scoring

def absolute_error(y_true, y_pred):
    """
    Calculates the absolute error.
    Used in c1.py (Autoencoder reconstruction) and c2.py (residuals).
    """
    return np.abs(y_true - y_pred)

def studentized_score(y_true, y_pred, local_volatility):
    """
    Calculates the 'studentized' score by normalizing the error by local volatility.
    Used in c2.py for the Adaptive Conformal Prediction.
    """
    safe_volatility = np.maximum(local_volatility, 1e-6)
    return np.abs(y_true - y_pred) / safe_volatility

def median_deviation(data, median=None):
    """
    Calculates the absolute deviation from the median.
    Used in c3.py as a robust, distribution-free score.
    """
    if median is None:
        median = np.median(data)
    return np.abs(data - median)


#Functions for calibration

def get_conformal_threshold(calibration_scores, alpha=0.05):
    """
    Calculates the (1-alpha) quantile with the finite-sample correction.
    
    Logic derived from c3.py:
    The threshold is the value q such that P(score <= q) >= 1 - alpha.
    """
    n = len(calibration_scores)
    
    # Finite sample correction: q_level = (1 - alpha) * (1 + 1/n)
    quantile_level = (1 - alpha) * (1 + 1/n)
    
    # Clip to max 1.0 to avoid errors if n is small
    quantile_level = min(quantile_level, 1.0)
    
    return np.quantile(calibration_scores, quantile_level)

def compute_p_value(test_score, calibration_scores):
    """
    Calculates the Conformal p-value for a single new data point.
    
    """
    n = len(calibration_scores)
    # How many calibration points were 'stranger' than this test point?
    count_greater_equal = np.sum(calibration_scores >= test_score)
    
    return (count_greater_equal + 1) / (n + 1)


#Functions for monitoring

def is_anomaly(score, threshold):
    """
    Binary check: Is the score higher than the calibrated threshold?
    Used in c3.py.
    """
    return score > threshold

def get_adaptive_interval(prediction, threshold, local_volatility):
    """
    Constructs the prediction interval [Lower, Upper].
    
    Logic derived from c2.py:
    Interval = Prediction +/- (Threshold * Local_Volatility)
    """
    margin = threshold * local_volatility
    return prediction - margin, prediction + margin

#Functions for simple visualizations

STYLES = {
    'title_size': 16,
    'label_size': 14,
    'tick_size': 12,
    'legend_size': 12
}

def plot_p_values(p_values, alpha=0.05, shift_point=None, title="Conformal p-Values"):
    """
    Plots the p-value evolution (from c1.py).
    """
    p_values = np.array(p_values)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(p_values, 'ko-', label='p-value', markersize=4)
    ax.axhline(alpha, color='red', linestyle='--', label=f'Alpha ({alpha})')
    
    # Highlight anomalies
    anomalies = np.where(p_values < alpha)[0]
    ax.plot(anomalies, p_values[anomalies], 'ro', label='Anomaly')
    
    if shift_point:
        ax.axvline(shift_point, color='green', linestyle=':', label='Shift')
        
    ax.set_title(title, fontsize=STYLES['title_size'])
    ax.set_ylabel("p-value", fontsize=STYLES['label_size'])
    ax.legend(fontsize=STYLES['legend_size'])
    plt.tight_layout()
    plt.show()

def plot_adaptive_intervals(data, predictions, lower, upper, shift_point=None):
    """
    Plots data with dynamic intervals and uncertainty width (from c2.py).
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Main chart
    ax1.plot(data, 'k-', alpha=0.6, label='Actual')
    ax1.plot(predictions, 'b--', label='Prediction')
    ax1.fill_between(range(len(data)), lower, upper, color='blue', alpha=0.2, label='Interval')
    ax1.set_title("Adaptive Conformal Intervals", fontsize=STYLES['title_size'])
    
    # Uncertainty chart
    width = upper - lower
    ax2.plot(width, 'r-', label='Interval Width')
    ax2.set_title("Uncertainty (Interval Width)", fontsize=STYLES['title_size'])
    
    if shift_point:
        ax1.axvline(shift_point, color='green', linestyle=':')
        ax2.axvline(shift_point, color='green', linestyle=':')
        
    plt.tight_layout()
    plt.show()
    
    
    
    
