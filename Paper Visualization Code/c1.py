import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

def generate_multivariate_data(n_calib, n_test, shift_point, shift_magnitude):
    """
    Generates 2D multivariate data with an in-control and out-of-control phase.
    
    Args:
        n_calib (int): Number of calibration samples.
        n_test (int): Number of test samples.
        shift_point (int): Index in the test set where the process shifts.
        shift_magnitude (float): The magnitude of the process shift in one dimension.
        
    Returns:
        tuple: A tuple containing calibration_data and test_data.
    """
    # In-control process: 2D Normal distribution centered at (0, 0)
    # --- CORRECTED LINE ---
    mean_in_control = [0,0]
    cov_in_control = [[1, 0.5], [0.5, 1]] # Correlated variables
    
    # Generate calibration data (Phase I)
    calib_data = np.random.multivariate_normal(mean_in_control, cov_in_control, n_calib)
    
    # Generate test data (Phase II)
    test_in_control = np.random.multivariate_normal(mean_in_control, cov_in_control, shift_point)
    
    # Out-of-control process: Shift in the mean of the first variable
    mean_out_of_control = [shift_magnitude, 0]
    test_out_of_control = np.random.multivariate_normal(mean_out_of_control, cov_in_control, n_test - shift_point)
    
    test_data = np.vstack((test_in_control, test_out_of_control))
    
    return calib_data, test_data

def build_autoencoder(input_dim, encoding_dim=1):
    """
    Builds and compiles a simple autoencoder model.
    
    Args:
        input_dim (int): The number of features in the input data (e.g., 2 for 2D).
        encoding_dim (int): The dimension of the compressed representation.
        
    Returns:
        Model: A compiled Keras autoencoder model.
    """
    # Input layer
    input_layer = Input(shape=(input_dim,))
    # Encoder layer
    encoder = Dense(encoding_dim, activation='relu')(input_layer)
    # Decoder layer
    decoder = Dense(input_dim, activation='sigmoid')(encoder)
    
    # Autoencoder model
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer=Adam(0.01), loss='mean_squared_error')
    
    return autoencoder

def plot_p_value_chart(p_values, alpha, shift_point, distribution_name):
    """
    Plots the conformal p-values over time.
    
    Args:
        p_values (np.ndarray): Array of calculated p-values for the test set.
        alpha (float): The significance level.
        shift_point (int): The index where the process shift occurred.
        distribution_name (str): Name of the process for the title.
    """
    # --- Font size definitions ---
    title_fontsize = 16
    label_fontsize = 14
    tick_fontsize = 12
    legend_fontsize = 12
    
    fig, ax = plt.subplots(figsize=(12, 6))
    time_steps = np.arange(1, len(p_values) + 1)
    
    # Plot p-values
    ax.plot(time_steps, p_values, 'ko-', label='Conformal p-value')
    
    # Plot the significance level (control limit)
    ax.axhline(y=alpha, color='red', linestyle='--', label=f'Significance Level (α = {alpha})')
    
    # Highlight points below the control limit
    out_of_control_indices = np.where(p_values < alpha)
    ax.plot(time_steps[out_of_control_indices], p_values[out_of_control_indices], 'ro', markersize=10, label='Anomaly Detected')
    
    # Add a line indicating the process shift
    if shift_point is not None:
        ax.axvline(x=shift_point, color='green', linestyle=':', linewidth=2, label='Process Shift')
        
    ax.set_title(f'Conformal p-Value Chart for {distribution_name}', fontsize=title_fontsize)
    ax.set_xlabel('Time / Sample Number', fontsize=label_fontsize)
    ax.set_ylabel('p-value', fontsize=label_fontsize)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right', fontsize=legend_fontsize)
    ax.grid(True)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # --- Simulation Parameters ---
    N_CALIBRATION = 500      # Number of data points for Phase I calibration
    N_TEST = 200             # Number of data points for Phase II monitoring
    ALPHA = 0.05             # Desired false alarm rate
    SHIFT_POINT = 100        # Point in time where the process shifts
    SHIFT_MAGNITUDE = 3.0    # How much the process mean shifts
    
    # 1. Generate Data
    calib_data, test_data = generate_multivariate_data(N_CALIBRATION, N_TEST, SHIFT_POINT, SHIFT_MAGNITUDE)
    
    # Normalize data to be between 0 and 1 for the autoencoder
    # A simple min-max scaler based on calibration data
    min_vals = calib_data.min(axis=0)
    max_vals = calib_data.max(axis=0)
    calib_data_scaled = (calib_data - min_vals) / (max_vals - min_vals)
    test_data_scaled = (test_data - min_vals) / (max_vals - min_vals)

    # 2. Build and Train the Autoencoder on in-control data
    autoencoder = build_autoencoder(input_dim=2)
    autoencoder.fit(calib_data_scaled, calib_data_scaled, epochs=50, batch_size=32, shuffle=True, verbose=0)
    
    # 3. Calculate Non-Conformity Scores (Reconstruction Error) for Calibration Set
    calib_predictions = autoencoder.predict(calib_data_scaled, verbose=0)
    calib_scores = np.mean(np.abs(calib_data_scaled - calib_predictions), axis=1)
    
    # 4. Calculate Non-Conformity Scores for Test Set
    test_predictions = autoencoder.predict(test_data_scaled, verbose=0)
    test_scores = np.mean(np.abs(test_data_scaled - test_predictions), axis=1)
    
    # 5. Calculate Conformal p-values for each test point
    p_values = np.zeros(len(test_scores))
    for i, test_score in enumerate(test_scores):
        # Count how many calibration scores are greater than or equal to the test score
        count_greater_equal = np.sum(calib_scores >= test_score)
        # Calculate the p-value
        p_values[i] = (count_greater_equal + 1) / (len(calib_scores) + 1)
        
    # 6. Plot the results
    plot_p_value_chart(p_values, ALPHA, SHIFT_POINT, "Multivariate Process")
