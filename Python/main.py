from data.data_handler import DataHandler
from data.data_processor import DataProcessor
from models.rnn_model import RNNModel
from models.clustering import Clustering
from utils.directory_manager import DirectoryManager
from utils.plotting import Plotting
from utils.utility import Utility
import logging
import time
import numpy as np
import pandas as pd

# This number defines how many houses will be processed in our code
# The bigger the number, the bigger the time to complete the task
NUM_HOUSES = 1

# Mapping of columns for house appliances
house_column_mapping = {
    0: {'chairDesk-energy':'Computer Site', 'dryer-energy':'Tumble Dryer', 'kitchenMicrowave-energy':'Microwave', 'televisionModern-energy':'Television Site', 'washer-energy':'Washing Machine'},
    1: {'Appliance1': 'Fridge', 'Appliance2': 'Freezer1', 'Appliance3': 'Freezer2', 'Appliance4': 'Washer Dryer', 'Appliance5': 'Washing Machine', 'Appliance6': 'Dishwasher', 'Appliance7': 'Computer Site', 'Appliance8': 'Television Site', 'Appliance9': 'Electric Heater'},
    2: {'Appliance1': 'Fridge-Freezer', 'Appliance2': 'Washing Machine', 'Appliance3': 'Dishwasher', 'Appliance4': 'Television Site', 'Appliance5': 'Microwave', 'Appliance6': 'Toaster', 'Appliance7': 'Hi-Fi', 'Appliance8': 'Kettle', 'Appliance9': 'Overhead Fan'},
    3: {'Appliance1': 'Toaster', 'Appliance2': 'Fridge-Freezer', 'Appliance3': 'Freezer', 'Appliance4': 'Tumble Dryer', 'Appliance5': 'Dishwasher', 'Appliance6': 'Washing Machine', 'Appliance7': 'Television Site', 'Appliance8': 'Microwave', 'Appliance9': 'Kettle'},
    4: {'Appliance1': 'Fridge', 'Appliance2': 'Freezer', 'Appliance3': 'Fridge-Freezer', 'Appliance4': 'Washing Machine', 'Appliance5': 'Washing Machine2', 'Appliance6': 'Computer Site', 'Appliance7': 'Television Site', 'Appliance8': 'Microwave', 'Appliance9': 'Kettle'},
    5: {'Appliance1': 'Fridge-Freezer', 'Appliance2': 'Tumble Dryer', 'Appliance3': 'Washing Machine', 'Appliance4': 'Dishwasher', 'Appliance5': 'Computer Site', 'Appliance6': 'Television Site', 'Appliance7': 'Microwave', 'Appliance8': 'Kettle', 'Appliance9': 'Toaster'},
    6: {'Appliance1': 'Freezer', 'Appliance2': 'Washing Machine', 'Appliance3': 'Dishwasher', 'Appliance4': 'Computer Site', 'Appliance5': 'Television Site', 'Appliance6': 'Microwave', 'Appliance7': 'Kettle', 'Appliance8': 'Toaster', 'Appliance9': 'Computer Site2'},
    7: {'Appliance1': 'Fridge', 'Appliance2': 'Freezer1', 'Appliance3': 'Freezer2', 'Appliance4': 'Tumble Dryer', 'Appliance5': 'Washing Machine', 'Appliance6': 'Dishwasher', 'Appliance7': 'Television Site', 'Appliance8': 'Toaster', 'Appliance9': 'Kettle'},
    8: {'Appliance1': 'Fridge', 'Appliance2': 'Freezer', 'Appliance3': 'Washer Dryer', 'Appliance4': 'Washing Machine', 'Appliance5': 'Toaster', 'Appliance6': 'Computer Site', 'Appliance7': 'Television Site', 'Appliance8': 'Microwave', 'Appliance9': 'Kettle'},
    9: {'Appliance1': 'Fridge-Freezer', 'Appliance2': 'Washer Dryer', 'Appliance3': 'Washing Machine', 'Appliance4': 'Dishwasher', 'Appliance5': 'Television Site', 'Appliance6': 'Microwave', 'Appliance7': 'Kettle', 'Appliance8': 'Hi-Fi', 'Appliance9': 'Electric Heater'},
    10: {'Appliance1': 'Blender', 'Appliance2': 'Toaster', 'Appliance3': 'Chest Freezer', 'Appliance4': 'Fridge-Freezer', 'Appliance5': 'Washing Machine', 'Appliance6': 'Dishwasher', 'Appliance7': 'Television Site', 'Appliance8': 'Microwave', 'Appliance9': 'K Mix'},
    11: {'Appliance1': 'Fridge', 'Appliance2': 'Fridge-Freezer', 'Appliance3': 'Washing Machine', 'Appliance4': 'Dishwasher', 'Appliance5': 'Computer Site', 'Appliance6': 'Microwave', 'Appliance7': 'Kettle', 'Appliance8': 'Router', 'Appliance9': 'Hi-Fi'},
    12: {'Appliance1': 'Fridge-Freezer', 'Appliance2': 'Unknown1', 'Appliance3': 'Unknown2', 'Appliance4': 'Computer Site', 'Appliance5': 'Microwave', 'Appliance6': 'Kettle', 'Appliance7': 'Toaster', 'Appliance8': 'Television Site', 'Appliance9': 'Unknown'},
    13: {'Appliance1': 'Television Site', 'Appliance2': 'Freezer', 'Appliance3': 'Washing Machine', 'Appliance4': 'Dishwasher', 'Appliance5': 'Unknown', 'Appliance6': 'Router', 'Appliance7': 'Microwave', 'Appliance8': 'Microwave2', 'Appliance9': 'Kettle'},
    14: {'Appliance1': 'Fridge-Freezer', 'Appliance2': 'Tumble Dryer', 'Appliance3': 'Washing Machine', 'Appliance4': 'Dishwasher', 'Appliance5': 'Food Mixer', 'Appliance6': 'Television Site', 'Appliance7': 'Kettle', 'Appliance8': 'Vivarium', 'Appliance9': 'Pond Pump'},
    15: {'Appliance1': 'Fridge-Freezer', 'Appliance2': 'Tumble Dryer', 'Appliance3': 'Washing Machine', 'Appliance4': 'Dishwasher', 'Appliance5': 'Computer Site', 'Appliance6': 'Television Site', 'Appliance7': 'Microwave', 'Appliance8': 'Hi-Fi', 'Appliance9': 'Toaster'},
    16: {'Appliance1': 'Fridge-Freezer', 'Appliance2': 'Fridge-Freezer2', 'Appliance3': 'Electric Heater', 'Appliance4': 'Electric Heater2', 'Appliance5': 'Washing Machine', 'Appliance6': 'Dishwasher', 'Appliance7': 'Computer Site', 'Appliance8': 'Television Site', 'Appliance9': 'Dehumidifier'},
    17: {'Appliance1': 'Freezer', 'Appliance2': 'Fridge-Freezer', 'Appliance3': 'Tumble Dryer', 'Appliance4': 'Washing Machine', 'Appliance5': 'Computer Site', 'Appliance6': 'Television Site', 'Appliance7': 'Microwave', 'Appliance8': 'Kettle', 'Appliance9': 'Television Site2'},
    18: {'Appliance1': 'Fridge', 'Appliance2': 'Freezer', 'Appliance3': 'Fridge-Freezer', 'Appliance4': 'Washer Dryer', 'Appliance5': 'Washing Machine', 'Appliance6': 'Dishwasher', 'Appliance7': 'Computer Site', 'Appliance8': 'Television Site', 'Appliance9': 'Microwave'},
    19: {'Appliance1': 'Fridge-Freezer', 'Appliance2': 'Washing Machine', 'Appliance3': 'Television Site', 'Appliance4': 'Microwave', 'Appliance5': 'Kettle', 'Appliance6': 'Toaster', 'Appliance7': 'Bread Maker', 'Appliance8': 'Games Console', 'Appliance9': 'Hi-Fi'},
    20: {'Appliance1': 'Fridge', 'Appliance2': 'Freezer', 'Appliance3': 'Tumble Dryer', 'Appliance4': 'Washing Machine', 'Appliance5': 'Dishwasher', 'Appliance6': 'Computer Site', 'Appliance7': 'Television Site', 'Appliance8': 'Microwave', 'Appliance9': 'Kettle'},
}

# Only used in the Google Colab version
#DirectoryManager.change_directory(DATA_PATH)

# Process data for training/testing
data_handler = DataHandler(NUM_HOUSES, house_column_mapping, 'First_Half')
house_data_month2 = data_handler.read_and_relabel_individual_houses()

# Now use the trained model to predict on other data
data_handler.file_suffix = 'Second_Half'
house_data_month3 = data_handler.read_and_relabel_individual_houses()

# Loop through each house's data dictionary
for house_id, house_data in house_data_month2.items():
    start_time = time.time()  # Start timing the processing for this house

    # Calculate correlation matrix for numerical data in house_data
    correlations = house_data.corr(numeric_only=True)
    Plotting.plot_heatmap_correlations(correlations)  # Visualize the correlation matrix as a heatmap

    # Define columns that are relevant to appliances for extraction
    appliance_columns = ['Aggregate', 'Microwave', 'Microwave2', 'Blender', 'Washing Machine', 'Washing Machine2',
                         'Kettle', 'Tumble Dryer', 'Dishwasher', 'Toaster', 'Computer Site', 'Computer Site2',
                         'Television Site', 'Television Site2', 'Electric Heater', 'Electric Heater2', 'Hi-Fi', 'K Mix',
                         'Pond Pump', 'Bread Maker', 'Games Console']
    
    # Extract these columns from house_data
    appliances = DataProcessor.extract_appliances_data(house_data, appliance_columns)
    Plotting.plot_appliance_data(appliances)  # Visualize the appliance data

    # Group appliance data into activity categories
    appliances_grouped = DataProcessor.process_activities(appliances)
    activities = ['Aggregate', 'Cooking', 'Cleaning', 'Working', 'Entertainment', 'House Keeping']
    activities_sum = pd.DataFrame(appliances_grouped, columns=activities)
    
    # Sum activities data to get a total for selected activities
    summed_selected_activities = DataProcessor.sum_activities(activities_sum, ['Cooking', 'Cleaning', 'Working', 'Entertainment', 'House Keeping'])
    activities_array = DataProcessor.sum_activities(activities_sum, ['Cooking', 'Cleaning', 'Working', 'Entertainment', 'House Keeping'])
    
    # Count frequency of activity sums
    frequency_df = DataProcessor.count_frequency(activities_array)
    Plotting.plot_frequency(frequency_df, 'blue', (0, 2), (0, 650000), 2, 5)  # Visualize frequency of activities

    # Filter frequency data for values greater than 10
    values_below_10 = frequency_df.loc[frequency_df['Value'] > 10]
    total_frequency_sum = values_below_10['Frequency'].sum()  # Sum frequencies for filtered values
    print("Total sum of frequencies for values lower than 10:", total_frequency_sum)

    # Filter activities for summed values greater than 10
    high_value_activities_df = activities_sum[summed_selected_activities > 10]
    
    # Scale high value activity data
    activities_sum_scaled, scaler = DataProcessor.scale_data(high_value_activities_df)
    
    # Apply PCA to the scaled data
    reduced_data, pca = DataProcessor.apply_pca(activities_sum_scaled)

    # Dictionary to hold the results of each algorithm
    results = {}

    # Run KMeans clustering and store results
    labels_kmeans, score_kmeans, params_kmeans, centroids_kmeans = DataProcessor.perform_kmeans(reduced_data)
    results['KMeans'] = {'labels': labels_kmeans, 'score': score_kmeans, 'params': params_kmeans, 'centroids': centroids_kmeans}

    # Run Gaussian Mixture Model clustering and store results
    labels_gmm, score_gmm, params_gmm, centroids_gmm = DataProcessor.perform_gmm(reduced_data)
    results['GMM'] = {'labels': labels_gmm, 'score': score_gmm, 'params': params_gmm, 'centroids': centroids_gmm}

    # Run Hierarchical Clustering and store results
    labels_hierarchical, score_hierarchical, params_hierarchical, Z_hierarchical = DataProcessor.perform_hierarchical_clustering(reduced_data, plot_dendrogram=True)
    results['Hierarchical'] = {'labels': labels_hierarchical, 'score': score_hierarchical, 'params': params_hierarchical, 'Z': Z_hierarchical}

    # Select the best clustering algorithm based on the highest Silhouette Score
    best_algorithm = max(results, key=lambda k: results[k]['score'])
    best_result = results[best_algorithm]

    # Output the best algorithm and its score
    print(f"Best algorithm: {best_algorithm} with Silhouette Score: {best_result['score']}")

    # Use the labels and centroids from the best clustering result
    labels = best_result['labels']
    centroids = best_result.get('centroids', None)

    # Visualize the clusters formed in the reduced PCA data space
    Plotting.plot_clusters(reduced_data, labels, centroids)

    # Assign cluster labels to the high value activities DataFrame
    high_value_activities_df.loc[:, 'Activity'] = labels
    activities_sum_scaled['Activity'] = labels

    # Reset index on scaled data and extract hours for further processing
    activities_sum_scaled_index = DataProcessor.reset_index_and_extract_hour(activities_sum_scaled)
    
    # Calculate and visualize hourly distribution percentages
    hourly_distribution_percent = DataProcessor.calculate_hourly_distribution_percent(activities_sum_scaled_index)
    Plotting.plot_heatmap_hours(hourly_distribution_percent)

    # Compute means of activities grouped by clusters
    cluster_means = activities_sum_scaled.groupby('Activity').mean()
    print(cluster_means)

    # Convert activity labels to one-hot encoding for further analysis
    activities_sum_scaled = pd.get_dummies(activities_sum_scaled, columns=['Activity'])
    
    # Compute and display correlation matrix of activities
    corr_matrix = DataProcessor.compute_activity_correlations(activities_sum_scaled)

    # Prepare data for training a recurrent neural network (RNN)
    num_clusters = len(np.unique(labels))
    X_train, X_test, y_train, y_test = RNNModel.prepare_rnn_data(activities_sum_scaled, num_clusters)

    # Define, compile, and train the RNN model
    model = RNNModel.define_rnn_model(X_train.shape[1], y_train.shape[1])
    history = RNNModel.train_rnn_model(model, X_train, y_train, X_test, y_test)
    
    # Visualize training history of the RNN model
    Plotting.plot_training_history(history)

    # Evaluate the trained RNN model
    y_pred = RNNModel.evaluate_model(model, X_test, y_test)
    
    # Rename clusters based on RNN results and correlations
    renamed_clusters, renamed_df = RNNModel.rename_clusters_based_on_rnn_results(y_test, y_pred, corr_matrix)
    
    # Display statistics for linked activities
    RNNModel.show_stats_for_linked_activities(renamed_clusters)

    # Update index and clean up the 'Activity' column in renamed clusters DataFrame
    renamed_clusters.index = y_test.index
    renamed_clusters['Activity'] = renamed_clusters['Activity'].str.replace(r"(&?\s*Activity_\d+)", "", regex=True).str.strip()
    
    # Filter out empty or NaN entries in 'Activity' column
    renamed_clusters = renamed_clusters[renamed_clusters['Activity'].notna() & (renamed_clusters['Activity'] != '')]
    renamed_clusters.dropna(subset=['Activity'], inplace=True)
    
    # Convert index to datetime for proper time handling
    renamed_clusters.index = pd.to_datetime(renamed_clusters.index)
    first_date = renamed_clusters.index.min()
    last_date = renamed_clusters.index.max()

    # Calculate hour of day and convert to circular coordinates for plotting
    renamed_clusters['HourOfDay'] = renamed_clusters.index.hour + renamed_clusters.index.minute / 60 + renamed_clusters.index.second / 3600
    unique_activities = renamed_clusters['Activity'].unique()
    num_activities = len(unique_activities)

    # Convert hour of day to sine and cosine for circular plotting
    renamed_clusters['hour_sin'] = np.sin(2 * np.pi * renamed_clusters['HourOfDay'] / 24)
    renamed_clusters['hour_cos'] = np.cos(2 * np.pi * renamed_clusters['HourOfDay'] / 24)

    # Handle cases where 'Activity' column is missing or all values are NaN
    if 'Activity' not in renamed_clusters.columns or renamed_clusters['Activity'].isnull().all():
        logging.error("Activity column missing or all values are NaN.")
        continue

    # Handle cases where no unique activities are found, which indicates a data processing issue earlier
    if len(unique_activities) == 0:
        logging.error("No unique activities found. Check earlier steps for data processing issues.")
        continue

    # Plot circular distribution and outliers for activities
    Plotting.plot_circular_and_outliers(renamed_clusters, unique_activities, first_date, last_date)

    end_time = time.time()  # End timing the processing
    execution_time = end_time - start_time  # Calculate the total execution time
    logging.info(f"Execution time: {execution_time} seconds")  # Log the execution time
