import pandas as pd
import numpy as np
import logging
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import matplotlib.pyplot as plt

class DataProcessor:
    @staticmethod
    def extract_appliances_data(data, appliance_columns):
        """
        This function extracts the appliance data from the original DataFrame and ingnores 
        non-relevant columns from it.

        Args:
          data: The DataFrame with the data.
          appliance_columns: The columns that contain appliance data.

        Returns:
          A DataFrame with only the appliance data.
        """
        try:
            appliances = pd.DataFrame(data, columns=appliance_columns)
            appliances = appliances.fillna(0)
            logging.info("Successfully extracted appliance data")
            return appliances
        except Exception as e:
            logging.error(f"Error extracting appliance data: {e}")
            raise

    @staticmethod
    def process_activities(appliances):
        """
        Processes the appliance data to derive meaningful activities.
        These activities are categorized into instrumental groups like Cooking, Cleaning, etc.

        Args:
          appliances: The DataFrame with the appliance data.

        Returns:
          A DataFrame with the activity data.
        """
        try:
            ADLs = {
                'Cooking': ['Microwave', 'Microwave2', 'Blender', 'Kettle', 'Toaster', 'Bread Maker', 'K Mix'],
                'Cleaning': ['Washing Machine', 'Washing Machine2', 'Tumble Dryer', 'Dishwasher'],
                'Working': ['Computer Site', 'Computer Site2'],
                'Entertainment': ['Television Site', 'Television Site2', 'Hi-Fi', 'Games Console'],
                'House Keeping': ['Electric Heater', 'Electric Heater2', 'Pond Pump']
            }

            for activity, appliance in ADLs.items():
                appliances[activity] = appliances[appliance].sum(axis=1)

            logging.info("Successfully processed activities")
            return appliances
        except Exception as e:
            logging.error(f"Error processing activities: {e}")
            raise

    @staticmethod
    def sum_activities(df, columns):
        """
        This function calculates the sum of the activity columns for each row in a DataFrame and returns a numpy array.
        
        Args:
        df: The DataFrame with the activity data.
        columns: A list of activity column names to be summed.
        
        Returns:
        A numpy array with the summed values for each row.
        """
        try:
            summed_selected_activities = df[columns].sum(axis=1)
            activities_array = summed_selected_activities.to_numpy()
            logging.info("Successfully summed activities")
            return activities_array
        except Exception as e:
            logging.error(f"Error summing activities: {e}")
            raise

    @staticmethod
    def count_frequency(array):
        """
        This function counts the frequency of each unique value in a numpy array and returns a DataFrame.
        We want to know if low values like 0, 1, 2... have a high frequency.
        
        Args:
        array: A numpy array with the values to be counted.
        
        Returns:
        A DataFrame with two columns: 'Value' and 'Frequency'. The DataFrame is sorted by the 'Value' column in ascending order.
        """
        try:
            unique_values, counts = np.unique(array, return_counts=True)
            frequency_dict = dict(zip(unique_values, counts))
            frequency_df = pd.DataFrame(list(frequency_dict.items()), columns=['Value', 'Frequency'])
            frequency_df = frequency_df.sort_values(by='Value')
            logging.info("Successfully counted frequency")
            return frequency_df
        except Exception as e:
            logging.error(f"Error counting frequency: {e}")
            raise

    @staticmethod
    def scale_data(activities_sum, scaler=None):
        """
        Standardizes the activity data using StandardScaler.
        
        Args:
          activities_sum: The DataFrame with the activity data.
          scaler: A fitted StandardScaler instance or None.
        
        Returns:
          A tuple of the scaled data and the fitted scaler instance.
        """    
        try:
            if not scaler:
                scaler = StandardScaler()
                activities_sum_scaled = pd.DataFrame(scaler.fit_transform(activities_sum),
                                                     columns=activities_sum.columns,
                                                     index=activities_sum.index)
            else:
                activities_sum_scaled = pd.DataFrame(scaler.transform(activities_sum),
                                                     columns=activities_sum.columns,
                                                     index=activities_sum.index)
            logging.info("Successfully scaled data")
            return activities_sum_scaled, scaler
        except Exception as e:
            logging.error(f"Error scaling data: {e}")
            raise

    @staticmethod
    def apply_pca(scaled_data, pca=None):
        """
        Reduces the dimensionality of the activity data using PCA.
        
        Args:
          scaled_data: Scaled DataFrame.
          pca: A fitted PCA instance or None.
        
        Returns:
          A tuple of the PCA results and the fitted PCA instance.
        """
        try:
            if not pca:
                pca = PCA(n_components=2)
                reduced_data = pca.fit_transform(scaled_data)
            else:
                reduced_data = pca.transform(scaled_data)
            logging.info("Successfully applied PCA")
            return reduced_data, pca
        except Exception as e:
            logging.error(f"Error applying PCA: {e}")
            raise

    @staticmethod
    def compute_activity_correlations(activities_sum):
        """
        This function computes the correlations between the activities.
        
        Args:
          activities_sum: The DataFrame with the activity data.
        
        Returns:
          The correlation matrix for the activity columns.
        """
        try:
            corr_matrix = pd.DataFrame(activities_sum)
            corr_matrix = corr_matrix.drop(columns=["Aggregate"])
            corr = corr_matrix.corr()
            logging.info("Successfully computed activity correlations")
            return corr
        except Exception as e:
            logging.error(f"Error computing activity correlations: {e}")
            raise

    @staticmethod
    def reset_index_and_extract_hour(df):
        """
        This function resets the index of a DataFrame with a MultiIndex and extracts the hour from the time column.
        
        Args:
          df: The DataFrame with the activity data and a MultiIndex.
        
        Returns:
          A DataFrame with the index reset and a new column 'hour' with the hour of the day.
        """
        try:
            activities_sum_scaled_month2_index = df.reset_index()
            activities_sum_scaled_month2_index['hour'] = activities_sum_scaled_month2_index['Time'].dt.hour
            logging.info("Successfully reset index and extracted hour")
            return activities_sum_scaled_month2_index
        except Exception as e:
            logging.error(f"Error resetting index and extracting hour: {e}")
            raise

    @staticmethod
    def calculate_hourly_distribution_percent(df):
        """
        This function calculates the percentage of instances per hour for each cluster in a DataFrame.
        
        Args:
        df: The DataFrame with the activity data and a column 'hour'.
        
        Returns:
        A DataFrame with the percentage of instances per hour for each cluster. The DataFrame is unstacked and has 'hour' as columns and 'Activity' as index.
        """
        try:
            hourly_counts = df.groupby(['Activity', 'hour']).size()
            total_counts_per_cluster = df.groupby('Activity').size()
            hourly_distribution_percent = hourly_counts.div(total_counts_per_cluster, level='Activity') * 100
            hourly_distribution_percent = hourly_distribution_percent.unstack(fill_value=0)
            logging.info("Successfully calculated hourly distribution percent")
            return hourly_distribution_percent
        except Exception as e:
            logging.error(f"Error calculating hourly distribution percent: {e}")
            raise

    @staticmethod
    def perform_kmeans(reduced_data, param_dist=None):
        """
        Performs KMeans clustering using a grid search to determine the best
        parameters for the number of clusters, initialization method,
        and maximum iterations.
        
        Args:
          reduced_data: The data to cluster.
        
        Returns:
          The parameters of the best KMeans model, the labels, the Silhouette Score 
          and the cluster centers.
        """
        if param_dist is None:
            param_dist = {
                'n_clusters': [4, 5, 6],
                'init': ['k-means++', 'random'],
                'max_iter': [300, 500, 700, 900],
                'n_init': [5, 10, 15, 20],
            }
        kmeans = KMeans(random_state=42)
        random_search = GridSearchCV(kmeans, param_dist, cv=3)
        random_search.fit(reduced_data)
        best_params = random_search.best_params_
        kmeans = KMeans(**best_params, random_state=42).fit(reduced_data)
        labels = kmeans.labels_
        score = silhouette_score(reduced_data, labels)
        return labels, score, best_params, kmeans.cluster_centers_

    @staticmethod
    def perform_gmm(reduced_data, n_components_range=range(4, 10)):
        """
        Performs clustering using a Gaussian Mixture Model (GMM) and determines the optimal number of components
        by evaluating the silhouette score for each possible number of components within the specified range.
        The function iterates over each possible number of components, fits a GMM to the data, and calculates the silhouette score.
        The number of components resulting in the highest silhouette score is selected as the best model. If no model results in more
        than one cluster, a ValueError is raised indicating that silhouette scores could not be computed.
    
        Args:
            reduced_data (array-like): The preprocessed data to cluster. This data should already be scaled or normalized as appropriate.
            n_components_range (range, optional): A range of values specifying the possible numbers of components (clusters) to try.
                Defaults to range(4, 10).
    
        Returns:
            tuple: A tuple containing:
                - labels (array): An array of cluster labels for each data point.
                - highest_silhouette (float): The highest silhouette score obtained among the tested component numbers.
                - best_params (dict): A dictionary containing the best number of components found.
                - means (array): An array of the means of each Gaussian component in the best model.
    
        Raises:
            ValueError: If all models tested result in only one cluster being found, making silhouette score calculation impossible.
        """
        highest_silhouette = -1
        best_n_components = None

        for n_components in n_components_range:
            gmm = GaussianMixture(n_components=n_components, random_state=42)
            gmm.fit(reduced_data)
            labels = gmm.predict(reduced_data)

            if len(set(labels)) > 1:
                silhouette_avg = silhouette_score(reduced_data, labels)
                if silhouette_avg > highest_silhouette:
                    highest_silhouette = silhouette_avg
                    best_n_components = n_components
            else:
                print(f"Only one cluster found with n_components={n_components}, cannot compute Silhouette Score.")

        gmm = GaussianMixture(best_n_components, random_state=42)
        gmm.fit(reduced_data)
        labels = gmm.predict(reduced_data)
        return labels, highest_silhouette, {'n_components': best_n_components}, gmm.means_

    @staticmethod
    def perform_hierarchical_clustering(reduced_data, method='ward', metric='euclidean', plot_dendrogram=False):
        """
        Performs Hierarchical Clustering and optionally plots the dendrogram.
        Additionally, it tests different 'max_d' values to find the best based on Silhouette Score.
        
        Args:
          reduced_data: The data to cluster.
          method: The linkage criterion determines which distance to use between sets of observations.
          metric: The metric to use when calculating distance between instances in a feature array.
          plot_dendrogram: If True, plots the dendrogram.
        
        Returns:
          Cluster labels for each point and the optimal 'max_d' based on Silhouette Score, the score of
          the best value and the linkage Z.
        """
        Z = linkage(reduced_data, method=method, metric=metric)

        distance_thresholds = np.arange(1, 100, 2)
        best_score = -1
        best_threshold = None
        best_clusters = None

        for d in distance_thresholds:
            clusters = fcluster(Z, d, criterion='distance')
            if len(np.unique(clusters)) > 1:
                score = silhouette_score(reduced_data, clusters)
                if score > best_score:
                    best_score = score
                    best_threshold = d
                    best_clusters = clusters

        if plot_dendrogram:
            plt.figure(figsize=(25, 10))
            dendrogram(Z, color_threshold=best_threshold, above_threshold_color='grey')
            plt.axhline(y=best_threshold, c='black', lw=1, linestyle='dashed')
            plt.title('Hierarchical Clustering Dendrogram with Threshold')
            plt.xlabel('Sample index or (cluster size)')
            plt.ylabel('Distance')
            plt.show()

        return best_clusters - 1, best_score, {'max_d': best_threshold}, Z
