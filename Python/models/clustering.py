import numpy as np
import pandas as pd
import logging
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

class Clustering:
    @staticmethod
    def evaluate_clustering(X, labels):
        """
        Evaluates the clustering performance using silhouette score, Davies-Bouldin score, and Calinski-Harabasz score.
    
        Args:
            X (numpy.ndarray): The input data with shape (n_samples, n_features).
            labels (numpy.ndarray): The labels for each sample in X.
    
        Returns:
            tuple: A tuple containing the silhouette score, Davies-Bouldin score, and Calinski-Harabasz score.
        """
        try:
            silhouette_avg = silhouette_score(X, labels)
            davies_bouldin = davies_bouldin_score(X, labels)
            calinski_harabasz = calinski_harabasz_score(X, labels)
            logging.info("Successfully evaluated clustering")
            return silhouette_avg, davies_bouldin, calinski_harabasz
        except Exception as e:
            logging.error(f"Error evaluating clustering: {e}")
            raise

    @staticmethod
    def detect_outliers_dbscan(subset, eps=0.1, min_samples=10):
        """
        Detects outliers (anomalies) in a DataFrame using the DBSCAN clustering algorithm.
    
        Args:
            subset (DataFrame): The DataFrame containing the data points to analyze.
            eps (float, optional): The maximum distance between two samples for them to be considered as in the same neighborhood. Defaults to 0.1.
            min_samples (int, optional): The number of samples in a neighborhood for a point to be considered as a core point. Defaults to 10.
    
        Returns:
            subset (DataFrame): The DataFrame with an additional 'outlier' column indicating whether each point is an outlier.
        """
        try:
            clustering_data = subset[['hour_sin', 'hour_cos']]
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(clustering_data)
            subset['outlier'] = db.labels_ == -1
            logging.info("Successfully detected outliers using DBSCAN")
            return subset
        except Exception as e:
            logging.error(f"Error detecting outliers using DBSCAN: {e}")
            raise
