import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from models.clustering import Clustering
from utils.utility import Utility

class Plotting:
    @staticmethod
    def plot_heatmap_correlations(correlations):
        """
        Displays a heatmap visualization of the given correlation matrix.

        Args:
          correlations: The correlation matrix of the columns from the DataFrame.
        """
        try:
            sns.heatmap(correlations)
            plt.show()
            logging.info("Successfully plotted heatmap correlations")
        except Exception as e:
            logging.error(f"Error plotting heatmap correlations: {e}")
            raise

    @staticmethod
    def plot_appliance_data(appliances):
        """
        This function plots the use of the appliance per hour interactively using Plotly,
        with hover information specific to each appliance.

        Args:
          appliances: The DataFrame with the appliance data.
        """
        try:
            hourly_data = appliances.groupby(appliances.index.get_level_values('Time').hour).mean()
            columns_to_display = [
                'Microwave', 'Microwave2', 'Blender', 'Washing Machine', 'Washing Machine2',
                'Kettle', 'Tumble Dryer', 'Dishwasher', 'Toaster', 'Computer Site',
                'Computer Site2', 'Television Site', 'Television Site2', 'Electric Heater',
                'Electric Heater2', 'Hi-Fi', 'K Mix', 'Pond Pump', 'Bread Maker', 'Games Console'
            ]
            hourly_data = hourly_data[columns_to_display]

            fig = go.Figure()
            for col in columns_to_display:
                fig.add_trace(go.Scatter(x=hourly_data.index, y=hourly_data[col], mode='lines', name=col))

            fig.update_layout(
                title='Average Hourly Usage of Appliances',
                xaxis_title='Hour of Day',
                yaxis_title='Average Usage',
                legend_title='Appliances',
                hovermode='closest'
            )
            fig.show()
            logging.info("Successfully plotted appliance data")
        except Exception as e:
            logging.error(f"Error plotting appliance data: {e}")
            raise

    @staticmethod
    def plot_frequency(df, color, x_range, y_range, x_size, y_size):
        """
        This function plots a bar chart of the frequency distribution of the rows of summed activities.
        
        Args:
        df: The DataFrame with the frequency data. It should have two columns: 'Value' and 'Frequency'.
        color: The color of the bars in the chart.
        x_range: A tuple of two numbers indicating the lower and upper limits of the x-axis.
        y_range: A tuple of two numbers indicating the lower and upper limits of the y-axis.
        """
        try:
            plt.figure(figsize=(x_size, y_size))
            plt.bar(df['Value'], df['Frequency'], color=color)
            plt.xlim(x_range[0], x_range[1])
            plt.ylim(y_range[0], y_range[1])
            plt.xlabel('Summed Activity Values')
            plt.ylabel('Frequency')
            plt.title('Frequency Distribution of Summed Activities')
            plt.grid(True)
            plt.show()
            logging.info("Successfully plotted frequency distribution")
        except Exception as e:
            logging.error(f"Error plotting frequency distribution: {e}")
            raise

    @staticmethod
    def plot_heatmap_hours(df):
        """
        This function plots a heatmap of the percentage hourly distribution of activities across clusters.
    
        Args:
          df: The DataFrame with the percentage hourly distribution of activities across clusters. The DataFrame should have 'hour' as columns and 'Activity' as index.
        """
        try:
            plt.figure(figsize=(15, 10))
            sns.heatmap(df, cmap='coolwarm', annot=True, fmt=".1f")
            plt.title('Percentage Hourly Distribution of Activities Across Clusters')
            plt.xlabel('Hour of Day')
            plt.ylabel('Cluster')
            plt.show()
            logging.info("Successfully plotted heatmap")
        except Exception as e:
            logging.error(f"Error plotting heatmap: {e}")
            raise

    @staticmethod
    def plot_clusters(reduced_data, labels, centroids=None):
        """
        Visualizes the data clusters in a 2D space using a scatter plot, including the centroids.
        Different clusters are distinguished by unique colors, and centroids are marked distinctly.
        
        Args:
          reduced_data: The data to plot.
          labels: The labels of the clusters.
          centroids: The coordinates of the centroids.
        """
        try:
            df = pd.DataFrame(reduced_data, columns=['Dimension 1', 'Dimension 2'])
            df['Activity'] = labels

            plt.figure(figsize=(8, 8))
            sns.scatterplot(data=df, x='Dimension 1', y='Dimension 2', hue='Activity', palette='viridis')

            if centroids is not None:
                for centroid in centroids:
                    plt.scatter(centroid[0], centroid[1], s=100, c='red', marker='X')

            plt.title('Clusters of Activities with Centroids')
            plt.show()
            logging.info("Successfully plotted clusters")
        except Exception as e:
            logging.error(f"Error plotting clusters: {e}")
            raise

    @staticmethod
    def plot_training_history(history):
        """
        Visualizes the training history of the RNN model,
        showing the accuracy over epochs for both training and validation data.
    
        Args:
          history: The training history.
        """
        try:
            plt.plot(history.history['accuracy'], label='Train Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.show()
            logging.info("Successfully plotted training history")
        except Exception as e:
            logging.error(f"Error plotting training history: {e}")
            raise

    @staticmethod
    def plot_comparison(X, labels, title, subplot_index):
        """
        Plots a scatter plot comparing different clusters based on their labels.
    
        Args:
            X (numpy.ndarray): The input data with shape (n_samples, n_features).
            labels (numpy.ndarray): The labels for each sample in X.
            title (str): The title for the plot.
            subplot_index (int): The index of the subplot in the figure.
    
        Returns:
            None: The plot is displayed directly.
        """
        try:
            plt.subplot(subplot_index)
            plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
            plt.title(title)
            logging.info("Successfully plotted comparison")
        except Exception as e:
            logging.error(f"Error plotting comparison: {e}")
            raise

    @staticmethod
    def plot_circular_and_outliers(renamed_clusters, unique_activities, first_date, last_date):
        """
        Plots circular distributions of activities and their outlier occurrences over time,
        each within its own subplot.
            
        The method generates a multi-panel plot with two subplots for each activity:
        1. A circular distribution showing normal and outlier points.
        2. A scatter plot showing the times of outlier occurrences.
    
        Each subplot includes:
        - A circular plot for normal and outlier data points based on their sine and cosine transformations.
        - Annotation of hours around the circle.
        - A time series scatter plot for outliers showing when during the day outliers occur.
    
        This visualization helps in understanding the temporal distribution of different activities and 
        their deviations throughout the day.
    
        Args:
            renamed_clusters (DataFrame): The DataFrame containing clustered data along with 'hour_sin' and 'hour_cos' 
                                          columns representing the circular coordinates, and an 'outlier' boolean column.
            unique_activities (list of str): A list of unique activities to plot, which should correspond to 
                                             the 'Activity' column in the renamed_clusters DataFrame.
            first_date (datetime): The earliest date from which data is considered, used to set plot limits.
            last_date (datetime): The latest date up to which data is considered, used to set plot limits.
        """
        
        fig, axes = plt.subplots(nrows=len(unique_activities), ncols=2, figsize=(15, 10 * len(unique_activities)), squeeze=False)

        for i, activity in enumerate(unique_activities):
            subset = renamed_clusters[renamed_clusters['Activity'] == activity].copy()
            Utility.calculate_marker_sizes(subset)   
            subset = Clustering.detect_outliers_dbscan(subset) 

            # Normal points on the circular plot
            normal_points = subset[~subset['outlier']]
            axes[i, 0].scatter(normal_points['hour_sin'], normal_points['hour_cos'], s=normal_points['marker_size'], alpha=0.6)

            # Outliers on the circular plot
            outlier_points = subset[subset['outlier']]
            axes[i, 0].scatter(outlier_points['hour_sin'], outlier_points['hour_cos'], s=outlier_points['marker_size'], color='red', alpha=0.6, edgecolor='black', label='Outlier')
            axes[i, 0].set_title(f'Cyclical Time Distribution for {activity}')
            axes[i, 0].set_aspect('equal')
            unit_circle = plt.Circle((0, 0), 1, edgecolor='black', fill=False)
            axes[i, 0].add_artist(unit_circle)
            axes[i, 0].set_xlim(-1.5, 1.5)
            axes[i, 0].set_ylim(-1.5, 1.5)

            # Annotating hours on the circular plot
            for hour in range(24):
                radians = Utility.hour_to_radians(hour)
                axes[i, 0].annotate(f'{hour}:00', xy=(np.sin(radians), np.cos(radians)), xytext=(1.2 * np.sin(radians), 1.2 * np.cos(radians)),
                                    arrowprops=dict(arrowstyle="-", color='gray'), ha='center', va='center')

            # Time differences between consecutive outliers
            if not outlier_points.empty:
                outlier_points['DecimalHour'] = outlier_points.index.hour + outlier_points.index.minute / 60
                axes[i, 1].scatter(outlier_points.index, outlier_points['DecimalHour'], color='red', label='Outliers')
                axes[i, 1].set_xlabel('Time')
                axes[i, 1].set_ylabel('Hour of Day')
                axes[i, 1].set_title(f'Outlier Occurrences Over Time for {activity}')
                axes[i, 1].legend()
                axes[i, 1].set_yticks(range(24))
                ytick_labels = [datetime.strptime(str(hour), "%H").strftime("%H:%M") for hour in range(24)]
                axes[i, 1].set_yticklabels(ytick_labels)
                axes[i, 1].set_xlim(first_date, last_date)

        plt.tight_layout()
        plt.show()
