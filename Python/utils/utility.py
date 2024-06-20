import numpy as np
import logging

class Utility:
    @staticmethod
    def hour_to_radians(hour):
        """
        Converts an hour to radians, accounting for the cyclical nature of time within a 24-hour period.
    
        Args:
            hour (int or float): The hour to convert.
    
        Returns:
            float: The hour converted to radians.
        """
        try:
            radians = hour / 24.0 * 2.0 * np.pi
            logging.info(f"Converted hour {hour} to radians {radians}")
            return radians
        except Exception as e:
            logging.error(f"Error converting hour to radians: {e}")
            raise

    @staticmethod
    def calculate_marker_sizes(subset, scale_factor=10):
        """
        Calculates the marker size for each data point in a subset based on the count of activities per hour.
    
        Args:
            subset (DataFrame): The DataFrame containing the data points and their corresponding hours.
            scale_factor (int, optional): The factor by which to scale the marker size. Defaults to 10.
    
        Modifies:
            subset (DataFrame): Adds a 'marker_size' column to the DataFrame with calculated sizes.
        """
        try:
            hour_counts = subset.groupby('HourOfDay').size()
            subset['marker_size'] = subset['HourOfDay'].map(hour_counts) * scale_factor
            logging.info("Successfully calculated marker sizes")
        except Exception as e:
            logging.error(f"Error calculating marker sizes: {e}")
            raise
