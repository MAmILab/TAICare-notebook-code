import os
import pandas as pd
import logging

class DataHandler:
    def __init__(self, num_houses, house_column_mapping, file_suffix):
        self.num_houses = num_houses
        self.house_column_mapping = house_column_mapping
        self.file_suffix = file_suffix

    def read_and_relabel(self):
        """
        Reads data from multiple CSV files, renames the columns based on the provided mapping,
        and concatenates the results into a single DataFrame.
        The resulting DataFrame is sorted by time, and any missing values are filled with zeros.
        This method is deprecated as the houses are now processed individually.

        Args:
          self.num_houses: The number of houses.
          self.house_column_mapping: A dictionary that maps the column names in each house to the new column names.
          self.file_suffix: The suffix for the file names to read (weeks, half a month, or a whole month).

        Returns:
          A combined DataFrame with the data from all houses.
        """
        try:
            data_frames = []
            for i in range(0, self.num_houses + 1):
                filename = f'REDUCED_House{i}_{self.file_suffix}.csv'
                house_data = pd.read_csv(filename, parse_dates=['Time'])
                house_data.drop(columns=['Issues'], errors='ignore', inplace=True)
                house_data.rename(columns=self.house_column_mapping[i], inplace=True)
                house_data['house_id'] = i
                data_frames.append(house_data)

            combined_data = pd.concat(data_frames, ignore_index=True)
            combined_data.fillna(0, inplace=True)
            combined_data.set_index(['Time'], inplace=True)
            combined_data.sort_values(by='Time', inplace=True)
            logging.info("Successfully read and relabeled data")
            return combined_data
        except Exception as e:
            logging.error(f"Error reading and relabeling data: {e}")
            raise

    def read_and_relabel_individual_houses(self):
        """
        Reads data from multiple CSV files, renames the columns based on the provided mapping,
        and stores each house's data in a separate DataFrame within a dictionary.
        Each resulting DataFrame is sorted by time and any missing values are filled with zeros.

        Args:
          self.num_houses: The number of houses.
          self.house_column_mapping: A dictionary that maps the column names in each house to the new column names.
          self.file_suffix: The suffix for the file names to read.

        Returns:
          A dictionary where each key is the house ID and the corresponding value is
          the DataFrame for that house.
        """
        try:
            house_data_dict = {}
            for i in range(0, self.num_houses + 1):
                filename = f'REDUCED_House{i}_{self.file_suffix}.csv'
                house_data = pd.read_csv(filename, parse_dates=['Time'])
                house_data.drop(columns=['Issues'], errors='ignore', inplace=True)
                house_data.rename(columns=self.house_column_mapping[i], inplace=True)
                house_data['house_id'] = i

                house_data.set_index(['Time'], inplace=True)
                house_data.sort_values(by='Time', inplace=True)
                house_data.fillna(0, inplace=True)
                house_data_dict[i] = house_data

            logging.info("Successfully read and relabeled individual houses data")
            return house_data_dict
        except Exception as e:
            logging.error(f"Error reading and relabeling individual houses data: {e}")
            raise
