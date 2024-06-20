# Energy Consumption Analysis Project

This project aims to analyze energy consumption data, identify patterns, and predict activities using various unsupervised learning algorithms and an RNN model. It also identifies the anomalies on it. The project is structured into different modules for data handling, processing, modeling, and utilities.

## Folder and File Descriptions

### `data/`
This directory contains modules related to data handling and processing.
- **`__init__.py`**: Initializes the data package and defines the public interface.
- **`data_handler.py`**: Contains the `DataHandler` class, responsible for reading and relabeling data from CSV files.
- **`data_processor.py`**: Contains the `DataProcessor` class, responsible for processing data, extracting appliance data, performing clustering (KMeans, GMM, Hierarchical), and calculating statistics.

### `models/`
This directory contains modules related to machine learning models.
- **`__init__.py`**: Initializes the models package and defines the public interface.
- **`rnn_model.py`**: Contains the `RNNModel` class, responsible for preparing data, defining, training, and evaluating the RNN model.
- **`clustering.py`**: Contains the `Clustering` class, responsible for evaluating clustering results.

### `utils/`
This directory contains utility modules for directory management, plotting, and other helper functions.
- **`__init__.py`**: Initializes the utils package and defines the public interface.
- **`directory_manager.py`**: Contains the `DirectoryManager` class, responsible for changing directories.
- **`plotting.py`**: Contains the `Plotting` class, responsible for creating various plots, including heatmaps and cluster visualizations.
- **`utility.py`**: Contains the `Utility` class, responsible for various utility functions, such as converting hours to radians and detecting outliers.

### `main.py`
The main execution script for the project. It orchestrates the data processing, clustering, and model training/evaluation flow. It includes:
- Reading and relabeling data.
- Extracting appliance data and processing activities.
- Applying multiple unsupervised learning algorithms (KMeans, GMM, Hierarchical Clustering) and selecting the best one based on Silhouette Score.
- Preparing data for RNN training and evaluating the RNN model.
- Visualizing the results of the anomalies detected.

### `requirements.txt`
A file listing all the Python dependencies required for the project. This ensures that the project can be easily set up with the correct package versions.

### `README.md`
This file provides an overview of the project, including the structure and descriptions of each module.

## General Flow of the Code

1. **Directory Setup**: The working directory is set up using the `DirectoryManager`.
2. **Data Handling**: Data for each house is read and relabeled using the `DataHandler`.
3. **Data Processing**:
   - Appliance data is extracted and processed to identify different activities.
   - Data is scaled and reduced in dimensionality using PCA.
4. **Clustering**:
   - Multiple clustering algorithms (KMeans, GMM, Hierarchical Clustering) are applied to the data.
   - The best clustering algorithm is selected based on the Silhouette Score.
5. **RNN Model**:
   - The data is prepared for training the RNN model.
   - The RNN model is defined, trained, and evaluated.
6. **Visualization**:
   - Clustering results and activity patterns are visualized using various plotting methods.
   - Outliers and anomalies in the patterns are highlighted.

## How to Run the Project

1. Install the required dependencies using:
   ```pip install -r requirements.txt ```
2. Run the main script:
   ```python main.py ```
