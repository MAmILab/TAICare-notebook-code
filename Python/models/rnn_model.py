import numpy as np
import pandas as pd
import logging
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

class RNNModel:
    @staticmethod
    def prepare_rnn_data(activities_sum, range_num):
        """
        Prepares the activity data for RNN training. The data is split into training and test sets,
        scaled to range between 0 and 1, and reshaped to be suitable for RNN input.
        
        Args:
          activities_sum: The DataFrame with the activity data.
        
        Returns:
          A tuple of the training and test data for RNN training.
        """
        try:
            X = activities_sum.drop(columns=[f'Activity_{i}' for i in range(range_num)])
            y = activities_sum[[f'Activity_{i}' for i in range(range_num)]]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            sc = MinMaxScaler(feature_range=(0, 1))
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

            logging.info("Successfully prepared RNN data")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error preparing RNN data: {e}")
            raise

    @staticmethod
    def define_rnn_model(input_shape, output_shape):
        """
        Defines the architecture of the RNN model with LSTM layers and dropout for regularization.
        
        Args:
          input_shape: The shape of the input data.
          output_shape: The number of output classes.
        
        Returns:
          The RNN model.
        """
        try:
            model = tf.keras.Sequential([
                tf.keras.layers.LSTM(units=100, return_sequences=True, input_shape=(input_shape, 1)),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.LSTM(units=100),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(units=output_shape, activation='softmax')
            ])

            model.summary()
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            logging.info("Successfully defined RNN model")
            return model
        except Exception as e:
            logging.error(f"Error defining RNN model: {e}")
            raise

    @staticmethod
    def train_rnn_model(model, X_train, y_train, X_test, y_test):
        """
        Trains the provided RNN model using the training data and validates it using the test data.
        
        Args:
          model: The RNN model.
          X_train: The training data.
          y_train: The labels for the training data.
          X_test: The test data.
          y_test: The labels for the test data.
        
        Returns:
          The training history.
        """
        try:
            history = model.fit(X_train, y_train, epochs=8, batch_size=32, validation_data=(X_test, y_test), verbose=1)
            logging.info("Successfully trained RNN model")
            return history
        except Exception as e:
            logging.error(f"Error training RNN model: {e}")
            raise

    @staticmethod
    def evaluate_model(model, X_test, y_test):
        """
        Evaluates the RNN model's performance on the test data.
        This includes calculating classification metrics
        and visualizing the confusion matrix.
    
        Args:
          model: The RNN model.
          X_test: The test data.
          y_test: The labels for the test data.
        """
        try:
            y_pred = model.predict(X_test)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_test_classes = np.argmax(y_test.values, axis=1)

            unique_classes = np.unique(np.concatenate((y_test_classes, y_pred_classes)))

            print(classification_report(y_test_classes, y_pred_classes, labels=unique_classes))

            cm = confusion_matrix(y_test_classes, y_pred_classes, labels=unique_classes)
            class_names = [f'Activity_{i}' for i in unique_classes]

            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
            disp.plot(xticks_rotation="vertical")
            plt.show()

            logging.info("Successfully evaluated RNN model")
            return y_pred
        except Exception as e:
            logging.error(f"Error evaluating RNN model: {e}")
            raise

    @staticmethod
    def rename_clusters_based_on_rnn_results(y_test, y_pred, precomputed_correlations, corr_threshold=0.6):
        """
        Rename clusters based on the RNN classification results and heatmap correlations.
    
        Args:
        y_test (DataFrame): The true labels in one-hot encoding format.
        y_pred (np.array): The predicted labels from the RNN model.
        precomputed_correlations (DataFrame): Precomputed correlations between activities and clusters.
        corr_threshold (float, optional): The correlation threshold to determine if an activity is dominant in a cluster. Defaults to 0.6.
    
        Returns:
        DataFrame: The DataFrame with the renamed cluster labels.
        """
        try:
            y_test_classes = np.argmax(y_test.values, axis=1)
            y_pred_classes = np.argmax(y_pred, axis=1)

            y_test_onehot = pd.get_dummies(y_test_classes, prefix='Activity')
            y_pred_onehot = pd.get_dummies(y_pred_classes, prefix='Activity')

            plt.figure(figsize=(15, 15))
            sns.heatmap(precomputed_correlations, annot=True, cmap="coolwarm", center=0, linewidths=.5)
            plt.title('Activity-Cluster Precomputed Correlations')
            plt.show()

            cluster_mapping = {}
            for cluster in y_pred_onehot.columns:
                correlations = precomputed_correlations[cluster]
                positive_correlations = correlations[correlations > 0]
                dominant_activities = positive_correlations[positive_correlations >= corr_threshold].index.tolist()
                cluster_name = " & ".join(dominant_activities)
                cluster_mapping[cluster] = cluster_name

            renamed_clusters = []
            for cluster in y_pred_classes:
                cluster_name = 'Activity_' + str(cluster)
                if cluster_name in cluster_mapping:
                    renamed_clusters.append(cluster_mapping[cluster_name])
                else:
                    renamed_clusters.append(cluster_name)

            logging.info("Successfully renamed clusters based on RNN results")
            return pd.DataFrame(renamed_clusters, columns=['Activity']), renamed_clusters
        except Exception as e:
            logging.error(f"Error renaming clusters based on RNN results: {e}")
            raise

    @staticmethod
    def show_stats_for_linked_activities(df):
        """
        Displays the counts of each unique linked activity present in the DataFrame.
    
        Args:
            df (DataFrame): The DataFrame containing the activities.
    
        Prints:
            The count of each unique linked activity found in the DataFrame.
        """
        try:
            linked_activities = df[df['Activity'].str.contains('&', na=False)]
            unique_linked_activities = linked_activities['Activity'].unique()

            if len(unique_linked_activities) == 0:
                logging.info("No linked activities found.")
                return

            logging.info("Counts for each unique linked activity:")
            for activity in unique_linked_activities:
                print('-' * 40)
                count = len(df[df['Activity'] == activity])
                print(f"{activity}: {count}")
        except Exception as e:
            logging.error(f"Error showing stats for linked activities: {e}")
            raise
