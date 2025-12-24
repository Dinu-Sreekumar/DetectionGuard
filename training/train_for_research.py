import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Config:
    class Files:
        DATASET = Path('../data/cicids2017_cleaned.csv')
        OUTPUT_CHART = Path('../research_outputs/research_comparison_chart.png')

    class Data:
        TARGET_COLUMN = 'Attack Type'
        NORMAL_LABEL = 'Normal Traffic'
        TEST_SPLIT_RATIO = 0.2
        RANDOM_STATE = 42

    class Federation:
        TOTAL_CLIENTS = 20
        DATA_SHARDS = 200
        MIN_SHARDS_PER_CLIENT = 2
        MAX_SHARDS_PER_CLIENT = 10

    class Training:
        ROUNDS = 2
        CLIENTS_PER_ROUND = 10
        BATCH_SIZE = 32
        BENIGN_EPOCHS = 1
        MALICIOUS_EPOCHS = 3
        TRIMMED_MEAN_BETA = 0.1
        MALICIOUS_RATIOS = [0.0, 0.3]

    class Model:
        ACTIVATION = 'relu'
        OUTPUT_ACTIVATION = 'sigmoid'
        HIDDEN_LAYER_1_UNITS = 64
        HIDDEN_LAYER_2_UNITS = 32
        DROPOUT_RATE = 0.5
        OPTIMIZER = 'adam'
        LOSS = 'binary_crossentropy'
        METRICS = ['accuracy']

    class Plotting:
        FIGURE_SIZE = (12, 7)
        BAR_WIDTH = 0.35
        TITLE = 'Model Accuracy vs. Percentage of Malicious Clients'
        X_LABEL = 'Percentage of Malicious Clients'
        Y_LABEL = 'Global Model Accuracy'
        Y_LIMIT = 1.1
        FED_AVG_LABEL = 'Federated Averaging'
        DETECTION_GUARD_LABEL = 'DetectionGuard (Robust)'

class DataManager:
    def __init__(self, file_path, config):
        self.file_path = file_path
        self.config = config
        self.x_train, self.x_test = None, None
        self.y_train, self.y_test = None, None
        self.input_shape = None

    def load_and_preprocess(self):
        print("Step 1: Loading and preprocessing data...")
        dataset = pd.read_csv(self.file_path)
        features = dataset.drop([self.config.TARGET_COLUMN], axis=1)
        labels = (dataset[self.config.TARGET_COLUMN] != self.config.NORMAL_LABEL).astype(int)
        
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(features)
        self.input_shape = scaled_features.shape[1]
        
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            scaled_features,
            labels.to_numpy().reshape(-1, 1),
            test_size=self.config.TEST_SPLIT_RATIO,
            random_state=self.config.RANDOM_STATE
        )
        print("Data successfully preprocessed.")

class Client:
    def __init__(self, client_id, x_data, y_data, is_malicious):
        self.client_id, self.x_data, self.y_data, self.is_malicious = client_id, x_data, y_data, is_malicious

    def train(self, global_model):
        if self.x_data.size == 0:
            return None

        local_model = self._build_local_model(global_model)
        epochs, training_labels = self._get_training_parameters()
        
        local_model.fit(self.x_data, training_labels, epochs=epochs, batch_size=Config.Training.BATCH_SIZE, verbose=0)
        return local_model.get_weights()

    def _build_local_model(self, global_model):
        local_model = tf.keras.models.clone_model(global_model)
        local_model.set_weights(global_model.get_weights())
        local_model.compile(optimizer=Config.Model.OPTIMIZER, loss=Config.Model.LOSS, metrics=Config.Model.METRICS)
        return local_model
    
    def _get_training_parameters(self):
        if self.is_malicious:
            return Config.Training.MALICIOUS_EPOCHS, 1 - self.y_data
        return Config.Training.BENIGN_EPOCHS, self.y_data

class FederatedNetwork:
    def __init__(self, x_train, y_train, fed_config):
        self.x_train, self.y_train, self.fed_config = x_train, y_train, fed_config
        self.clients = []

    def provision_clients(self, malicious_ratio):
        print(f"\nStep 2: Simulating {self.fed_config.TOTAL_CLIENTS} clients ({int(malicious_ratio*100)}% malicious)...")
        num_samples = len(self.x_train)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        shards = np.array_split(indices, self.fed_config.DATA_SHARDS)
        num_malicious_clients = int(self.fed_config.TOTAL_CLIENTS * malicious_ratio)
        
        self.clients.clear()
        current_shard_idx = 0
        for i in range(self.fed_config.TOTAL_CLIENTS):
            num_shards_for_client = np.random.randint(self.fed_config.MIN_SHARDS_PER_CLIENT, self.fed_config.MAX_SHARDS_PER_CLIENT + 1)
            end_index = min(current_shard_idx + num_shards_for_client, self.fed_config.DATA_SHARDS)
            assigned_shards = shards[current_shard_idx:end_index]
            current_shard_idx = end_index
            
            client_indices = np.concatenate(assigned_shards) if assigned_shards else []
            is_malicious = i < num_malicious_clients

            client = Client(i, self.x_train[client_indices], self.y_train[client_indices], is_malicious)
            self.clients.append(client)
        print("Client simulation complete.")

class Aggregator:
    @staticmethod
    def federated_average(client_weights):
        if not client_weights: return None
        avg_weights = [np.mean(np.array(layer_weights), axis=0) for layer_weights in zip(*client_weights)]
        return avg_weights

    @staticmethod
    def detectionguard_aggregation(client_weights):
        if not client_weights: return None
        num_to_trim = int(Config.Training.TRIMMED_MEAN_BETA * len(client_weights))
        
        aggregated_weights = []
        for layer_weights_tuple in zip(*client_weights):
            stacked = np.array(layer_weights_tuple)
            sorted_weights = np.sort(stacked, axis=0)
            
            trimmed = sorted_weights[num_to_trim:-num_to_trim] if num_to_trim > 0 else sorted_weights
            aggregated_weights.append(np.mean(trimmed, axis=0))
        return aggregated_weights

class ExperimentRunner:
    def __init__(self, network, data_manager):
        self.network, self.data_manager = network, data_manager
        self.train_config = Config.Training

    def _create_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(self.data_manager.input_shape,)),
            tf.keras.layers.Dense(Config.Model.HIDDEN_LAYER_1_UNITS, activation=Config.Model.ACTIVATION),
            tf.keras.layers.Dropout(Config.Model.DROPOUT_RATE),
            tf.keras.layers.Dense(Config.Model.HIDDEN_LAYER_2_UNITS, activation=Config.Model.ACTIVATION),
            tf.keras.layers.Dropout(Config.Model.DROPOUT_RATE),
            tf.keras.layers.Dense(1, activation=Config.Model.OUTPUT_ACTIVATION)
        ])
        model.compile(optimizer=Config.Model.OPTIMIZER, loss=Config.Model.LOSS, metrics=Config.Model.METRICS)
        return model

    def run_scenario(self, aggregator, malicious_ratio):
        self.network.provision_clients(malicious_ratio)
        aggregator_name = "DetectionGuard" if "detectionguard" in aggregator.__name__ else "Federated Averaging"
        print(f"\n--- Starting Training: {aggregator_name} ---")
        
        global_model = self._create_model()
        for round_num in range(self.train_config.ROUNDS):
            selected_clients = np.random.choice(self.network.clients, self.train_config.CLIENTS_PER_ROUND, replace=False)
            
            client_weights = [c.train(global_model) for c in selected_clients]
            valid_weights = [w for w in client_weights if w is not None]

            if valid_weights:
                new_weights = aggregator(valid_weights)
                global_model.set_weights(new_weights)
            
            _, accuracy = global_model.evaluate(self.data_manager.x_test, self.data_manager.y_test, verbose=0)
            print(f"Round {round_num + 1}/{self.train_config.ROUNDS} - Global Accuracy: {accuracy:.4f}")

        _, final_accuracy = global_model.evaluate(self.data_manager.x_test, self.data_manager.y_test, verbose=0)
        return final_accuracy

class ResultVisualizer:
    def __init__(self, results, plot_config):
        self.results, self.config = results, plot_config

    def print_table(self):
        print("\n--- Research Paper Results (Table 1) ---")
        header = f"| {'% Malicious':<11} | {self.config.FED_AVG_LABEL + ' Accuracy':<30} | {self.config.DETECTION_GUARD_LABEL + ' Accuracy':<35} |"
        print("=" * len(header)); print(header); print("-" * len(header))
        for i, ratio in enumerate(self.results['malicious_ratio']):
            fed_avg_acc = f"{self.results[self.config.FED_AVG_LABEL][i]:.4f}"
            dg_acc = f"{self.results[self.config.DETECTION_GUARD_LABEL][i]:.4f}"
            print(f"| {int(ratio*100):<11} | {fed_avg_acc:<30} | {dg_acc:<35} |")
        print("=" * len(header))

    def save_chart(self, filename):
        labels = [f'{p*100:.0f}%' for p in self.results['malicious_ratio']]
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=self.config.FIGURE_SIZE)
        
        rects1 = ax.bar(x - self.config.BAR_WIDTH/2, self.results[self.config.FED_AVG_LABEL], self.config.BAR_WIDTH, label=self.config.FED_AVG_LABEL)
        rects2 = ax.bar(x + self.config.BAR_WIDTH/2, self.results[self.config.DETECTION_GUARD_LABEL], self.config.BAR_WIDTH, label=self.config.DETECTION_GUARD_LABEL)
        
        ax.set_ylabel(self.config.Y_LABEL); ax.set_xlabel(self.config.X_LABEL)
        ax.set_title(self.config.TITLE); ax.set_xticks(x, labels); ax.legend()
        ax.set_ylim(0, self.config.Y_LIMIT); ax.bar_label(rects1, padding=3, fmt='%.3f'); ax.bar_label(rects2, padding=3, fmt='%.3f')
        
        fig.tight_layout()
        plt.savefig(filename)
        print(f"\n[SUCCESS] Comparison chart saved as '{filename.name}'.")

def main():
    data_manager = DataManager(Config.Files.DATASET, Config.Data)
    data_manager.load_and_preprocess()

    network = FederatedNetwork(data_manager.x_train, data_manager.y_train, Config.Federation)
    runner = ExperimentRunner(network, data_manager)
    
    results = {'malicious_ratio': Config.Training.MALICIOUS_RATIOS, Config.Plotting.FED_AVG_LABEL: [], Config.Plotting.DETECTION_GUARD_LABEL: []}

    for ratio in Config.Training.MALICIOUS_RATIOS:
        fed_avg_acc = runner.run_scenario(Aggregator.federated_average, ratio)
        results[Config.Plotting.FED_AVG_LABEL].append(fed_avg_acc)
        
        dg_acc = runner.run_scenario(Aggregator.detectionguard_aggregation, ratio)
        results[Config.Plotting.DETECTION_GUARD_LABEL].append(dg_acc)
    
    visualizer = ResultVisualizer(results, Config.Plotting)
    visualizer.print_table()
    visualizer.save_chart(Config.Files.OUTPUT_CHART)

if __name__ == "__main__":
    main()