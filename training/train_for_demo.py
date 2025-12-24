import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.utils import to_categorical
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Config:
    class Paths:
        DATASET = Path('../data/cicids2017_cleaned.csv')
        ASSETS_DIR = Path('../assets/')
        MODEL = ASSETS_DIR / 'DetectionGuard_model.h5'
        SCALER = ASSETS_DIR / 'scaler.gz'
        ENCODER = ASSETS_DIR / 'encoder.gz'
        PLAYBOOK_TEMPLATE = ASSETS_DIR / 'TP_{name}_Attacks.csv'

    class Data:
        TARGET_COLUMN = 'Attack Type'
        IGNORE_LABEL = 'Normal Traffic'
        TEST_SPLIT_RATIO = 0.2
        RANDOM_STATE = 42

    class Federation:
        TOTAL_CLIENTS = 20
        DATA_SHARDS = 200
        MIN_SHARDS_PER_CLIENT = 2
        MAX_SHARDS_PER_CLIENT = 10

    class Training:
        ROUNDS = 10
        CLIENTS_PER_ROUND = 10
        BATCH_SIZE = 32
        EPOCHS = 1
        TRIMMED_MEAN_BETA = 0.1

    class Model:
        ACTIVATION = 'relu'
        OUTPUT_ACTIVATION = 'softmax'
        HIDDEN_LAYER_1_UNITS = 64
        HIDDEN_LAYER_2_UNITS = 32
        DROPOUT_RATE = 0.5
        OPTIMIZER = 'adam'
        LOSS = 'categorical_crossentropy'
        METRICS = ['accuracy']

class DataManager:
    def __init__(self, config):
        self.config = config
        self.raw_df = None
        self.encoder = LabelEncoder()
        self.scaler = MinMaxScaler()
        self.num_classes = 0
        self.input_shape = 0
        self.x_train, self.x_test, self.y_train, self.y_test = [None] * 4
        self.df_train, self.df_test = None, None

    def load_and_preprocess(self):
        print("Step 1: Loading and preprocessing data for multi-class classification...")
        self.raw_df = pd.read_csv(self.config.Paths.DATASET)
        
        features = self.raw_df.drop([self.config.Data.TARGET_COLUMN], axis=1)
        labels = self.raw_df[self.config.Data.TARGET_COLUMN]

        encoded_labels = self.encoder.fit_transform(labels)
        one_hot_labels = to_categorical(encoded_labels)
        self.num_classes = len(self.encoder.classes_)
        print(f"Found {self.num_classes} classes: {list(self.encoder.classes_)}")
        
        scaled_features = self.scaler.fit_transform(features)
        self.input_shape = scaled_features.shape[1]
        
        self.x_train, self.x_test, self.y_train, self.y_test, self.df_train, self.df_test = train_test_split(
            scaled_features,
            one_hot_labels,
            self.raw_df,
            test_size=self.config.Data.TEST_SPLIT_RATIO,
            random_state=self.config.Data.RANDOM_STATE
        )
        print("Data preprocessed successfully.")

class Client:
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def train(self, global_model, train_config, model_builder):
        if self.x_data.size == 0:
            return None

        local_model = model_builder()
        local_model.set_weights(global_model.get_weights())
        
        local_model.fit(
            self.x_data,
            self.y_data,
            epochs=train_config.EPOCHS,
            batch_size=train_config.BATCH_SIZE,
            verbose=0
        )
        return local_model.get_weights()

class FederatedNetwork:
    def __init__(self, x_train, y_train, fed_config):
        self.x_train = x_train
        self.y_train = y_train
        self.fed_config = fed_config
        self.clients = []

    def provision_clients(self):
        print(f"\nStep 2: Simulating {self.fed_config.TOTAL_CLIENTS} federated clients...")
        num_samples = len(self.x_train)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        shards = np.array_split(indices, self.fed_config.DATA_SHARDS)
        
        self.clients.clear()
        current_shard_idx = 0
        for _ in range(self.fed_config.TOTAL_CLIENTS):
            num_shards_for_client = np.random.randint(
                self.fed_config.MIN_SHARDS_PER_CLIENT, self.fed_config.MAX_SHARDS_PER_CLIENT + 1
            )
            end_index = min(current_shard_idx + num_shards_for_client, self.fed_config.DATA_SHARDS)
            assigned_shards = shards[current_shard_idx:end_index]
            current_shard_idx = end_index
            
            client_indices = np.concatenate(assigned_shards) if assigned_shards else []
            client = Client(self.x_train[client_indices], self.y_train[client_indices])
            self.clients.append(client)
        print("Client simulation complete.")

class ModelTrainer:
    def __init__(self, data_manager, network, config):
        self.data = data_manager
        self.network = network
        self.config = config

    def _build_model(self):
        model_config = self.config.Model
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(self.data.input_shape,)),
            tf.keras.layers.Dense(model_config.HIDDEN_LAYER_1_UNITS, activation=model_config.ACTIVATION),
            tf.keras.layers.Dropout(model_config.DROPOUT_RATE),
            tf.keras.layers.Dense(model_config.HIDDEN_LAYER_2_UNITS, activation=model_config.ACTIVATION),
            tf.keras.layers.Dropout(model_config.DROPOUT_RATE),
            tf.keras.layers.Dense(self.data.num_classes, activation=model_config.OUTPUT_ACTIVATION)
        ])
        model.compile(optimizer=model_config.OPTIMIZER, loss=model_config.LOSS, metrics=model_config.METRICS)
        return model

    def _detectionguard_aggregation(self, client_weights):
        if not client_weights: return None
        num_clients = len(client_weights)
        num_to_trim = int(self.config.Training.TRIMMED_MEAN_BETA * num_clients)
        
        aggregated_weights = []
        for weights_list_tuple in zip(*client_weights):
            stacked = np.array(weights_list_tuple)
            sorted_weights = np.sort(stacked, axis=0)
            
            if num_to_trim > 0:
                trimmed = sorted_weights[num_to_trim:-num_to_trim]
            else:
                trimmed = sorted_weights
                
            aggregated_weights.append(np.mean(trimmed, axis=0))
        return aggregated_weights

    def train_with_federated_learning(self):
        print("\nStep 3: Starting final model training with robust federated learning...")
        global_model = self._build_model()

        for round_num in range(self.config.Training.ROUNDS):
            global_weights = global_model.get_weights()
            selected_clients = np.random.choice(self.network.clients, self.config.Training.CLIENTS_PER_ROUND, replace=False)
            
            client_weights = [c.train(global_model, self.config.Training, self._build_model) for c in selected_clients]
            valid_weights = [w for w in client_weights if w is not None]

            if valid_weights:
                new_weights = self._detectionguard_aggregation(valid_weights)
                global_model.set_weights(new_weights)
            
            _, accuracy = global_model.evaluate(self.data.x_test, self.data.y_test, verbose=0)
            print(f"Round {round_num + 1}/{self.config.Training.ROUNDS} - Global Accuracy: {accuracy:.4f}")

        _, final_accuracy = global_model.evaluate(self.data.x_test, self.data.y_test, verbose=0)
        print(f"\nFinal model training complete. Test Accuracy: {final_accuracy:.4f}")
        return global_model

class AssetManager:
    def __init__(self, data_manager, config):
        self.data = data_manager
        self.config = config

    def save_all(self, model):
        print("\nStep 4: Saving final model and preprocessing assets...")
        self.config.Paths.ASSETS_DIR.mkdir(exist_ok=True)
        model.save(self.config.Paths.MODEL)
        joblib.dump(self.data.scaler, self.config.Paths.SCALER)
        joblib.dump(self.data.encoder, self.config.Paths.ENCODER)
        print("Model, scaler, and encoder saved successfully.")

    def generate_playbooks(self, model):
        print("\nStep 5: Generating attack playbooks from true positives...")
        predictions = model.predict(self.data.x_test)
        predicted_indices = np.argmax(predictions, axis=1)
        true_indices = np.argmax(self.data.y_test, axis=1)

        for i, class_name in enumerate(self.data.encoder.classes_):
            if class_name == self.config.Data.IGNORE_LABEL:
                continue

            mask = (true_indices == i) & (predicted_indices == i)
            true_positives_df = self.data.df_test[mask]
            
            if not true_positives_df.empty:
                self._save_playbook(class_name, true_positives_df)

    def _save_playbook(self, class_name, data):
        safe_name = class_name.replace(' ', '_').replace('/', '_')
        filename = self.config.Paths.PLAYBOOK_TEMPLATE.with_name(f"TP_{safe_name}_Attacks.csv")
        data.to_csv(filename, index=False)
        print(f"Saved {len(data)} '{class_name}' samples to '{filename.name}'.")

def main():
    config = Config()
    
    data_manager = DataManager(config)
    data_manager.load_and_preprocess()
    
    network = FederatedNetwork(data_manager.x_train, data_manager.y_train, config.Federation)
    network.provision_clients()
    
    trainer = ModelTrainer(data_manager, network, config)
    final_model = trainer.train_with_federated_learning()
    
    if final_model:
        asset_manager = AssetManager(data_manager, config)
        asset_manager.save_all(final_model)
        asset_manager.generate_playbooks(final_model)

if __name__ == "__main__":
    main()