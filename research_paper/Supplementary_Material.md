# Supplementary Material for "DetectionGuard: A Robust Federated Learning Intrusion Detection System"

## S1. Extended Dataset Information

**Dataset Name:** CICIDS2017 (Cleaned Version)

**Description:**
The CICIDS2017 dataset contains benign and the most up-to-date common attacks, which resembles the true real-world data (PCAPs). It includes the results of the network traffic analysis using CICFlowMeter with labeled flows based on the time stamp, source, and destination IPs, source and destination ports, protocols and attack (CSV files).

**Statistics:**
- **Total Samples:** 2,520,751
- **Number of Features:** 52
- **Classes:**
    - Normal Traffic (2,095,057 samples)
    - Attack Types: DoS (193,745), DDoS (128,014), Port Scanning (90,694), Brute Force (9,150), Web Attacks (2,143), Bots (1,948)

**Preprocessing Steps:**
1.  **Cleaning:** Removal of infinite and null values.
2.  **Feature Selection:** Dropping the target column for feature set.
3.  **Normalization:** MinMax Scaling to map features to the [0, 1] range.
4.  **Splitting:** 80% Training, 20% Testing using a random seed of 42.

## S2. Additional Experiments & Full Results

### Table S1: Full Performance Comparison (Accuracy)

| % Malicious Clients | Federated Averaging (FedAvg) | DetectionGuard (Proposed) |
| :--- | :--- | :--- |
| 0% | 0.9648 | 0.9636 |
| 30% | 0.8982 | 0.9311 |

### Figure S1: Model Accuracy vs. Percentage of Malicious Clients
![Model Accuracy Comparison](C:/Users/dinus/.gemini/antigravity/brain/4728937a-1c96-4338-aa14-9f69c5f7e898/research_comparison_chart.png)

## S3. Algorithms

### Algorithm S1: Federated Learning Training Loop

```python
# Pseudocode for FL Training Loop
Initialize Global_Model
For round in 1 to ROUNDS:
    Select subset of clients (C_round)
    For each client c in C_round:
        Local_Model_c = Clone(Global_Model)
        Weights_c = c.Train(Local_Model_c, Local_Data_c)
        Store Weights_c
    
    Global_Weights = Aggregation_Function(All_Client_Weights)
    Global_Model.Update(Global_Weights)
    Evaluate Global_Model on Test_Set
```

### Algorithm S2: DetectionGuard Aggregation (Trimmed Mean)

```python
# Pseudocode for DetectionGuard Aggregation
Function DetectionGuard_Aggregation(Client_Weights, Beta):
    # Beta is the trimming parameter (e.g., 0.1 for 10%)
    Num_to_Trim = Beta * Count(Client_Weights)
    
    Aggregated_Weights = []
    For each layer in Model_Layers:
        Layer_Weights = [w[layer] for w in Client_Weights]
        Sorted_Weights = Sort(Layer_Weights)
        
        # Remove the smallest and largest Beta% of weights
        Trimmed_Weights = Sorted_Weights[Num_to_Trim : -Num_to_Trim]
        
        # Compute mean of the remaining weights
        Layer_Mean = Mean(Trimmed_Weights)
        Append Layer_Mean to Aggregated_Weights
        
    Return Aggregated_Weights
```

## S4. Code Snippets

### Preprocessing Logic

```python
def load_and_preprocess(self):
    dataset = pd.read_csv(self.file_path)
    features = dataset.drop([self.config.TARGET_COLUMN], axis=1)
    labels = (dataset[self.config.TARGET_COLUMN] != self.config.NORMAL_LABEL).astype(int)
    
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    
    self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
        scaled_features,
        labels.to_numpy().reshape(-1, 1),
        test_size=self.config.TEST_SPLIT_RATIO,
        random_state=self.config.RANDOM_STATE
    )
```

### Local Client Training

```python
def train(self, global_model):
    local_model = self._build_local_model(global_model)
    # Malicious clients train for more epochs to overfit/poison
    epochs = Config.Training.MALICIOUS_EPOCHS if self.is_malicious else Config.Training.BENIGN_EPOCHS
    target_label = 1 - self.y_data if self.is_malicious else self.y_data # Label flipping
    
    local_model.fit(self.x_data, target_label, epochs=epochs, verbose=0)
    return local_model.get_weights()
```

## S5. Additional Tables

### Table S2: Hyperparameters

| Parameter | Value |
| :--- | :--- |
| Total Clients | 20 |
| Clients per Round | 10 |
| Global Rounds | 10 |
| Batch Size | 32 |
| Optimizer | Adam |
| Loss Function | Binary Crossentropy |
| Hidden Layers | [64, 32] |
| Activation | ReLU |
| Dropout Rate | 0.5 |

## S6. Limitations & Future Work

**Limitations:**
- **Non-IID Data:** The current simulation assumes a relatively uniform distribution of data shards. In real-world IoT scenarios, data is often highly Non-IID, which can degrade FL performance.
- **Advanced Attacks:** The system is tested against Label Flipping attacks. More sophisticated attacks like Backdoor attacks or Model Replacement might require additional defenses.

**Future Work:**
- **Adaptive Trimming:** Implementing a dynamic Beta parameter for Trimmed Mean based on estimated malicious client density.
- **Homomorphic Encryption:** Integrating Secure Multi-Party Computation (SMPC) to further protect model updates from privacy leakage.

## S7. Reproducibility Checklist

- **Hardware:** Standard PC with CPU support (GPU optional but recommended for faster training).
- **Software:** Python 3.9+, TensorFlow 2.x, Scikit-learn, Pandas, Numpy.
- **Random Seed:** Set to `42` for all random number generators (Numpy, Python random, Train-Test split) to ensure result consistency.
