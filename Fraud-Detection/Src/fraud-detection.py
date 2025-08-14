import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


torch.manual_seed(42)
np.random.seed(42)

df = pd.read_csv('../Data/creditcard.csv')

print("Dataset shape:", df.shape)

print("Dataset info:")
print(df.info())

print("\nFirst few rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nClass distribution:")
class_counts = df['Class'].value_counts()
print(class_counts)
print(f"Fraud percentage: {(class_counts[1] / len(df)) * 100:.2f}%")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
df['Class'].value_counts().plot(kind='bar')
plt.title('Class Distribution')
plt.xlabel('Class (0: Normal, 1: Fraud)')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
df['Class'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Class Distribution Percentage')
plt.ylabel('')

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
df[df['Class'] == 0]['Amount'].hist(bins=50, alpha=0.7, label='Normal')
df[df['Class'] == 1]['Amount'].hist(bins=50, alpha=0.7, label='Fraud')
plt.xlabel('Transaction Amount')
plt.ylabel('Frequency')
plt.title('Amount Distribution by Class')
plt.legend()

plt.subplot(1, 2, 2)
df.boxplot(column='Amount', by='Class')
plt.title('Amount Distribution by Class (Boxplot)')
plt.suptitle('')

plt.tight_layout()
plt.show()


X = df.drop('Class', axis=1)
y = df['Class']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  

X_test_scaled = scaler.transform(X_test)       
                                                
print("Data normalization completed!")

X_train_tensor = torch.FloatTensor(X_train_scaled)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_train_tensor = torch.FloatTensor(y_train.values)
y_test_tensor = torch.FloatTensor(y_test.values)

print("Data converted to PyTorch tensors!")

class AdvancedFraudDetectionNN(nn.Module):
    def __init__(self, input_size, hidden_sizes=[256, 128, 64, 32], dropout_rate=0.3):
        """
        Advanced Neural Network for Fraud Detection
        
        Args:
            input_size: Number of input features 
            hidden_sizes: List of hidden layer sizes 
            dropout_rate: Dropout probability 
        """
        super(AdvancedFraudDetectionNN, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),    # Batch normalization 
                nn.ReLU(),                      # ReLU activation 
                nn.Dropout(dropout_rate)        # Dropout for regularization 
            ])
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())             # Sigmoid for binary classification
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through the network
        """
        return self.network(x)

input_size = X_train_tensor.shape[1]
model = AdvancedFraudDetectionNN(input_size)

print(f"Model architecture:")
print(model)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

class_counts = Counter(y_train)
total_samples = len(y_train)
weight_for_0 = total_samples / (2 * class_counts[0])
weight_for_1 = total_samples / (2 * class_counts[1])

print(f"Class weights - Normal: {weight_for_0:.4f}, Fraud: {weight_for_1:.4f}")

class_weights = torch.FloatTensor([weight_for_0, weight_for_1])
criterion = nn.BCELoss()  


optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  
                                                                        

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10, verbose=True
)

print("Loss function and optimizer initialized!")

class FraudDataset(Dataset):
    def __init__(self, X, y):        
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = FraudDataset(X_train_tensor, y_train_tensor)
test_dataset = FraudDataset(X_test_tensor, y_test_tensor)

batch_size = 512
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Data loaders created with batch size: {batch_size}")
print(f"Number of training batches: {len(train_loader)}")
print(f"Number of test batches: {len(test_loader)}")

def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=100):
    """
    Train the fraud detection model
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs: Number of training epochs
    """

    model.train()
    train_losses = []
    train_accuracies = []
    
    print("Starting training...")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):            
            optimizer.zero_grad()
            
            output = model(data).squeeze()
                        
            loss = criterion(output, target)
                        
            loss.backward()
                        
            optimizer.step()
                        
            epoch_loss += loss.item()
            predicted = (output > 0.5).float()
            epoch_correct += (predicted == target).sum().item()
            epoch_total += target.size(0)
                
        avg_loss = epoch_loss / len(train_loader)
        accuracy = epoch_correct / epoch_total
        
        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)
                
        scheduler.step(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')        
    
    print("Training completed!")
    
    return train_losses, train_accuracies

num_epochs = 100
train_losses, train_accuracies = train_model(
    model, train_loader, criterion, optimizer, scheduler, num_epochs
)


def evaluate_model(model, test_loader, threshold=0.5):
    """
    Evaluate the trained model
    
    Args:
        model: Trained model
        test_loader: Test data loader
        threshold: Classification threshold
    """

    model.eval()

    all_predictions = []
    all_probabilities = []
    all_targets = []
    
    print("Evaluating model...")

    with torch.no_grad(): 
        for data, target in test_loader:            
            output = model(data).squeeze()                        
            all_probabilities.extend(output.cpu().numpy())
            all_predictions.extend((output > threshold).cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_probabilities), np.array(all_targets)

predictions, probabilities, targets = evaluate_model(model, test_loader)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(targets, predictions)
precision = precision_score(targets, predictions)
recall = recall_score(targets, predictions)
f1 = f1_score(targets, predictions)
auc_score = roc_auc_score(targets, probabilities)

print("\n" + "="*50)
print("MODEL EVALUATION RESULTS")
print("="*50)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")
print(f"AUC-ROC : {auc_score:.4f}")


print("\nDetailed Classification Report:")

print(classification_report(targets, predictions, target_names=['Normal', 'Fraud']))

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(train_losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(train_accuracies)
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)

plt.subplot(1, 3, 3)
cm = confusion_matrix(targets, predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Fraud'], 
            yticklabels=['Normal', 'Fraud'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.show()

# ROC Curve
from sklearn.metrics import roc_curve

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
fpr, tpr, _ = roc_curve(targets, probabilities)
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
precision_vals, recall_vals, _ = precision_recall_curve(targets, probabilities)
plt.plot(recall_vals, precision_vals, label=f'PR Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
normal_probs = probabilities[targets == 0]
fraud_probs = probabilities[targets == 1]

plt.hist(normal_probs, bins=50, alpha=0.7, label='Normal\nعادی', density=True)
plt.hist(fraud_probs, bins=50, alpha=0.7, label='Fraud', density=True)
plt.xlabel('Prediction Probability')
plt.ylabel('Density')
plt.title('Distribution of Prediction Probabilities')
plt.legend()
plt.grid(True)

# Feature importance (using model weights)
plt.subplot(1, 2, 2)
first_layer_weights = model.network[0].weight.data.abs().mean(dim=0).cpu().numpy()
feature_names = [f'Feature {i+1}' for i in range(len(first_layer_weights))]

# Show top 10 features
top_indices = np.argsort(first_layer_weights)[-10:]
top_weights = first_layer_weights[top_indices]
top_features = [feature_names[i] for i in top_indices]

plt.barh(range(len(top_weights)), top_weights)
plt.yticks(range(len(top_weights)), top_features)
plt.xlabel('Average Absolute Weight')
plt.title('Top 10 Feature Importance')
plt.grid(True)

plt.tight_layout()
plt.show()


def predict_fraud(model, scaler, transaction_data):
    """
    Predict fraud for new transaction data
    Args:
        model: Trained model
        scaler: Fitted scaler
        transaction_data: New transaction data
    """
    
    model.eval()
    
    # Normalize the data using the fitted scaler
    transaction_scaled = scaler.transform(transaction_data.reshape(1, -1))
    
    # Convert to tensor
    transaction_tensor = torch.FloatTensor(transaction_scaled)
    
    with torch.no_grad():
        # Get prediction probability
        probability = model(transaction_tensor).item()
        
        # Make prediction            
        prediction = 1 if probability > 0.5 else 0
    
    return prediction, probability

# Example usage with test data
print("\nTesting with sample transactions:")

# Test with a few random samples from test set
sample_indices = np.random.choice(len(X_test), 5, replace=False)

for i, idx in enumerate(sample_indices):
    sample_data = X_test.iloc[idx].values
    actual_label = y_test.iloc[idx]
    
    prediction, probability = predict_fraud(model, scaler, sample_data)
    
    print(f"\nSample {i+1}:")
    print(f"  Actual: {'Fraud' if actual_label == 1 else 'Normal'}")
    print(f"  Predicted: {'Fraud' if prediction == 1 else 'Normal'}")
    print(f"  Probability: {probability:.4f}")
    print(f"  Match: {'✓' if prediction == actual_label else '✗'}")
    
    # Save the trained model and scaler
import pickle

# Save PyTorch model
torch.save({
    'model_state_dict': model.state_dict(),
    'input_size': input_size,
    'model_architecture': {
        'hidden_sizes': [256, 128, 64, 32],
        'dropout_rate': 0.3
    }
}, 'fraud_detection_model.pth')

# Save scaler
with open('fraud_detection_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model and scaler saved successfully!")

# Function to load model for future use
def load_fraud_model(model_path, scaler_path):
    """
    Load trained fraud detection model
    """
    
    # Load model
    checkpoint = torch.load(model_path)
    
    # Recreate model architecture
    model = AdvancedFraudDetectionNN(
        input_size=checkpoint['input_size'],
        hidden_sizes=checkpoint['model_architecture']['hidden_sizes'],
        dropout_rate=checkpoint['model_architecture']['dropout_rate']
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    return model, scaler

print("\nTo load the model in the future, use:")
print("model, scaler = load_fraud_model('fraud_detection_model.pth', 'fraud_detection_scaler.pkl')")



print("\n" + "="*70)
print("FRAUD DETECTION SYSTEM SUMMARY")
print("="*70)

print(f"""
Key Features of this Advanced Fraud Detection System:

1. Deep Neural Network with multiple hidden layers

2. Batch Normalization for stable training

3. Dropout for regularization and preventing overfitting

4. Proper data normalization (fit_transform on train, transform on test)

5. Learning rate scheduling for better convergence

6. Comprehensive evaluation with multiple metrics

7. Visualization of results and model performance

8. Model persistence for future use

Final Model Performance:
- Accuracy: {accuracy:.4f}
- Precision: {precision:.4f}  
- Recall: {recall:.4f}
- F1-Score: {f1:.4f}
- AUC-ROC: {auc_score:.4f}

This system can effectively detect fraudulent transactions with high accuracy
while maintaining good precision and recall balance.
""")
