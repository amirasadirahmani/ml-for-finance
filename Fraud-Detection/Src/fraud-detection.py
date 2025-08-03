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

# Set random seeds for reproducibility
# تنظیم seed برای تکرارپذیری نتایج
torch.manual_seed(42)
np.random.seed(42)

# Load the dataset
df = pd.read_csv('creditcard.csv')

# Basic information about the dataset
print("Dataset shape:", df.shape)

print("Dataset info:")
print(df.info())

print("\nFirst few rows:")
print(df.head())

# Check for missing values
print("\nMissing values:")
print("مقادیر گم‌شده:")
print(df.isnull().sum())

# Check class distribution
print("\nClass distribution:")
print("توزیع کلاس‌ها:")
class_counts = df['Class'].value_counts()
print(class_counts)
print(f"Fraud percentage: {(class_counts[1] / len(df)) * 100:.2f}%")
print(f"درصد تقلب: {(class_counts[1] / len(df)) * 100:.2f}%")

# Visualize class distribution
# تجسم توزیع کلاس‌ها
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
df['Class'].value_counts().plot(kind='bar')
plt.title('Class Distribution\nتوزیع کلاس‌ها')
plt.xlabel('Class (0: Normal, 1: Fraud)\nکلاس (0: عادی، 1: تقلب)')
plt.ylabel('Count\nتعداد')

plt.subplot(1, 2, 2)
df['Class'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Class Distribution Percentage\nدرصد توزیع کلاس‌ها')
plt.ylabel('')

plt.tight_layout()
plt.show()

# Visualize amount distribution for fraud vs normal transactions
# تجسم توزیع مبلغ برای تراکنش‌های تقلبی و عادی
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
df[df['Class'] == 0]['Amount'].hist(bins=50, alpha=0.7, label='Normal\nعادی')
df[df['Class'] == 1]['Amount'].hist(bins=50, alpha=0.7, label='Fraud\nتقلب')
plt.xlabel('Transaction Amount\nمبلغ تراکنش')
plt.ylabel('Frequency\nفرکانس')
plt.title('Amount Distribution by Class\nتوزیع مبلغ بر اساس کلاس')
plt.legend()

plt.subplot(1, 2, 2)
df.boxplot(column='Amount', by='Class')
plt.title('Amount Distribution by Class (Boxplot)\nتوزیع مبلغ بر اساس کلاس (جعبه‌ای)')
plt.suptitle('')

plt.tight_layout()
plt.show()

# Separate features and target
# جداسازی ویژگی‌ها و هدف
X = df.drop('Class', axis=1)
y = df['Class']

# Split the data into train and test sets
# تقسیم داده‌ها به مجموعه آموزش و تست
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]}")
print(f"اندازه مجموعه آموزش: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"اندازه مجموعه تست: {X_test.shape[0]}")

# Initialize and fit the scaler on training data only
# راه‌اندازی و تنظیم نرمال‌سازی فقط روی داده‌های آموزش
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit_transform for training data
                                                # fit_transform برای داده‌های آموزش

X_test_scaled = scaler.transform(X_test)        # transform only for test data
                                                # فقط transform برای داده‌های تست

print("Data normalization completed!")
print("نرمال‌سازی داده‌ها تکمیل شد!")

# Convert to PyTorch tensors
# تبدیل به تنسورهای PyTorch
X_train_tensor = torch.FloatTensor(X_train_scaled)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_train_tensor = torch.FloatTensor(y_train.values)
y_test_tensor = torch.FloatTensor(y_test.values)

print("Data converted to PyTorch tensors!")
print("داده‌ها به تنسورهای PyTorch تبدیل شدند!")


class AdvancedFraudDetectionNN(nn.Module):
    def __init__(self, input_size, hidden_sizes=[256, 128, 64, 32], dropout_rate=0.3):
        """
        Advanced Neural Network for Fraud Detection
        شبکه عصبی پیشرفته برای تشخیص تقلب
        
        Args:
            input_size: Number of input features / تعداد ویژگی‌های ورودی
            hidden_sizes: List of hidden layer sizes / لیست اندازه لایه‌های مخفی
            dropout_rate: Dropout probability / احتمال dropout
        """
        super(AdvancedFraudDetectionNN, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Create hidden layers with batch normalization and dropout
        # ایجاد لایه‌های مخفی با نرمال‌سازی دسته‌ای و dropout
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),    # Batch normalization / نرمال‌سازی دسته‌ای
                nn.ReLU(),                      # ReLU activation / فعال‌سازی ReLU
                nn.Dropout(dropout_rate)        # Dropout for regularization / Dropout برای تنظیم‌سازی
            ])
            prev_size = hidden_size
        
        # Output layer
        # لایه خروجی
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())             # Sigmoid for binary classification / Sigmoid برای طبقه‌بندی دوتایی
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through the network
        عبور رو به جلو از شبکه
        """
        return self.network(x)

# Initialize the model
# راه‌اندازی مدل
input_size = X_train_tensor.shape[1]
model = AdvancedFraudDetectionNN(input_size)

print(f"Model architecture:")
print(f"معماری مدل:")
print(model)

# Count total parameters
# شمارش کل پارامترها
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nTotal parameters: {total_params:,}")
print(f"کل پارامترها: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"پارامترهای قابل آموزش: {trainable_params:,}")


# Calculate class weights for imbalanced dataset
# محاسبه وزن کلاس‌ها برای دیتاست نامتعادل
class_counts = Counter(y_train)
total_samples = len(y_train)
weight_for_0 = total_samples / (2 * class_counts[0])
weight_for_1 = total_samples / (2 * class_counts[1])

print(f"Class weights - Normal: {weight_for_0:.4f}, Fraud: {weight_for_1:.4f}")
print(f"وزن کلاس‌ها - عادی: {weight_for_0:.4f}, تقلب: {weight_for_1:.4f}")

# Create weighted loss function
# ایجاد تابع هزینه وزن‌دار
class_weights = torch.FloatTensor([weight_for_0, weight_for_1])
criterion = nn.BCELoss()  # Binary Cross Entropy Loss / تابع هزینه آنتروپی متقابل دوتایی

# Initialize optimizer with learning rate scheduling
# راه‌اندازی بهینه‌ساز با زمان‌بندی نرخ یادگیری
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # L2 regularization
                                                                        # تنظیم‌سازی L2

# Learning rate scheduler
# زمان‌بند نرخ یادگیری
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10, verbose=True
)

print("Loss function and optimizer initialized!")
print("تابع هزینه و بهینه‌ساز راه‌اندازی شدند!")


# Create custom dataset class
# ایجاد کلاس دیتاست سفارشی
class FraudDataset(Dataset):
    def __init__(self, X, y):
        """
        Custom dataset for fraud detection
        دیتاست سفارشی برای تشخیص تقلب
        """
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create datasets
# ایجاد دیتاست‌ها
train_dataset = FraudDataset(X_train_tensor, y_train_tensor)
test_dataset = FraudDataset(X_test_tensor, y_test_tensor)

# Create data loaders
# ایجاد بارگذارهای داده
batch_size = 512
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Data loaders created with batch size: {batch_size}")
print(f"بارگذارهای داده با اندازه دسته {batch_size} ایجاد شدند")
print(f"Number of training batches: {len(train_loader)}")
print(f"تعداد دسته‌های آموزش: {len(train_loader)}")
print(f"Number of test batches: {len(test_loader)}")
print(f"تعداد دسته‌های تست: {len(test_loader)}")


def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=100):
    """
    Train the fraud detection model
    آموزش مدل تشخیص تقلب
    
    Args:
        model: Neural network model / مدل شبکه عصبی
        train_loader: Training data loader / بارگذار داده‌های آموزش
        criterion: Loss function / تابع هزینه
        optimizer: Optimizer / بهینه‌ساز
        scheduler: Learning rate scheduler / زمان‌بند نرخ یادگیری
        num_epochs: Number of training epochs / تعداد دوره‌های آموزش
    """
    
    model.train()  # Set model to training mode / تنظیم مدل در حالت آموزش
    train_losses = []
    train_accuracies = []
    
    print("Starting training...")
    print("شروع آموزش...")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Zero the gradients
            # صفر کردن گرادیان‌ها
            optimizer.zero_grad()
            
            # Forward pass
            # عبور رو به جلو
            output = model(data).squeeze()
            
            # Calculate loss
            # محاسبه هزینه
            loss = criterion(output, target)
            
            # Backward pass
            # عبور رو به عقب
            loss.backward()
            
            # Update weights
            # به‌روزرسانی وزن‌ها
            optimizer.step()
            
            # Statistics
            # آمارها
            epoch_loss += loss.item()
            predicted = (output > 0.5).float()
            epoch_correct += (predicted == target).sum().item()
            epoch_total += target.size(0)
        
        # Calculate epoch metrics
        # محاسبه معیارهای دوره
        avg_loss = epoch_loss / len(train_loader)
        accuracy = epoch_correct / epoch_total
        
        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)
        
        # Update learning rate
        # به‌روزرسانی نرخ یادگیری
        scheduler.step(avg_loss)
        
        # Print progress
        # نمایش پیشرفت
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
            print(f'دوره [{epoch+1}/{num_epochs}], هزینه: {avg_loss:.4f}, دقت: {accuracy:.4f}')
    
    print("Training completed!")
    print("آموزش تکمیل شد!")
    
    return train_losses, train_accuracies

# Train the model
# آموزش مدل
num_epochs = 100
train_losses, train_accuracies = train_model(
    model, train_loader, criterion, optimizer, scheduler, num_epochs
)


def evaluate_model(model, test_loader, threshold=0.5):
    """
    Evaluate the trained model
    ارزیابی مدل آموزش‌دیده
    
    Args:
        model: Trained model / مدل آموزش‌دیده
        test_loader: Test data loader / بارگذار داده‌های تست
        threshold: Classification threshold / آستانه طبقه‌بندی
    """
    
    model.eval()  # Set model to evaluation mode / تنظیم مدل در حالت ارزیابی
    
    all_predictions = []
    all_probabilities = []
    all_targets = []
    
    print("Evaluating model...")
    print("ارزیابی مدل...")
    
    with torch.no_grad():  # Disable gradient computation / غیرفعال کردن محاسبه گرادیان
        for data, target in test_loader:
            # Forward pass
            # عبور رو به جلو
            output = model(data).squeeze()
            
            # Store results
            # ذخیره نتایج
            all_probabilities.extend(output.cpu().numpy())
            all_predictions.extend((output > threshold).cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_probabilities), np.array(all_targets)

# Evaluate the model
# ارزیابی مدل
predictions, probabilities, targets = evaluate_model(model, test_loader)

# Calculate metrics
# محاسبه معیارها
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(targets, predictions)
precision = precision_score(targets, predictions)
recall = recall_score(targets, predictions)
f1 = f1_score(targets, predictions)
auc_score = roc_auc_score(targets, probabilities)

print("\n" + "="*50)
print("MODEL EVALUATION RESULTS")
print("نتایج ارزیابی مدل")
print("="*50)

print(f"Accuracy / دقت: {accuracy:.4f}")
print(f"Precision / صحت: {precision:.4f}")
print(f"Recall / بازخوانی: {recall:.4f}")
print(f"F1-Score / امتیاز F1: {f1:.4f}")
print(f"AUC-ROC / منطقه زیر منحنی: {auc_score:.4f}")

# Detailed classification report
# گزارش تفصیلی طبقه‌بندی
print("\nDetailed Classification Report:")
print("گزارش تفصیلی طبقه‌بندی:")
print(classification_report(targets, predictions, target_names=['Normal', 'Fraud']))


# Plot training curves
# رسم منحنی‌های آموزش
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(train_losses)
plt.title('Training Loss\nهزینه آموزش')
plt.xlabel('Epoch\nدوره')
plt.ylabel('Loss\nهزینه')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(train_accuracies)
plt.title('Training Accuracy\nدقت آموزش')
plt.xlabel('Epoch\nدوره')
plt.ylabel('Accuracy\nدقت')
plt.grid(True)

# Confusion Matrix
# ماتریس ابهام
plt.subplot(1, 3, 3)
cm = confusion_matrix(targets, predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Fraud'], 
            yticklabels=['Normal', 'Fraud'])
plt.title('Confusion Matrix\nماتریس ابهام')
plt.xlabel('Predicted\nپیش‌بینی‌شده')
plt.ylabel('Actual\nواقعی')

plt.tight_layout()
plt.show()

# ROC Curve
# منحنی ROC
from sklearn.metrics import roc_curve

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
fpr, tpr, _ = roc_curve(targets, probabilities)
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate\nنرخ مثبت کاذب')
plt.ylabel('True Positive Rate\nنرخ مثبت واقعی')
plt.title('ROC Curve\nمنحنی ROC')
plt.legend()
plt.grid(True)

# Precision-Recall Curve
# منحنی صحت-بازخوانی
plt.subplot(1, 2, 2)
precision_vals, recall_vals, _ = precision_recall_curve(targets, probabilities)
plt.plot(recall_vals, precision_vals, label=f'PR Curve')
plt.xlabel('Recall\nبازخوانی')
plt.ylabel('Precision\nصحت')
plt.title('Precision-Recall Curve\nمنحنی صحت-بازخوانی')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Distribution of prediction probabilities
# توزیع احتمالات پیش‌بینی
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
normal_probs = probabilities[targets == 0]
fraud_probs = probabilities[targets == 1]

plt.hist(normal_probs, bins=50, alpha=0.7, label='Normal\nعادی', density=True)
plt.hist(fraud_probs, bins=50, alpha=0.7, label='Fraud\nتقلب', density=True)
plt.xlabel('Prediction Probability\nاحتمال پیش‌بینی')
plt.ylabel('Density\nتراکم')
plt.title('Distribution of Prediction Probabilities\nتوزیع احتمالات پیش‌بینی')
plt.legend()
plt.grid(True)

# Feature importance (using model weights)
# اهمیت ویژگی‌ها (با استفاده از وزن‌های مدل)
plt.subplot(1, 2, 2)
first_layer_weights = model.network[0].weight.data.abs().mean(dim=0).cpu().numpy()
feature_names = [f'Feature {i+1}' for i in range(len(first_layer_weights))]

# Show top 10 features
# نمایش 10 ویژگی برتر
top_indices = np.argsort(first_layer_weights)[-10:]
top_weights = first_layer_weights[top_indices]
top_features = [feature_names[i] for i in top_indices]

plt.barh(range(len(top_weights)), top_weights)
plt.yticks(range(len(top_weights)), top_features)
plt.xlabel('Average Absolute Weight\nمیانگین وزن مطلق')
plt.title('Top 10 Feature Importance\nاهمیت 10 ویژگی برتر')
plt.grid(True)

plt.tight_layout()
plt.show()

