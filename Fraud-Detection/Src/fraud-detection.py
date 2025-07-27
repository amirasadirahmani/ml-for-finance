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

