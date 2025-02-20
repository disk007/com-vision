import kagglehub
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from google.colab import drive

drive.mount('/content/drive')

# เส้นทางไปยังไฟล์ CSV
data = '/content/drive/MyDrive/Colab/apple_quality.csv'

# อ่านไฟล์ CSV
df = pd.read_csv(data)

# ตรวจสอบข้อมูล
print(df.head(10))

# เลือกฟีเจอร์และเป้าหมาย
X = df.drop('Quality', axis=1)  # 'target' เป็นคอลัมน์เป้าหมาย (คุณอาจต้องเช็คชื่อที่แน่นอน)
y = df['Quality']

# แบ่งข้อมูล train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ปรับขนาดข้อมูล
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# สร้างโมเดล KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)

# ทำนายผล
y_pred = knn.predict(X_test_scaled)

# ประเมินผล
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(classification_report(y_test, y_pred))


# สร้าง confusion matrix
cm = confusion_matrix(y_test, y_pred)

# สร้าง heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', 
            xticklabels=['Good', 'Bad'], 
            yticklabels=['Good', 'Bad'])

plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
