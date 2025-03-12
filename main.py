import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt 
import seaborn as sns 
import warnings 
warnings.filterwarnings('ignore')
df = pd.read_csv(r'C:\Users\Qc\Desktop\AI Powered Job Recomendation\job_recommendation_dataset.csv')
plt.figure(figsize=(10,6)) 
sns.boxplot(x='Experience Level',y='Salary',data=df)
plt.title('The experience level wise salary')
plt.show()
top_job = df['Job Title'].value_counts().sort_values(ascending=False).head(6)

plt.figure(figsize=(10,6))
top_job.plot(kind='pie',autopct='%1.1f%%')
plt.show()
df['Required Skills'].value_counts().head(10).plot(kind='pie',autopct='%1.1f%%')
plt.show()
# Assuming y_test and y_pred are defined somewhere in your code
# For demonstration, let's create some dummy data
y_test = np.random.rand(100) * 100000  # Replace with actual y_test data
y_pred = np.random.rand(100) * 100000  # Replace with actual y_pred data

plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel('Actual Salary')
plt.ylabel('Predicted Salary')
plt.title('Actual vs Predicted Salary')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.tight_layout()
plt.show()
# 📌 6. Most Required Skills (Word Cloud)
from wordcloud import WordCloud
skills_text = ' '.join(df['Required Skills'].dropna())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(skills_text)

# Create subplots
fig, axs = plt.subplots(3, 2, figsize=(15, 10))

# Display WordCloud
axs[2, 1].imshow(wordcloud, interpolation='bilinear')
axs[2, 1].axis('off')
axs[2, 1].set_title("Most Required Skills")

plt.tight_layout()
plt.show()
