import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("iris.csv")

# Display first 5 rows
print(df.head())

# Display dataset information
print(df.info())

# Display dataset shape
print(df.shape)

# Check missing values
print(df.isnull().sum())

# Summary statistics
print(df.describe())

# Histogram
df.hist()
plt.show()

# Box Plot
sns.boxplot(data=df.iloc[:, :4])
plt.show()

# Scatter Plot
sns.scatterplot(x="sepal_length", y="petal_length", hue="species", data=df)
plt.show()

# Heatmap
sns.heatmap(df.iloc[:, :4].corr(), annot=True)
plt.show()

# Bar Graph
df["species"].value_counts().plot(kind="bar")
plt.show()

print("EDA Completed Successfully")