# Day-82-Feature-engineering

###  Overview

**Feature Engineering** is the process of **transforming raw data into meaningful features** that improve the performance of machine learning models.

It is one of the most important steps in building **high-performing ML models**.

---

##  Why Feature Engineering?

* Improves model accuracy
* Helps models learn better patterns
* Converts raw data into usable format

---

##  Common Feature Engineering Techniques

### 1️ Handling Missing Values

```python
df = df.dropna()
```

---

### 2️ Encoding Categorical Variables

#### 🔹 Label Encoding

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df["category"] = le.fit_transform(df["category"])
```

####  One-Hot Encoding

```python
df = pd.get_dummies(df, columns=["category"])
```

---

### 3️ Feature Scaling

#### 🔹 Standardization

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
```

####  Normalization

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)
```

---

### 4️ Feature Creation

Creating new features from existing data.

Example:

```python
df["total_price"] = df["price"] * df["quantity"]
```

---

### 5️ Feature Selection

Selecting important features to improve performance.

Methods:

* Correlation
* Feature importance
* Recursive feature elimination

---

## Benefits

- Better model performance
- Reduced overfitting
- Improved accuracy

---

##  Use Cases

* Machine learning pipelines
* Predictive modeling
* Data preprocessing

