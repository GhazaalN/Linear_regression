import pandas as pd
import matplotlib.pyplot as plt
import sns as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Load the dataset
data = pd.read_csv('Body Measurements _ original_CSV.csv')

# Explore the dataset
# View the first few rows of the DataFrame
print(data.head())

# Get information about the DataFrame, such as columns and data types
print(data.info())
print(data.columns)

# Handling Missing Values
# Check for missing values
print(data.isnull().sum())

# Impute missing values in numerical columns with the mean
numerical_columns = ['Age', 'HeadCircumference', 'ShoulderWidth', 'ChestWidth ',
       'Belly ', 'Waist ', 'Hips ', 'ArmLength ', 'ShoulderToWaist ',
       'WaistToKnee ', 'LegLength', 'TotalHeight']
imputer = SimpleImputer(strategy='mean')
data[numerical_columns] = imputer.fit_transform(data[numerical_columns])

# Handling Outliers using IQR method
Q1 = data[numerical_columns].quantile(0.25)
Q3 = data[numerical_columns].quantile(0.75)
IQR = Q3 - Q1
threshold = 1.5
outliers = ((data[numerical_columns] < (Q1 - threshold * IQR)) |
            (data[numerical_columns] > (Q3 + threshold * IQR)))
data[outliers] = pd.NA
data.dropna(axis=0, inplace=True)

# Encoding Categorical Variables
categorical_columns = ['Gender']
encoder = OneHotEncoder(sparse=False, drop='first')
encoded_columns = pd.DataFrame(encoder.fit_transform(data[categorical_columns]))
feature_names = encoder.get_feature_names_out(categorical_columns)
encoded_columns.columns = feature_names
data_encoded = pd.concat([data, encoded_columns], axis=1)
data_encoded.drop(columns=categorical_columns, inplace=True)
# Scaling Numerical Variables
scaler = MinMaxScaler()
data_encoded[numerical_columns] = scaler.fit_transform(data_encoded[numerical_columns])

# Check the preprocessed data
print(data_encoded.head())

# Save the preprocessed data to a new CSV file
data_encoded.to_csv('preprocessed_data.csv', index=False)

# Create histograms for Gender, Age, and TotalHeight
plt.figure(figsize=(8, 6))
plt.hist(data_encoded['Gender_2.0'], bins=100)
plt.xlabel('Gender')
plt.ylabel('Frequency')
plt.title('Distribution of Gender')
plt.show()

plt.figure(figsize=(8, 6))
plt.hist(data_encoded['Age'], bins=100)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Distribution of Age')
plt.show()

plt.figure(figsize=(8, 6))
plt.hist(data_encoded['TotalHeight'], bins=100)
plt.xlabel('Total Height')
plt.ylabel('Frequency')
plt.title('Distribution of Total Height')
plt.show()

# Box plot of TotalHeight grouped by Gender
# Box plot of TotalHeight grouped by Gender
plt.figure(figsize=(8, 6))
plt.boxplot([data[data['Gender'] == 1]['TotalHeight'],
             data[data['Gender'] == 2]['TotalHeight']],
            labels=['Male', 'Female'])
plt.xlabel('Gender')
plt.ylabel('Total Height')
plt.title('Distribution of Total Height by Gender')
plt.show()
# Plotting Age vs TotalHeight
plt.figure(figsize=(8, 6))
plt.scatter(data_encoded['Age'], data_encoded['TotalHeight'])
plt.xlabel('Age')
plt.ylabel('Total Height')
plt.title('Age vs Total Height')
plt.show()


# Split the data into input features (X) and target variable (y)
X = data_encoded[['ShoulderToWaist ', 'WaistToKnee ', 'LegLength']]
y = data_encoded['TotalHeight']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the linear regression model
model = LinearRegression()
# Drop rows with missing values
data_encoded.dropna(inplace=True)

# Split the data into training and testing sets
X = data_encoded.drop('TotalHeight', axis=1)
y = data_encoded['TotalHeight']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Plotting actual vs predicted values
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Total Height')
plt.ylabel('Predicted Total Height')
plt.title('Actual vs Predicted Total Height')
plt.show()

# Split the data into training and testing sets
X = data_encoded.drop('TotalHeight', axis=1)
y = data_encoded['TotalHeight']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline to impute missing values in the target and train the model
model = make_pipeline(SimpleImputer(strategy='mean'), LinearRegression())
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Plotting actual vs predicted values
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Total Height')
plt.ylabel('Predicted Total Height')
plt.title('Actual vs Predicted Total Height')
plt.show()