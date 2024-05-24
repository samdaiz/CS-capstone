import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Create the dataset
data = {
    'Gender': ['Female', 'Female', 'Male', 'Male', 'Female', 'Female', 'Male', 'Male', 'Female', 'Male', 'Male', 'Male', 'Female', 'Female', 'Female', 'Female', 'Female'],
    'Age': [19, 35, 26, 27, 19, 27, 27, 32, 25, 35, 26, 26, 20, 32, 18, 29, 47],
    'Salary': [19000, 20000, 43000, 57000, 76000, 58000, 84000, 150000, 33000, 65000, 80000, 52000, 86000, 18000, 82000, 80000, 25000],
    'Purchased': [0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Encode categorical variable (Gender) into numerical values
df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})

# Split dataset into features and target variable
X = df.drop('Purchased', axis=1)
y = df['Purchased']

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create logistic regression model
model = LogisticRegression()

# Train the model using the training sets
model.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = model.predict(X_test)

# Calculate accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy of model:', accuracy)

# Function to predict whether a new customer will buy the product
def predict_new_customer(model):
    gender = input("Enter gender (Male/Female): ")
    age = int(input("Enter age: "))
    salary = int(input("Enter salary: "))
    gender = 0 if gender.lower() == 'female' else 1
    new_data = [[gender, age, salary]]
    prediction = model.predict(new_data)
    if prediction == 0:
        return 'Not Purchased'
    else:
        return 'Purchased'

# Example usage of the prediction function
new_customer_prediction = predict_new_customer(model)
print('Prediction for new customer:', new_customer_prediction)