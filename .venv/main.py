import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score


# This is my try on a Kaggle Competition called Space Titanic https://www.kaggle.com/competitions/spaceship-titanic/overview
# Creating our train and test Data from CSV files
train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

# Select Features
features = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

# Split Data into Train and Validation Data
X_train, y_train = train_data[features], train_data['Transported']
X_valid = test_data[features]

# Selecting categorical and numerical Columns
numerical_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
categorical_cols = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP']

# Preprocessing Data to fill missing values and One hot encode categorical data
numerical_transformer = SimpleImputer(strategy='most_frequent')

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Creating our model
model = RandomForestClassifier(n_estimators=200, random_state=1)

my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                              ])

my_pipeline.fit(X_train, y_train)

# Cross validating our Model for testing
score = cross_val_score(my_pipeline, X_train, y_train, cv=5, scoring="accuracy")
print(score.mean())

# Creating our Output to submit to the Competition
test_pred = my_pipeline.predict(X_valid)

output = pd.DataFrame({'PassengerId': test_data.PassengerId,
                       'Transported': test_pred.astype(bool)})
output.to_csv('submission.csv', index=False)