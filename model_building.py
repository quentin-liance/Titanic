import pandas as pd

# Pipeline
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Models 
from sklearn.linear_model import LogisticRegression

# Metrics & model selection
from sklearn.model_selection import RandomizedSearchCV

# Scipy
from scipy.stats import reciprocal

# Save model
import pickle

train = pd.read_csv("dataset/train.csv")

X_train = train.drop(['Survived', 'PassengerId', 'Ticket', 'Cabin', 'Name'], axis=1)
y_train = train['Survived']

num_attribs = ['Age', 'SibSp', 'Parch', 'Fare']
cat_attribs = ['Pclass', 'Sex', 'Embarked']

num_pipeline = Pipeline([
        ('num_imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

cat_pipeline = Pipeline([
    ('cat_imputer', SimpleImputer(strategy="most_frequent")),
    ('ohe', OneHotEncoder()),
])

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs),
    ])

X_train_prepared = full_pipeline.fit_transform(X_train)

## Binary classifier

param_distribs = {
        'penalty': ['l1', 'l2'],
        'C': reciprocal(0.01, 10),
        'solver': ['liblinear'],
        'class_weight' : ['balanced', None],
        'fit_intercept' : [True, False],
    }

lr = LogisticRegression()
rnd_search = RandomizedSearchCV(lr, param_distributions=param_distribs,
                                n_iter=500, cv=5, scoring='accuracy',
                                verbose=3, random_state=42)

rnd_search.fit(X_train_prepared, y_train)
final_model = rnd_search.best_estimator_

with open('model_file.p', 'wb') as f_out: # wb = write, binary
    pickle.dump((full_pipeline, final_model), f_out)