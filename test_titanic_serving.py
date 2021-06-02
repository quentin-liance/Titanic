import requests

passenger = {'Pclass': 1,
 'Sex': 'male',
 'Age': 40.0,
 'SibSp': 1,
 'Parch': 0,
 'Fare': 7.25,
 'Embarked': 'S'}

url = 'http://localhost:9696/predict'
response = requests.post(url, json=passenger)
result = response.json()
print(result)