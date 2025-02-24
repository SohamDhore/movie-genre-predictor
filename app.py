from flask import Flask, render_template, request
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load and prepare the dataset
try:
    data = pd.read_csv('movie_data.csv')
    print("Dataset loaded successfully:")
    print(data.head())
except FileNotFoundError:
    print("Error: movie_data.csv not found!")
    raise

# Encode categorical variables
le_gender = LabelEncoder()
le_actor = LabelEncoder()
le_past_genre = LabelEncoder()
le_target = LabelEncoder()

data['gender'] = le_gender.fit_transform(data['gender'])
data['fav_actor'] = le_actor.fit_transform(data['fav_actor'])
data['past_genre'] = le_past_genre.fit_transform(data['past_genre'])
data['preferred_genre'] = le_target.fit_transform(data['preferred_genre'])

# Features and target
X = data[['age', 'gender', 'fav_actor', 'past_genre']]
y = data['preferred_genre']

# Train the decision tree
model = DecisionTreeClassifier(max_depth=3)
model.fit(X, y)
print("Model trained successfully!")

# Home route
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            # Get user input from the form
            age = int(request.form['age'])
            gender = le_gender.transform([request.form['gender']])[0]
            fav_actor = le_actor.transform([request.form['fav_actor']])[0]
            past_genre = le_past_genre.transform([request.form['past_genre']])[0]

            # Predict
            user_input = [[age, gender, fav_actor, past_genre]]
            pred = model.predict(user_input)[0]
            prediction = le_target.inverse_transform([pred])[0]
        except ValueError as e:
            prediction = f"Error: Invalid input - {str(e)}"
        except Exception as e:
            prediction = "Error: Something went wrong."

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)