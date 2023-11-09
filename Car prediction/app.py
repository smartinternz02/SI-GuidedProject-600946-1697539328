import numpy as np
from flask import Flask, request, render_template, jsonify
import pickle

app = Flask(__name__)

# Load the pickled model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    form_values = request.form.to_dict()
    user_id = int(form_values.get("userid",0))
    gender = 1 if form_values.get("gender", "").lower() == 'male' else 0
    age = int(form_values.get("age", 0))
    salary = int(form_values.get("salary", 0))

    # Exclude the "User ID" feature from the input data
    input = [gender, age, salary]

    # Standardize the data using the loaded scaler
    input_reshape = np.asarray(input).reshape(1, -1)
    std_input = scaler.transform(input_reshape)

    # Predicting the output
    prediction = model.predict(std_input)

    result = "purchase a car" if prediction == 1 else "not purchase a car"

    features_str = ", ".join(map(str, input))
    prediction_text = f"You should {result}"
    # prediction_text = f"You should {result}. Features: {features_str}"

    return render_template("index.html", prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)
