from flask import Flask, request, render_template
import pickle
import numpy as np

# Load trained model
with open("california_house_price.pkl", "rb") as file:
    model = pickle.load(file)

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract data from form
        int_features = [float(x) for x in request.form.values()]
        final_features = [np.array(int_features)]

        # Make prediction
        prediction = model.predict(final_features)
        output = round(prediction[0], 2)

        return render_template("index.html", prediction_text=f"Predicted House Price: ${output}K")
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)
