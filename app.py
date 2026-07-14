from flask import Flask, render_template, request, jsonify, session
import pickle
import pandas as pd

app = Flask(__name__)
app.secret_key = "student-performance-secret-key"

# ==========================
# Load Model & Preprocessor
# ==========================

with open("artifacts/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("artifacts/preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)


# ==========================
# Web Pages
# ==========================

@app.route("/")
def landing():
    return render_template("landing.html")


@app.route("/predict")
def predict_form():
    return render_template("predict_form.html")


@app.route("/result")
def result():
    return render_template(
        "prediction_result.html",
        prediction=session.get("prediction"),
        reading=session.get("reading_score"),
        writing=session.get("writing_score"),
        gender=session.get("gender"),
        race=session.get("race_ethnicity"),
        parent_education=session.get("parental_level_of_education"),
        lunch=session.get("lunch"),
        course=session.get("test_preparation_course")
    )


@app.route("/analysis")
def analysis():
    return render_template(
        "result_analysis.html",
        prediction=session.get("prediction"),
        reading=session.get("reading_score"),
        writing=session.get("writing_score"),
        gender=session.get("gender"),
        race=session.get("race_ethnicity"),
        parent_education=session.get("parental_level_of_education"),
        lunch=session.get("lunch"),
        course=session.get("test_preparation_course")
    )


@app.route("/batch")
def batch_prediction():
    return render_template("batch_prediction.html")


# ==========================
# Single Prediction API
# ==========================

@app.route("/api/predict", methods=["POST"])
def predict_api():

    try:

        data = request.get_json()

        input_df = pd.DataFrame({
            "gender": [data["gender"]],
            "race_ethnicity": [data["race_ethnicity"]],
            "parental_level_of_education": [
                data["parental_level_of_education"]
            ],
            "lunch": [data["lunch"]],
            "test_preparation_course": [
                data["test_preparation_course"]
            ],
            "reading_score": [float(data["reading_score"])],
            "writing_score": [float(data["writing_score"])]
        })

        processed_data = preprocessor.transform(input_df)

        prediction = round(
            float(model.predict(processed_data)[0]),
            2
        )

        # Store values in session

        session["prediction"] = prediction
        session["reading_score"] = float(data["reading_score"])
        session["writing_score"] = float(data["writing_score"])
        session["gender"] = data["gender"]
        session["race_ethnicity"] = data["race_ethnicity"]
        session["parental_level_of_education"] = data[
            "parental_level_of_education"
        ]
        session["lunch"] = data["lunch"]
        session["test_preparation_course"] = data[
            "test_preparation_course"
        ]

        return jsonify({
            "success": True,
            "prediction": prediction,
            "redirect": "/result"
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ==========================
# Batch Prediction API
# ==========================

@app.route("/api/batch-predict", methods=["POST"])
def batch_predict_api():

    try:

        if "file" not in request.files:

            return jsonify({
                "success": False,
                "error": "No file uploaded."
            }), 400

        file = request.files["file"]

        df = pd.read_csv(file)

        processed_df = preprocessor.transform(df)

        predictions = model.predict(processed_df)

        df["Predicted_Math_Score"] = predictions

        return jsonify({
             "success": True,
             "rows": df.to_dict(orient="records")
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500




if __name__ == "__main__":
    app.run(debug=True)