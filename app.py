from flask import Flask, render_template, request
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            data = CustomData(
                age=int(request.form.get("age")),
                gender=request.form.get("gender"),
                study_hours_per_day=float(request.form.get("study_hours_per_day")),
                social_media_hours=float(request.form.get("social_media_hours")),
                netflix_hours=float(request.form.get("netflix_hours")),
                part_time_job=request.form.get("part_time_job"),
                attendance_percentage=float(request.form.get("attendance_percentage")),
                sleep_hours=float(request.form.get("sleep_hours")),
                diet_quality=request.form.get("diet_quality"),
                exercise_frequency=int(request.form.get("exercise_frequency")),
                parental_education_level=request.form.get("parental_education_level"),
                internet_quality=request.form.get("internet_quality"),
                mental_health_rating=int(request.form.get("mental_health_rating")),
                extracurricular_participation=request.form.get("extracurricular_participation")
            )

            input_df = data.get_data_as_df()
            print(input_df)

            predict_pipeline = PredictPipeline()
            prediction = predict_pipeline.predict(input_df)
            predicted_score = round(prediction[0], 2)

            return render_template('result.html', score=predicted_score)

        except Exception as e:
            return f"Something went wrong: {e}"

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0",port=80)
