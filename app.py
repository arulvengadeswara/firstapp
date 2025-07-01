from flask import Flask, render_template, request, redirect, session
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import time
import csv
import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import plot

app = Flask(__name__)
app.secret_key = 'hi_this_is_harrish_cool!'


@app.route('/check_value_page')
def check_value_page():
    return render_template('recommendation_input.html')


def calculate_percentage_reduction(actual_value, optimal_range_start, optimal_range_end):
    if actual_value < optimal_range_start:
        percentage_reduction = ((optimal_range_end - optimal_range_start) - (actual_value - optimal_range_start)) / (
                    optimal_range_end - optimal_range_start) * 100
    elif actual_value > optimal_range_end:
        percentage_reduction = 100
    else:
        percentage_reduction = 0
    return round(percentage_reduction, 2)


def check_element_levels(nitrogen, phosphorus, sulfur, zinc, iron, manganese, copper, potassium, calcium, magnesium,
                         sodium):
    Crop = session.get('Crop')
    if Crop == "Rice":
        normal_ranges = {"Nitrogen (N)": (150, 250), "Phosphorus (P)": (40, 80), "Potassium (K)": (100, 200),
                         "Sulfur (S)": (10, 20), "Zinc (Zn)": (1, 2), "Iron (Fe)": (20, 80),
                         "Manganese (Mn)": (1, 5), "Copper (Cu)": (0.1, 3), "Calcium (Ca)": (400, 1000),
                         "Magnesium (Mg)": (50, 200), "Sodium (Na)": (4, 20)}
    elif Crop == "Groundnut":
        normal_ranges = {"Nitrogen (N)": (40, 50), "Phosphorus (P)": (20, 30), "Potassium (K)": (30, 40),
                         "Sulfur (S)": (10, 15), "Zinc (Zn)": (1, 2), "Iron (Fe)": (40, 80),
                         "Manganese (Mn)": (2, 5), "Copper (Cu)": (0.2, 1), "Calcium (Ca)": (1000, 2000),
                         "Magnesium (Mg)": (200, 400), "Sodium (Na)": (0, 20)}
    elif Crop == "Black Gram":
        normal_ranges = {"Nitrogen (N)": (25, 35), "Phosphorus (P)": (40, 50), "Potassium (K)": (20, 30),
                         "Sulfur (S)": (10, 15), "Zinc (Zn)": (1, 2), "Iron (Fe)": (40, 80),
                         "Manganese (Mn)": (2, 5), "Copper (Cu)": (0.2, 1.0), "Calcium (Ca)": (1000, 2000),
                         "Magnesium (Mg)": (200, 400), "Sodium (Na)": (0, 20)}
    elif Crop == "Bengal Gram":
        normal_ranges = {"Nitrogen (N)": (25, 35), "Phosphorus (P)": (40, 50), "Potassium (K)": (20, 30),
                         "Sulfur (S)": (10, 15), "Zinc (Zn)": (1, 2), "Iron (Fe)": (40, 80),
                         "Manganese (Mn)": (2, 5), "Copper (Cu)": (0.2, 1.0), "Calcium (Ca)": (1000, 2000),
                         "Magnesium (Mg)": (200, 400), "Sodium (Na)": (0, 20)}

    results = []
    total_reduction = 0
    count = 0
    for element, value in zip(normal_ranges.keys(),
                              [nitrogen, phosphorus, potassium, sulfur, zinc, iron, manganese, copper, calcium,
                               magnesium, sodium]):
        lower_limit, upper_limit = normal_ranges[element]
        percentage_reduction = None  # Initialize percentage_reduction
        if value < lower_limit:
            status = "Below the range"
            if lower_limit != 0:  # Check for division by zero
                percentage_reduction = round(((lower_limit - value) / lower_limit) * 50, 2)
            else:
                percentage_reduction = None
            total_reduction += percentage_reduction
            count += 1
        elif value > upper_limit:
            status = "Above the range"
            if upper_limit != 0:  # Check for division by zero
                percentage_reduction = round(((value - upper_limit) / upper_limit) * 50, 2)
            else:
                percentage_reduction = None
            total_reduction += percentage_reduction
            count += 1
        else:
            status = "Within the range"
        results.append(
            {"Element": element, "Status": status, "Input Value": value, "Percentage Reduction": percentage_reduction})

    if count == 0:
        average_reduction = 0
        message = "No reduction in yield"
    else:
        average_reduction = total_reduction / count
        message = ""

    return results, average_reduction, message


@app.route('/', methods=['GET', 'POST'])
def index():
    with open('Agri-enhancement-System-main/Agri-enhancement-System-main/fertilizer_recommendation.csv', mode='r') as file:
        csv_reader = csv.DictReader(file)
        recommendations = [row for row in csv_reader]

    if request.method == 'POST':
        phosphorus = float(request.form['phosphorus'])
        sulfur = float(request.form['sulfur'])
        zinc = float(request.form['zinc'])
        iron = float(request.form['iron'])
        manganese = float(request.form['manganese'])
        copper = float(request.form['copper'])
        potassium = float(request.form['potassium'])
        calcium = float(request.form['calcium'])
        magnesium = float(request.form['magnesium'])
        sodium = float(request.form['sodium'])
        nitrogen = float(request.form['nitrogen'])
        Crop = request.form['Crop']
        session['Crop'] = Crop

        results = check_element_levels(nitrogen, phosphorus, sulfur, zinc, iron, manganese, copper, potassium, calcium,
                                       magnesium, sodium)

        session['recommendations'] = recommendations
        session['results'] = results[0]
        session['average_reduction'] = round(results[1], 3) if results is not None else None

        return render_template('recommendations_page.html', Crop=Crop,
                               average_reduction=session.get('average_reduction'), results=results[0],
                               message=results[2], recommendations=recommendations)

    return render_template('hero.html')


@app.route('/get_inputs')
def upload_csv():
    return render_template('upload_csv.html')


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']

    if not file:
        return redirect(request.url)

    try:
        file.save('uploaded_file.csv')
        dataset = pd.read_csv('uploaded_file.csv')

        districts = dataset['district'].unique().tolist()
        crops = dataset['crop'].unique().tolist()
        seasons = dataset['season'].unique().tolist()

        time.sleep(9)
        return render_template('prediction_inputs.html', districts=districts, crops=crops, seasons=seasons,
                               crop=session.get('Crop'))
    except pd.errors.ParserError:
        return render_template('upload_csv.html', error='Invalid CSV file format. Please upload a valid CSV file.')
    except KeyError:
        return render_template('upload_csv.html',
                               error='CSV file does not contain required columns. Please check the format and try again.')


@app.route('/predict', methods=['POST'])
def predict():
    average_reduction = session.get('average_reduction', 0.0)

    try:
        year = int(request.form['year'])
        year_get = int(request.form['year'])
        district = request.form['district']
        crop = session.get('Crop')
        season = request.form['season']
        area = float(request.form['area'])
    except ValueError:
        return render_template('prediction_inputs.html', error='Invalid input format. Please enter valid values.')

    dataset = pd.read_csv('uploaded_file.csv')
    y = dataset['production']
    X = dataset[['year', 'district', 'crop', 'season', 'area']]

    scaler = MinMaxScaler()
    y_scaled = scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

    categorical_features = ['district', 'crop', 'season']
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), categorical_features),
        ],
        remainder='passthrough'
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=0))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y_scaled, test_size=0.2, random_state=0)
    pipeline.fit(X_train, y_train)

    input_data = pd.DataFrame({
        'year': [year_get],
        'district': [district],
        'crop': [crop],
        'season': [season],
        'area': [area]
    })

    prediction = pipeline.predict(input_data)
    scaled_prediction = scaler.inverse_transform(prediction.reshape(-1, 1)).flatten()
    yield_prediction = round(scaled_prediction[0] * (1 - average_reduction / 100), 2)

    return render_template('prediction_output.html', prediction=yield_prediction,
                           reduction_percentage=average_reduction)


if __name__ == '__main__':
    app.run(debug=True)
