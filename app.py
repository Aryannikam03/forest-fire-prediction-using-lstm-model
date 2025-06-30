# Importing essential libraries and modules

from flask import Flask, render_template, request, Markup
import numpy as np
import pandas as pd
import seaborn as sns
import requests
import config
import pickle
import io
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt
import plotly.express as px
from models_details import multiple_models
import joblib  # for saving and loading the model
# ==============================================================================================

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

# Loading plant disease classification model

disease_dic= ['5','6', '7', '8','9']



#from model_predict  import pred_leaf_disease

# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------
from import_analyse import basic_info,preprocess_data,eda_plots

# Assuming df is already loaded somewhere globally or within a function

#from recamandation_code import recondation_fn

app = Flask(__name__)


# Load the trained XGBoost model
with open('xgboost_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the mappings for categorical columns
with open('mappings.pkl', 'rb') as file:
    mappings = pickle.load(file)





@ app.route('/')
def home():
    title = 'Crop Yield Prediction Using Machine Learning'
    return render_template('index.html', title=title)

# render crop recommendation form page
@app.route('/preprocessing_data')
def preprocessing_data():
    num_nulls_before, cat_nulls_before, num_nulls_after, cat_nulls_after, head_html = preprocess_data('output.csv')
    return render_template('preprocessing_page.html', num_nulls_before=num_nulls_before, cat_nulls_before=cat_nulls_before, num_nulls_after=num_nulls_after, cat_nulls_after=cat_nulls_after, head=head_html)


@app.route('/eda_data')
def eda_data():
    eda_plots('output.csv')  # Generate plots


    # Return the path to the images for embedding in eda_page.html
    return render_template('eda_page.html', numerical_dist_img='static/images/numerical_distribution.png',
                           categorical_counts_img='static/images/categorical_counts.png',
                           heatmap_img='static/images/heatmap.png')



@app.route('/eda_data2')
def eda_data2():
    # Load the dataset
    df = pd.read_csv('output.csv')  # Use your cleaned dataset
    
    # Drop any unwanted or unnecessary columns
    df.drop('Unnamed: 0', axis=1, inplace=True, errors='ignore')

    df = df.head(2000)  # Optional: Limit dataset to 2000 rows for faster plotting

    # Update column names according to your dataset
    df.columns = ['day', 'month', 'year', 'Temperature', ' RH', ' Ws', 'Rain ', 'FFMC',
                  'DMC', 'DC', 'ISI', 'BUI', 'FWI', 'Classes  ']

    # Separate categorical and numerical columns
    cat_cols = ['Classes  ']  # The target column is categorical
    num_cols = [col for col in df.columns if col not in cat_cols]  # The rest are numerical

    # Record the number of null values before imputation
    num_nulls_before = df[num_cols].isnull().sum().to_dict()
    cat_nulls_before = df[cat_cols].isnull().sum().to_dict()

    # Function for random value imputation on numerical columns
    def random_value_imputation(feature):
        random_sample = df[feature].dropna().sample(df[feature].isna().sum())
        random_sample.index = df[df[feature].isnull()].index
        df.loc[df[feature].isnull(), feature] = random_sample

    # Function for imputing missing categorical values using mode
    def impute_mode(feature):
        mode = df[feature].mode()[0]
        df[feature] = df[feature].fillna(mode)

    # Apply random value imputation for all numerical columns
    for col in num_cols:
        random_value_imputation(col)

    # Apply mode imputation for all categorical columns
    for col in cat_cols:
        impute_mode(col)

    # Record the number of null values after imputation
    num_nulls_after = df[num_cols].isnull().sum().to_dict()
    cat_nulls_after = df[cat_cols].isnull().sum().to_dict()

    # Step 1: Clean the column by stripping spaces and converting to lowercase
    df['Classes  '] = df['Classes  '].str.strip().str.lower()

    df = df[df['Classes  '] != 'classes']

    unique_values = df['Classes  '].unique()
    print("Unique values before replacement:", unique_values)

    # Plotting the violin plot for 'Temperature'
    plt.figure(figsize=(12, 8))
    sns.violinplot(data=df, y='Temperature', x='Classes  ')
    plt.ylabel('Temperature')
    plt.xlabel('Classes')
    plt.savefig('static/images/temperature_violin_plot.png')  # Save the plot
    plt.close()

    # Scatter plot for 'Temperature' vs 'Rain '
    plt.figure(figsize=(12, 8))
    sns.relplot(data=df, x='Temperature', y='Rain ', kind='scatter', hue='Classes  ')
    plt.xlabel('Temperature')
    plt.ylabel('Rain ')
    plt.savefig('static/images/temperature_vs_rain_scatter_plot.png')  # Save the plot
    plt.close()

    # Line plot for 'Temperature' vs 'FWI'
    sns.relplot(data=df, x='Temperature', y='FWI', kind='line', hue='Classes  ')
    plt.xlabel('Temperature')
    plt.ylabel('FWI')
    plt.savefig('static/images/temperature_vs_fwi_line_plot.png')  # Save the plot
    plt.close()

    # Line plot for 'Rain ' vs 'FFMC'
    sns.relplot(data=df, x='Rain ', y='FFMC', kind='line', hue='Classes  ')
    plt.xlabel('Rain ')
    plt.ylabel('FFMC')
    plt.savefig('static/images/rain_vs_ffmc_line_plot.png')  # Save the plot
    plt.close()

    # Return the path to the images for embedding in the HTML template
    return render_template('eda_page2.html', 
                           numerical_dist_img='static/images/temperature_violin_plot.png',
                           categorical_counts_img='static/images/temperature_vs_rain_scatter_plot.png',
                           heatmap_img='static/images/temperature_vs_fwi_line_plot.png',
                           heatmap_img2='static/images/rain_vs_ffmc_line_plot.png')


@app.route('/models_data')
def models_data():
    results=multiple_models('output.csv')
    #return render_template('index.html', results=results)
    return render_template('models_dt.html',results=results)

@app.route('/test_application')
def test_application():
    #return render_template('recommendation.html')
        # Pass the unique values for state, district, season, crop, and soil to the HTML form
    return render_template('recommendation.html', 
                           states=mappings['state_names'].keys(),
                           districts=mappings['district_names'].keys(),
                           seasons=mappings['season_names'].keys(),
                           crops=mappings['crop_names'].keys(),
                           soils=mappings['soil_type'].keys())



@app.route('/disease-predict2', methods=['GET', 'POST'])
def disease_prediction2():
    title = 'Crop Yield Prediction Using Machine Learning'
    return render_template('rust.html', title=title)



@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
            title = 'Crop Price Prediction Using Machinelearning'

    #if request.method == 'POST':
        #if 'file' not in request.files:
         #   return redirect(request.url)

            file = request.files.get('file')

           # if not file:
            #    return render_template('disease.html', title=title)

            #img = Image.open(file)
            file.save('output.csv')

#df.head(),df.shape,df.describe(),df.info()
            #df2=basic_info("output.csv")



            #table = df2.to_html(classes="table table-striped table-hover", border=0)
            head, shape, describe, info = basic_info('output.csv')
            return render_template('rust-result.html', head=head, shape=shape, describe=describe, info=info)
#prediction =pred_leaf_disease("output.BMP")

            #prediction = (str(disease_dic[prediction]))

           # print("print the blood group of the candidate ",prediction)

            #if prediction=="5":
            #        class_rust=5


            #elif prediction=="6":
            #        class_rust=6 


            #elif prediction=="7":
            #        class_rust=7



            #elif prediction=="8":
            #        class_rust=8



            #elif prediction=="9":
            #        class_rust="There is noe Corrosion"


           # return render_template('rust-result.html',table=table,title=title)
        #except:
         #   pass
    


# render disease prediction result page
# Load the saved model

loaded_model = joblib.load('random_forest_model.joblib')

#with open('xgb2.pkl', 'rb') as file:
#    loaded_model = pickle.load(file)

# Now you can use `loaded_model` to make predictions



@app.route('/predict1', methods=['POST'])
def predict1():
    if request.method == 'POST':
        try:
            # Get form data
            day = int(request.form['day'])
            month = int(request.form['month'])
            year = int(request.form['year'])
            temperature = float(request.form['Temperature'])
            RH = float(request.form['RH'])
            Ws = float(request.form['Ws'])
            Rain = float(request.form['Rain'])
            FFMC = float(request.form['FFMC'])
            DMC = float(request.form['DMC'])
            DC = float(request.form['DC'])
            ISI = float(request.form['ISI'])
            BUI = float(request.form['BUI'])
            FWI = float(request.form['FWI'])

            # Prepare input for the model
            features = np.array([[day, month, year, temperature, RH, Ws, Rain, FFMC, DMC, DC, ISI, BUI, FWI]])

            # Make prediction
            prediction = loaded_model.predict(features)
            predicted_class = prediction[0]  # Assuming the model returns a class label

            # Pass the prediction to the HTML form
            return render_template('recommendation.html', 
                                   prediction_text=f'Predicted Class: {predicted_class}',
                                   day=day, month=month, year=year,
                                   temperature=temperature, RH=RH, Ws=Ws, Rain=Rain,
                                   FFMC=FFMC, DMC=DMC, DC=DC, ISI=ISI, BUI=BUI, FWI=FWI)
        
        except Exception as e:
            return str(e)


# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=True)
