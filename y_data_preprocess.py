import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns
from datetime import datetime
from sklearn.metrics import mean_absolute_percentage_error
from catboost import CatBoostRegressor
from ydata_profiling import ProfileReport
import os

def y_data_analyse(path_name, file_name):
    csv_name = path_name

    csv_params = {'skiprows':9} # This part is copied from data provider check references
    data_20 = pd.read_csv(csv_name,**csv_params)


    #By previous analysis and not to overload the ydata_profiling method, the method to take mean of these highly correlated fields made sense
    blade_angles = ['Blade angle (pitch position) A (°)', 'Blade angle (pitch position) B (°)', 'Blade angle (pitch position) C (°)']
    data_20['blade_angle'] = data_20[blade_angles].mean(axis=1)

    # These variables feel most relevant to the problem
    use_columns = ['# Date and time','Power (kW)','Wind direction (°)','Nacelle position (°)','blade_angle','Rear bearing temperature (°C)',
                'Rotor speed (RPM)','Generator RPM (RPM)','Nacelle ambient temperature (°C)',
                'Front bearing temperature (°C)','Tower Acceleration X (mm/ss)','Wind speed (m/s)','Tower Acceleration y (mm/ss)','Metal particle count counter',]
    # These variables listed below define the problem
    input_var = '# Date and time'
    output_var = 'Power (kW)'

    #These columns were statistically irrelevant or have huge missing data components
    data_20.drop(data_20.filter(like='Maximum').columns, axis=1, inplace=True)
    data_20.drop(data_20.filter(like='Minimum').columns, axis=1, inplace=True)
    data_20.drop(data_20.filter(like='1').columns, axis=1, inplace=True)
    data_20.drop(data_20.filter(like='2').columns, axis=1, inplace=True)
    data_20.drop(data_20.filter(like='Available Capacity').columns, axis=1, inplace=True)
    data_20.drop(data_20.filter(like='Reactive Energy Export').columns, axis=1, inplace=True)
    data_20.drop(data_20.filter(like='Avail.').columns, axis=1, inplace=True)
    data_20.drop(data_20.filter(like='Potential power').columns, axis=1, inplace=True)
    data_20.drop(data_20.filter(like='Blade angle').columns, axis=1, inplace=True) 

    # drop all the columns except those in usecolumns
    notusecols=[]
    for col in data_20.columns:
        if col not in use_columns:
            notusecols.append(col)
    data_20 = data_20.drop(columns=[col for col in data_20.columns if col not in use_columns])

    profile = ProfileReport(data_20[use_columns])
    # Specify the file name and directory path
    directory_path = "reports"

    # Create the directory if it doesn't exist
    os.makedirs(directory_path, exist_ok=True)

    # Generate the file path
    file_path = os.path.join(directory_path, f"{file_name}.html")

    # Save the file in the specified directory
    profile.to_file(file_path)