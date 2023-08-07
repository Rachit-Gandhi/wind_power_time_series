# wind_power_time_series
Project under summer internship at KPMG 2023 under Management Advisory at Industry 4.0 Team
So below is a guide to use below repo!

We have divided the main repo btw 4 folders:
1. jupyter_notebooks
2. masks
3. one-shot-outputs
4. py_files

We have "tabnet_model_test_1.zip" which is a pretrained tabnet_regression model to use here.

"environment.yml" is a file made to recreate the conda environment without causing issue in all runs.

Now expanding on each folders here:

# jupyter_notebooks
* Ensemble_CNN_LSTM.ipynb is our final creation, it imports from pickle_files_for_turbines which has 65 Mb for each turbines so not on github can be uploaded from personal laptop.
* darts_model_process.ipynb is an intermediate step, darts is among the best timeseries libraries, widely supported and used. Here we stopped proceeding with same considering model fails for larger dataset, but could be used with smaller datasets to try multiple models.
* data_preprocess.ipynb, just fr directory variable declare where there are multiple csvs listed, here we make for turbine_1, but data can be generated for turbine_2-turbine_6 just as easily by using csv_{i}_files. Beyond there we are using datapreprocess.py from py folder, these will create perfect pickle files for data to be used in Ensemble notebook or to be imported into anyother notebook.
* dataloader.ipynb is as name suggests just a dataloader for train and validation dataset which can be used to plug and use for other models.
* model.ipynb is a single point prediction Attention LSTM transformer using tabnet weights, It has 80% accuracy
* multi_layer_LSTM.ipynb is a Multi-Layer LSTM model, not suitable for this dataset, this is just there for use later.
* tabnet_xgboost_mask.ipynb is a tabnet and xgboost regressor to generate masks.

# masks
This has 2 txt files and a png, the png gives feature importance updation persteps by TabNet, as txt files mention on their title each has the numerical masks for feature importance.

# one-shout-outputs
best_predict.png gives best possible output others can be ignored

# py_files
All Jupyter Notebooks have corresponding py files,
* data_preprocess.py is important and has 2 functions, one that selects features and other normalises all columns
* y_data_preprocess generates y_data_profiling reports, be careful of the feeded data, as there is a limit on processing



