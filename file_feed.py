import y_data_preprocess as y_data
import os

directory = input('Please enter the folder:')  # Replace with the actual directory path
year_repo = input('Please enter the year:')
# List all files in the directory
files = os.listdir(directory)

# Filter CSV files with names starting with 'Kelmarsh'
csv_files = [file for file in files if file.endswith('.csv') and file.startswith('Turbine')]

# Print the list of CSV files
for file in csv_files:
    file_path = os.path.join(directory, file)
    report_nam = f"Report_{file[:24]}"
    report_name = f"{report_nam}_{year_repo}"
    y_data.y_data_analyse(file_path, report_name)
    print("Report Published")
