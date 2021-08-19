# -*- coding: utf-8 -*-
"""
Utility functions to load data while avoiding CIRCULAR IMPORT anti-pattern.
"""
import pandas as pd

# Function used to load the CSV data:
def load_data(path : str):
    try:
        global df
        df = pd.read_csv(path)
        return df
    
    except:
        df = pd.read_csv(input('Input path to data:'))
        return df

# Function to export the cleaned and augmented data to Excel:
def export_data(dataframe):
    dataframe.to_excel(r'C:/Users/Acer/Desktop/data.xlsx')