import pandas as pd
import csv

def extract_sort_first_column(input_file, output_file, read_delimiter=';', write_delimiter=';', skip_rows=0):
    df = pd.read_csv(input_file, delimiter=read_delimiter, skiprows=skip_rows, usecols=[0], dtype=str)
    df = df.sort_values(by=df.columns[0]).reset_index(drop=True)
    df.to_csv(output_file, index=False, header=False, sep=write_delimiter, quoting=csv.QUOTE_MINIMAL)

extract_sort_first_column('output.csv', 'output_converted.csv', read_delimiter=';', write_delimiter=';')
extract_sort_first_column('Default.txt', 'default_converted.csv', read_delimiter='\t', write_delimiter=';', skip_rows=2)