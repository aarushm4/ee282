import pandas as pd
import sqlite3

def import_data(csv_file):
    conn = sqlite3.connect('variants.db')
    data = pd.read_csv(csv_file)
    data.to_sql('variant_annotations', conn, if_exists='append', index=False)
    conn.close()


if __name__ == '__main__':
    import_data('clinvar_data.csv')
