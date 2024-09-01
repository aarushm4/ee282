from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = './uploads/' + file.filename
            file.save(filepath)
            variants = read_variants(filepath)
            annotated_variants = annotate_variants(variants)
            return render_template('results.html', data=annotated_variants)
    return render_template('index.html')


def read_variants(filepath):
    return pd.read_csv(filepath, sep='\t', comment='#', names=['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO'])


def annotate_variants(variants):
    conn = sqlite3.connect('variants.db')
    annotated_data = []
    for _, row in variants.iterrows():
        cursor = conn.cursor()
        query = "SELECT clin_sig FROM variant_annotations WHERE chrom=? AND pos=? AND ref=? AND alt=?"
        cursor.execute(
            query, (row['CHROM'], row['POS'], row['REF'], row['ALT']))
        result = cursor.fetchone()
        row['Annotation'] = result[0] if result else 'Unknown'
        annotated_data.append(row)
    conn.close()
    return annotated_data


@app.route('/visualize')
def visualize():
    conn = sqlite3.connect('variants.db')
    df = pd.read_sql_query(
        "SELECT clin_sig, COUNT(*) as count FROM variant_annotations GROUP BY clin_sig", conn)
    conn.close()
    plt.figure(figsize=(10, 6))
    plt.bar(df['clin_sig'], df['count'], color='skyblue')
    plt.xlabel('Clinical Significance')
    plt.ylabel('Number of Variants')
    plt.title('Distribution of Variants by Clinical Significance')
    plt.xticks(rotation=45)
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return '<img src="data:image/png;base64,{}">'.format(plot_url)


if __name__ == '__main__':
    app.run(debug=True)
