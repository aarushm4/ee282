import io
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify
import requests
import json
from matplotlib.patches import Rectangle
from io import BytesIO
import base64
import matplotlib
matplotlib.use('Agg')  # Use the non-GUI Agg backend
app = Flask(__name__)

# Fetch gene information from Ensembl


def get_gene_info(gene_name):
    server = "https://rest.ensembl.org"
    ext = f"/lookup/symbol/homo_sapiens/{gene_name}?expand=1"
    headers = {"Content-Type": "application/json"}
    response = requests.get(server + ext, headers=headers)
    if not response.ok:
        return None
    return response.json()

# Fetch gene sequence from Ensembl


def get_gene_sequence(gene_id):
    server = "https://rest.ensembl.org"
    ext = f"/sequence/id/{gene_id}?"
    headers = {"Content-Type": "application/json"}
    response = requests.get(server + ext, headers=headers)
    if not response.ok:
        return None
    return response.json().get('seq')

# Plot exon locations and return as a base64 image

def plot_exon_locations(gene_info, gene_sequence):
    if not gene_info or not gene_sequence:
        return None

    plt.figure(figsize=(10, 2))
    ax = plt.gca()
    ax.add_patch(Rectangle((0, 0.5), len(gene_sequence), 0.5,
                 edgecolor='black', facecolor='lightgray'))
    for transcript in gene_info.get('Transcript', []):
        for exon in transcript.get('Exon', []):
            exon_start = exon['start'] - gene_info['start']
            exon_length = exon['end'] - exon['start'] + 1
            ax.add_patch(Rectangle((exon_start, 0.5), exon_length,
                         0.5, edgecolor='black', facecolor='blue'))

    ax.set_xlim(0, len(gene_sequence))
    ax.set_ylim(0, 1)
    ax.set_xlabel('Nucleotide Position')
    plt.title(f"Exon Locations in Gene: {gene_info.get('display_name')}")

    # Convert plot to PNG image and encode it
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf-8')
    return plot_url


@app.route('/', methods=['GET', 'POST'])
def index():
    gene_info = None  # Initialize as None
    plot_url = None
    error = None

    if request.method == 'POST':
        gene_name = request.form['gene_name']
        gene_info = get_gene_info(gene_name)
        if gene_info:
            gene_sequence = get_gene_sequence(gene_info['id'])
            if gene_sequence:
                plot_url = plot_exon_locations(gene_info, gene_sequence)
            else:
                error = "Could not retrieve gene sequence."
        else:
            error = "Gene not found or API error."

    return render_template('index.html', gene_info=gene_info, plot_url=plot_url, error=error)



if __name__ == '__main__':
    app.run(debug=True)
