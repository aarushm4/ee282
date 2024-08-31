import requests
import json
import matplotlib.pyplot as plt
from Bio.Align import PairwiseAligner
from ete3 import Tree, TreeStyle, NodeStyle
import sqlite3
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import flask
from flask import request, render_template

app = flask.Flask(__name__)

# Database connection


def connect_db():
    return sqlite3.connect('gene_analysis.db')


def create_tables():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS gene_analysis (
            gene_id TEXT PRIMARY KEY,
            gene_name TEXT,
            description TEXT,
            location TEXT,
            sequence TEXT,
            variants TEXT,
            alignment_results TEXT
        )
    ''')
    conn.commit()
    conn.close()


create_tables()

# Gene information retrieval


def get_gene_info(gene_name):
    server = "https://rest.ensembl.org"
    ext = f"/lookup/symbol/homo_sapiens/{gene_name}?expand=1"
    headers = {"Content-Type": "application/json"}
    response = requests.get(server + ext, headers=headers)
    if not response.ok:
        print(f"Error: {response.status_code}")
        return None
    print(response.json().get('text'))
    return response.json()


def get_gene_sequence(gene_id, species="homo_sapiens"):
    server = "https://rest.ensembl.org"
    ext = f"/sequence/id/{gene_id}?species={species}"
    headers = {"Content-Type": "application/json"}
    response = requests.get(server + ext, headers=headers)
    if not response.ok:
        print(f"Error: {response.status_code}")
        return None
    return response.json().get('seq')


def get_variants_in_gene(gene_id):
    server = "https://rest.ensembl.org"
    ext = f"/overlap/id/{gene_id}?feature=variation"
    headers = {"Content-Type": "application/json"}
    response = requests.get(server + ext, headers=headers)
    if not response.ok:
        print(f"Error: {response.status_code}")
        return None
    return response.json()

# Advanced Variant Filtering


def filter_variants(variants, criteria=None):
    if criteria:
        filtered = []
        for var in variants:
            match = True
            for key, value in criteria.items():
                if var.get(key) != value:
                    match = False
                    break
            if match:
                filtered.append(var)
        return filtered
    return variants

# Sequence Alignment (Global and Local)


def align_sequences(seq1, seq2, alignment_type="global"):
    aligner = PairwiseAligner()

    if alignment_type == "global":
        aligner.mode = "global"
    elif alignment_type == "local":
        aligner.mode = "local"
    else:
        raise ValueError("Invalid alignment type. Use 'global' or 'local'.")

    alignments = aligner.align(seq1, seq2)

    return alignments

# Phylogenetic Tree Construction


def construct_phylogenetic_tree(gene_sequences):
    tree = Tree()  # Create a tree with no initial structure
    for species, sequence in gene_sequences.items():
        # Add a child node for each species with dummy distance
        tree.add_child(name=species, dist=len(sequence))

    # Create a TreeStyle object
    ts = TreeStyle()
    ts.show_leaf_name = True  # Display leaf names
    ts.show_branch_support = True  # Optionally show branch support values

    # Set styles for nodes
    nstyle = NodeStyle()
    nstyle["size"] = 10
    for node in tree.traverse():
        node.set_style(nstyle)

    # Render the tree
    tree.show(tree_style=ts)

# Machine Learning for Variant Impact Prediction


def predict_variant_impact(variant_data):
    # Example training data (this should be expanded with real datasets)
    # Features (e.g., position, consequence type, frequency)
    X = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    y = np.array([0, 1, 0])  # Labels (e.g., 0 = benign, 1 = pathogenic)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy * 100:.2f}%")
    return model.predict(variant_data)

# Web Interface


@app.route('/plot')
def plot():
    # Example: Generate a bar chart for variant frequencies
    filtered_variants = request.args.get('filtered_variants', '[]')
    filtered_variants = json.loads(filtered_variants)

    # Extract example data from filtered variants
    variant_names = [var['name'] for var in filtered_variants]
    frequencies = [var.get('frequency', 0) for var in filtered_variants]

    fig, ax = plt.subplots()
    ax.bar(variant_names, frequencies)
    ax.set_xlabel('Variants')
    ax.set_ylabel('Frequency')
    ax.set_title('Variant Frequencies')

    # Save the plot to a BytesIO object and encode it as a base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    return img_base64


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        gene_name = request.form['gene_name']
        species_list = request.form['species_list'].split(',')
        alignment_type = request.form['alignment_type']
        criteria = {
            'consequence_type': request.form['consequence_type'],
            'clinical_significance': request.form['clinical_significance'].split(',')
        }

        gene_info = get_gene_info(gene_name)
        gene_sequence = get_gene_sequence(
            gene_info['id']) if gene_info else None
        variants = get_variants_in_gene(gene_info['id']) if gene_info else None

        # Filter variants
        filtered_variants = filter_variants(variants, criteria)

        # Sequence Alignment
        gene_sequences = {species: get_gene_sequence(
            gene_info['id'], species) for species in species_list}
        alignment_results = []

        # Phylogenetic Tree
        construct_phylogenetic_tree(gene_sequences)

        # Predict Variant Impact (dummy data)
        # Replace with actual variant features
        variant_data = np.array([[1, 1, 1]])
        predicted_impact = predict_variant_impact(variant_data)

        return render_template('result.html', gene_info=gene_info, filtered_variants=filtered_variants, alignment_results=alignment_results)

    return render_template('home.html')

# Database integration for storing analysis


def save_analysis_to_db(gene_info, gene_sequence, variants, alignment_results):
    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute('''
        INSERT OR REPLACE INTO gene_analysis (gene_id, gene_name, description, location, sequence, variants, alignment_results)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        gene_info['id'],
        gene_info['display_name'],
        gene_info['description'],
        f"{gene_info['seq_region_name']}:{
            gene_info['start']}-{gene_info['end']}",
        gene_sequence,
        json.dumps(variants),
        json.dumps(alignment_results)
    ))

    conn.commit()
    conn.close()


# Running the Flask application
if __name__ == "__main__":
    app.run(debug=True)
