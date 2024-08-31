import requests
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def get_gene_info(gene_name):
    server = "https://rest.ensembl.org"
    ext = f"/lookup/symbol/homo_sapiens/{gene_name}?expand=1"
    headers = {"Content-Type": "application/json"}
    response = requests.get(server + ext, headers=headers)
    if not response.ok:
        print(f"Error: {response.status_code}")
        return None
    return response.json()


def get_gene_sequence(gene_id):
    server = "https://rest.ensembl.org"
    ext = f"/sequence/id/{gene_id}?"
    headers = {"Content-Type": "application/json"}
    response = requests.get(server + ext, headers=headers)
    if not response.ok:
        print(f"Error: {response.status_code}")
        return None
    return response.json().get('seq')


def display_gene_info(gene_info):
    if gene_info:
        print(f"Gene Name: {gene_info.get('display_name')}")
        print(f"Gene ID: {gene_info.get('id')}")
        print(f"Description: {gene_info.get('description')}")
        print(f"Location: {gene_info.get('seq_region_name')}:{
              gene_info.get('start')}-{gene_info.get('end')}")
        print(f"Strand: {'+' if gene_info.get('strand') == 1 else '-'}")
        print("Transcript IDs:")
        for transcript in gene_info.get('Transcript', []):
            print(f"  - {transcript['id']}")
    else:
        print("No information available for the given gene.")


def plot_exon_locations(gene_info, gene_sequence):
    if not gene_info or not gene_sequence:
        return

    gene_length = len(gene_sequence)
    fig, ax = plt.subplots(figsize=(10, 2))

    # Plotting the gene sequence as a long rectangle
    ax.add_patch(Rectangle((0, 0.5), gene_length, 0.5,
                 edgecolor='black', facecolor='lightgray'))

    # Marking exons on the gene sequence
    for transcript in gene_info.get('Transcript', []):
        for exon in transcript.get('Exon', []):
            exon_start = exon['start'] - gene_info['start']
            exon_length = exon['end'] - exon['start'] + 1
            ax.add_patch(Rectangle((exon_start, 0.5), exon_length,
                         0.5, edgecolor='black', facecolor='blue'))

    ax.set_xlim(0, gene_length)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Nucleotide Position')
    ax.set_yticks([])
    ax.set_title(f"Exon Locations in Gene: {gene_info.get('display_name')}")

    plt.show()


def main():
    gene_name = input("Enter the gene name or ID: ").strip()
    gene_info = get_gene_info(gene_name)
    display_gene_info(gene_info)

    if gene_info:
        gene_sequence = get_gene_sequence(gene_info.get('id'))
        if gene_sequence:
            print(f"\nGene Sequence (first 100 bases): {
                  gene_sequence[:100]}... (total length: {len(gene_sequence)} bases)")
            plot_exon_locations(gene_info, gene_sequence)
        else:
            print("Could not retrieve gene sequence.")
    else:
        print("Could not retrieve gene information.")


if __name__ == "__main__":
    main()
