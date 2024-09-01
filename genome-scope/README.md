# GenomeScope: Genetic Variant Impact Analysis Tool

GenomeScope is a web-based tool designed to provide annotations for genetic variants and predict their potential impacts on health. It combines advanced data analysis and visualization to help users, from researchers to educators, better understand genetic data.

## Features

- **Variant Annotation:** Utilizes known databases like ClinVar to annotate genetic variants with clinical significance.
- **Impact Prediction:** Predicts health impacts of novel variants using a machine learning model.
- **Data Visualization:** Offers dynamic visualizations to explore the distribution and significance of genetic variants.
- **User-Friendly Interface:** Simple and interactive web interface for easy upload and analysis of genetic data.

## Installation

Before running GenomeScope, you will need to set up your environment. Here's how you can get started:

1. **Clone the Repository**

   ```bash
   git clone https://github.com/aarushm4/ee282.git
   
   cd genomescope
2. **Set up a Python Virtual Environment (recommended)**
   
   ```bash
   python -m venv venv
   
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
4. **Install Required Packages**
   ```bash
   pip install -r requirements.txt
   
5. **Initialize the Database Run the following script to create your SQLite database and tables:**
   
   ```bash
   python database_setup.py
   
6. **Import Data Make sure to have a .csv file with variant data ready (formatted according to the ClinVar database schema):**
   
   ```bash
   python data_import.py --file path_to_your_clinvar_data.csv

## Usage

Once the installation is complete, you can start the server:
```bash
python app.py
```

**Navigate to <http://127.0.0.1:5000/> in your web browser to access GenomeScope.**

## Uploading Data

On the homepage, use the file input to select your VCF file and click 'Upload'.
The application will process the file and display annotated results along with visualizations.

## Visualizing Data

Access the /visualize endpoint to view a histogram of variant annotations by clinical significance.

## Contributions

Contributions are welcome! Please read our contributing guidelines to learn how you can propose bug fixes, improvements, or new features.

## License

GenomeScope is released under the MIT License. See the LICENSE file for more details.

## Contact

For support or partnership inquiries, please contact <aarushi2.mehta@gmail.com>.

Enjoy analyzing genetic variants with GenomeScope!
