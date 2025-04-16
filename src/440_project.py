import sys
from pathlib import Path
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np

#get Python executable and version for debugging purposes
print("Python executable:", sys.executable)
print("Python version:", sys.version)

#define base directory and subdirectories
BASE_DIR = Path.home() / 'Downloads' / 'project440'
DATA_DIR = BASE_DIR / 'data'
RESULTS_DIR = BASE_DIR / 'results'

# Ensure the results directory exists
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """
    Load data from the specified files in the 'data' folder.
    Returns:
        Expression data, sample data, and drug data.
    """
    #load expression data from .tsv file
    expression_data_path = DATA_DIR / 'GSE217421_raw_counts_GRCh38.p13_NCBI.tsv'
    expression_data = pd.read_csv(expression_data_path, sep='\t', index_col=0).T

    # Load sample reference data from .xlsx file
    sample_data_path = DATA_DIR / 'Sample_reference.xlsx'
    sample_data = pd.read_excel(sample_data_path)

    # Load drug reference data from .txt file
    drug_data_path = DATA_DIR / 'Drug_reference.txt'
    drug_data = pd.read_csv(drug_data_path, sep='\t')

    return expression_data, sample_data, drug_data

def preprocess_data(expression, sample_data, drug_data):
    """
    Preprocess the data to create a normalized AnnData object w/ drug metadata
    """
    samples = pd.DataFrame(expression.index, columns=['Sample Name'])
    
    #clean column names and values by stripping whitespace
    sample_data.columns = sample_data.columns.str.strip()
    drug_data.columns = drug_data.columns.str.strip()
    
    # Standardize drug identifiers for merging
    drug_data['Drug abbreviation'] = drug_data['Drug abbreviation'].str.upper().str.strip()
    sample_data['Drug'] = sample_data['Drug'].str.upper().str.strip()

    #merge the samples and the sample metadata: Sample metadata
    sample_merged = pd.merge(
        samples,
        sample_data,
        on='Sample Name',
        how='left'
    )
    print(f"After sample merge: {len(sample_merged)} samples") #for debugging 

    #second merge for drug information on drug metadata
    sample_merged = pd.merge(
        sample_merged,
        drug_data,
        left_on='Drug',
        right_on='Drug abbreviation',
        how='left',  # Keep samples even if drug data missing
        suffixes=('', '_drug')
    )
    print(f"After drug merge: {len(sample_merged)} samples")
    
    #check to ensure merge success
    print("Merged columns:", sample_merged.columns.tolist())
    print("Null values in Drug class:", sample_merged['Drug class'].isnull().sum())

    #store final sample metadata
    final_sample_metadata = sample_merged.set_index('Sample Name')
    
    # Subset and align expression data
    relevant_samples = final_sample_metadata.index.tolist()
    expression_subset = expression[expression.index.isin(relevant_samples)]
    final_sample_metadata = final_sample_metadata.loc[expression_subset.index]

    # Create AnnData object
    adata = sc.AnnData(
        X=expression_subset.values,
        obs=final_sample_metadata,
        var=pd.DataFrame(index=expression_subset.columns)
    )

    #add data layers
    adata.layers['raw_counts'] = expression_subset.values
    adata.layers['normalized_counts'] = expression_subset.values

    #validate for debugging purposes 
    if 'Drug class' not in adata.obs.columns:
        print("drug class missing after all merges")
    else:
        print("Successfully merged drug classes:", adata.obs['Drug class'].unique())

    return adata


def save_csv_results(adata):
    """
    Save PCA and UMAP results to CSV files in the 'results' folder.
    Args:
        adata (sc.AnnData): AnnData object containing PCA and UMAP results.
    """
    pca_path = RESULTS_DIR / 'pca_results.csv'
    umap_path = RESULTS_DIR / 'umap_results.csv'

    pd.DataFrame(adata.obsm['X_pca']).to_csv(pca_path, index=False)
    pd.DataFrame(adata.obsm['X_umap']).to_csv(umap_path, index=False)
    
    print(f"PCA results saved to {pca_path}")
    print(f"UMAP results saved to {umap_path}")

def save_plots(adata):
    """
    Save PCA and UMAP plots to image files in the 'results' folder.
    arguments:
        adata (sc.AnnData): AnnData object containing PCA and UMAP results.
    """
    # Save PCA plot
    pca_plot_path = RESULTS_DIR / 'pca_plot.png'
    sc.pl.pca(
        adata,
        components=[('1,2'), ('2,3')],
        color=['Drug name', 'Drug class', 'Cell line'],
        ncols=2,
        wspace=0.3,
        size=100,
        show=False
    )
    plt.savefig(pca_plot_path)
    print(f"PCA plot saved to {pca_plot_path}")

    # Save UMAP plot
    umap_plot_path = RESULTS_DIR / 'umap_plot.png'
    sc.pl.umap(
        adata,
        color=['Drug class', 'Cell line', 'Drug name', 'leiden'],
        size=100,
        ncols=2,
        wspace=0.3,
        show=False
    )
    plt.savefig(umap_plot_path)
    print(f"UMAP plot saved to {umap_plot_path}")
# Main execution block:
if __name__ == "__main__":
    # Load data from 'data' folder
    expression, sample_data, drug_data = load_data()
    print(sample_data)
    
    # Preprocess data into an AnnData object
    analysis_object = preprocess_data(expression, sample_data, drug_data)
    
    #print all available drug classes for debugging
    print("All drug classes found:", analysis_object.obs['Drug class'].unique())
    
    # Try more flexible filtering with case-insensitive substring matching
    if 'Drug class' in analysis_object.obs.columns:
        # First try exact match
        anthracycline_mask = (
            (analysis_object.obs['Drug class'] == 'Anthracycline') |
            (analysis_object.obs['Drug class'] == 'Control')
        )
        print(anthracycline_mask)
        # If exact match returns no results, try case-insensitive partial match
        if anthracycline_mask.sum() == 0:
            print("No exact matches found. Trying case-insensitive matching...")
            anthracycline_mask = (
                analysis_object.obs['Drug class'].str.lower().str.contains('anthracycline') | 
                analysis_object.obs['Drug class'].str.lower().str.contains('control')
            )
        
        # If still no results, use all data and warn user
        if anthracycline_mask.sum() == 0:
            print("WARNING: No samples match Anthracycline/Control criteria. Using all data.")
            analysis_Anthracyclines = analysis_object.copy()
        else:
            print(f"Found {anthracycline_mask.sum()} samples matching criteria.")
            analysis_Anthracyclines = analysis_object[anthracycline_mask].copy()
    else:
        print("ERROR: No 'Drug class' column found!")
        analysis_Anthracyclines = analysis_object.copy()
    
    # Only proceed with analysis if we have samples
    if len(analysis_Anthracyclines.obs) > 0:
        print(f"Performing analysis on {len(analysis_Anthracyclines.obs)} samples")
        
        # Scale the data
        sc.pp.scale(analysis_Anthracyclines)
        
        # PCA with explicit number of components
        sc.tl.pca(analysis_Anthracyclines, n_comps=min(50, len(analysis_Anthracyclines.obs)-1))
        
        # UMAP projection
        sc.pp.neighbors(analysis_Anthracyclines, use_rep='X_pca', metric='cosine')
        sc.tl.umap(analysis_Anthracyclines)
        sc.tl.leiden(analysis_Anthracyclines, resolution=0.3)
        
        # Save results
        save_csv_results(analysis_Anthracyclines)
        save_plots(analysis_Anthracyclines)
        
        print(f"Analysis complete. Results saved to {RESULTS_DIR}")
    else:
        print("ERROR: Analysis cannot proceed with 0 samples!")
