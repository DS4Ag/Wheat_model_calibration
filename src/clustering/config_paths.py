"""
Input and output paths used across clustering analysis.
"""

# Input files
labels_dict = {
    '../data/clustering/subset_a.csv': 'A',
    '../data/clustering/subset_b.csv': 'B',
    '../data/clustering/subset_c.csv': 'C',
    '../data/clustering/subset_d.csv': 'D'
}
paths = list(labels_dict.keys())
labels = list(labels_dict.values())

# Output files
cluster_csv_output = '../output/clustering/clusters_k-means.csv'
final_figure_output = '../output/clustering/integrated_cluster_figure.svg'
feature_contributions_output = '../output/clustering/feature_contributions.csv'