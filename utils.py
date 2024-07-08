import pandas as pd
import numpy as np

from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from skimage.io import imread
from skimage.measure import regionprops
from skimage.color import label2rgb
from skimage.morphology import erosion
from skimage.morphology import octagon
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt


def get_new_mask(mask, df, cell_types):
    ccs = regionprops(mask)
    val2cc = {}
    for cc in ccs:
        c_ = list(cc.centroid)
        val = mask[int(c_[0]), int(c_[1])]
        val2cc[val] = cc
    new_mask = np.zeros(mask.shape, dtype=mask.dtype)
    for _, row in df.iterrows():
        tp = row['claster']
        try: 
            tp_idx = cell_types.index(tp) + 1 
        except ValueError: 
            print(f"Error: '{tp}' is not in the cell_types list. Trying with stripped space...") 
            tp = tp.replace(" ", "") # 去除空格重新尝试 
            try: 
                tp_idx = cell_types.index(tp) + 1 
            except ValueError: 
                print(f"Error: '{tp}' is still not in the cell_types list. Skipping this entry.") 
                continue
        # tp_idx = cell_types.index(tp) + 1
        c_ = [int(float(i)) for i in row['cell'].strip("()").split(",")]
        val = mask[c_[0], c_[1]]
        if val > 0:
            cc = val2cc[val]
            new_mask[cc.coords[:, 0], cc.coords[:, 1]] = tp_idx
        else:
            print(val)
    return new_mask

def hex_to_array(h):
    h = h.lstrip("#")
    l = [int(h[i:i+2], 16)/255 for i in (0, 2, 4)]
    return np.array(l)

def get_border(mask):
    border = mask - erosion(mask, octagon(3, 1))
    #border = mask - erosion(mask)
    border = border > 0
    return border

def draw_type_mask(ax, mask, colors, labels, output_path, bg_color="#000000", back=None, border=None):
    colors_ = [hex_to_array(i) for i in colors]
    bg_color = hex_to_array(bg_color)
    rgb_img = label2rgb(mask, colors=colors_, bg_color=bg_color)
    if border is not None:
        rgb_img[border] = bg_color
    if back is not None:
        back_ratio = 0.4
        back_ = (back ^ back.min()) / (back.max() ^ back.min())
        back_ = np.stack([back_, back_, back_], axis=2)
        img = back_ * 0.8 + rgb_img * 0.8
        ax.imshow(img)
        ax.imshow(back, cmap="gray", alpha=0.5)
        ax.imshow(rgb_img, alpha=0.75)
    else:
        ax.imshow(rgb_img)
    ax.grid(False)
    # Add legend
    patches = [plt.Rectangle((0,0),1,1,fc=color) for color in colors]
    ax.legend(patches, labels, loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)
    ax.axis('off')  # Do not display x and y axes
    ax.grid(False)
    plt.savefig(output_path, format='pdf')
    plt.show()

def find_density_centers(file_path, gene_name, top_n=1, n_clusters=3, min_distance=120):
    """
    Finds the density centers for a given gene in the dataset within different regions.

    Parameters:
    - file_path: str, path to the CSV file
    - gene_name: str, the gene to filter the data by
    - top_n: int, number of top density centers to return per region
    - n_clusters: int, number of clusters/regions to divide the data into
    - min_distance: float, minimum distance between density centers

    Returns:
    - filtered_coords_list: list of numpy.ndarray, coordinates of the filtered density centers for each region
    """
    data = pd.read_csv(file_path)
    gene_data = data[data['gene'] == gene_name]
    
    if gene_data.shape[0] < 2:
        print(f"The dataset for gene {gene_name} has insufficient data points for KDE.")
        return None
    else:
        x = gene_data['dim_2'].values
        y = gene_data['dim_1'].values
        xy = np.vstack([x, y]).T

        # Perform K-means clustering to divide the data into regions
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(xy)
        labels = kmeans.labels_

        top_coords_list = []

        plt.figure(figsize=(10, 6))

        for i in range(n_clusters):
            cluster_data = xy[labels == i]
            if cluster_data.shape[0] < 2:
                print(f"Cluster {i} has insufficient data points for KDE.")
                continue
            
            # Perform KDE for the current cluster
            kde = gaussian_kde(cluster_data.T, bw_method='scott')
            xmin, xmax = cluster_data[:, 0].min() - 1, cluster_data[:, 0].max() + 1
            ymin, ymax = cluster_data[:, 1].min() - 1, cluster_data[:, 1].max() + 1
            xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            density = kde(positions).reshape(xx.shape)
            density_flat = density.flatten()
            sorted_indices = np.argsort(density_flat)[::-1]  # Sort in descending order
            top_indices = sorted_indices[:top_n * 10]  # Take more top density points to filter later
            top_coords = np.vstack([positions[0][top_indices], positions[1][top_indices]]).T
            
            # Filter top density centers by minimum distance
            filtered_coords = []
            for coord in top_coords:
                if not filtered_coords or all(cdist([coord], filtered_coords) >= min_distance):
                    filtered_coords.append(coord)
                if len(filtered_coords) >= top_n:
                    break
            
            filtered_coords = np.array(filtered_coords)
            top_coords_list.append(filtered_coords)

            # Add KDE and top density centers for the current cluster to the global plot
            plt.imshow(np.rot90(density), cmap='Blues', extent=[xmin, xmax, ymin, ymax], alpha=0.5)
            plt.scatter(filtered_coords[:, 0], filtered_coords[:, 1], s=50, label=f'Top Density Centers (Cluster {i})')

        plt.title(f'Kernel Density Estimation for {gene_name} Gene (All Clusters)')
        plt.xlabel('dim_2 (x-axis)')
        plt.ylabel('dim_1 (y-axis)')
        plt.legend()
        plt.show()

        return top_coords_list

# def find_density_centers(file_path, gene_name, top_n=1, n_clusters=3):
#     """
#     Finds the density centers for a given gene in the dataset within different regions.

#     Parameters:
#     - file_path: str, path to the CSV file
#     - gene_name: str, the gene to filter the data by
#     - top_n: int, number of top density centers to return per region
#     - n_clusters: int, number of clusters/regions to divide the data into

#     Returns:
#     - top_coords_list: list of numpy.ndarray, coordinates of the top density centers for each region
#     """
#     data = pd.read_csv(file_path)
#     gene_data = data[data['gene'] == gene_name]
    
#     if gene_data.shape[0] < 2:
#         print(f"The dataset for gene {gene_name} has insufficient data points for KDE.")
#         return None
#     else:
#         x = gene_data['dim_2'].values
#         y = gene_data['dim_1'].values
#         xy = np.vstack([x, y]).T

#         # Perform K-means clustering to divide the data into regions
#         kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(xy)
#         labels = kmeans.labels_

#         top_coords_list = []

#         plt.figure(figsize=(10, 6))

#         for i in range(n_clusters):
#             cluster_data = xy[labels == i]
#             if cluster_data.shape[0] < 2:
#                 print(f"Cluster {i} has insufficient data points for KDE.")
#                 continue
            
#             # Perform KDE for the current cluster
#             kde = gaussian_kde(cluster_data.T, bw_method='scott')
#             xmin, xmax = cluster_data[:, 0].min() - 1, cluster_data[:, 0].max() + 1
#             ymin, ymax = cluster_data[:, 1].min() - 1, cluster_data[:, 1].max() + 1
#             xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
#             positions = np.vstack([xx.ravel(), yy.ravel()])
#             density = kde(positions).reshape(xx.shape)
#             density_flat = density.flatten()
#             sorted_indices = np.argsort(density_flat)[::-1]  # Sort in descending order
#             top_indices = sorted_indices[:top_n]  # Top N density points
#             top_coords = np.vstack([positions[0][top_indices], positions[1][top_indices]]).T
#             top_coords_list.append(top_coords)

#             # Add KDE and top density centers for the current cluster to the global plot
#             plt.imshow(np.rot90(density), cmap='Blues', extent=[xmin, xmax, ymin, ymax], alpha=0.5)
#             plt.scatter(top_coords[:, 0], top_coords[:, 1], s=50, label=f'Top Density Centers (Cluster {i})')

#         plt.title(f'Kernel Density Estimation for {gene_name} Gene (All Clusters)')
#         plt.xlabel('dim_2 (x-axis)')
#         plt.ylabel('dim_1 (y-axis)')
#         plt.legend()
#         plt.show()

#         return top_coords_list
    
def calculate_surrounding_cell_types(df, center_cells, radii=[60, 120, 180]):
    """
    Calculate the surrounding cell types within given radii for each center cell and
    return the difference in counts between each pair of consecutive radii.

    Parameters:
    - df: DataFrame, contains the cell data with columns 'cell' and 'claster'
    - center_cells: numpy.ndarray, array of shape (n, 2) where each row is [y, x] coordinates of center cells
    - radii: list of int, the radii within which to search for surrounding cells

    Returns:
    - results: list of dictionaries, each containing the center cell info and surrounding cell type counts for each radius
    """
    results = []
    df[['y', 'x']] = df['cell'].str.strip('()').str.split(',', expand=True).astype(float)

    for center in center_cells:
        center_x, center_y = center
        df['distance'] = np.sqrt((df['x'] - center_x)**2 + (df['y'] - center_y)**2)
        
        prev_counts = {}
        cell_type_counts = {radius: {} for radius in radii}
        
        for i, radius in enumerate(radii):
            surrounding_cells = df[(df['distance'] <= radius) & (df['distance'] > 0)]
            counts = surrounding_cells['claster'].value_counts().to_dict()
            
            if i == 0:
                cell_type_counts[radius] = counts
            else:
                diff_counts = {}
                for cell_type, count in counts.items():
                    prev_count = prev_counts.get(cell_type, 0)
                    diff_counts[cell_type] = count - prev_count
                cell_type_counts[radius] = diff_counts
            
            prev_counts = counts
        
        results.append({
            'center_y': center_y,
            'center_x': center_x,
            'surrounding_cell_types': cell_type_counts
        })
    
    return results

def analyze_samples(samples, input_dir, cell_types, colors, top_density_centers, radii=[50, 100, 150], dpi=96):
    all_results = []

    for sample in samples:
        print(sample)    
        cell_df = pd.read_csv(f"{input_dir}/{sample}.csv", sep=",")
        cell_df = cell_df.drop(columns=['niche'])
        file = sample.split('-')[0]
        id = sample.split('-')[1]
        mask = imread(f"{input_dir}/{file}_{id}_mask.tif")
        print(mask.shape)
        new_mask = get_new_mask(mask, cell_df, cell_types)

        fig, ax = plt.subplots(figsize=(20, 10), dpi=dpi)
        for centers in top_density_centers:
            print(centers)
            y = centers[:, 0]
            x = centers[:, 1]
            spots = 5
            spots_center = (spots / (dpi / 72))**2
            for radius in radii:
                area_in_points = (radius / (dpi / 72))**2
                ax.scatter(y, x, s=area_in_points, marker='o', edgecolors='white', facecolors='none', linewidths=4, linestyle='--')
            ax.scatter(y, x, s=spots_center, marker='o', edgecolors='white', facecolors='none', linewidths=3)

        output_path = f'./{sample}.pdf'
        draw_type_mask(ax, new_mask, colors=colors, labels=cell_types, output_path=output_path)

        for centers in top_density_centers:
            results = calculate_surrounding_cell_types(cell_df, centers, radii)
            all_results.extend(results)
    
    csv_data = []
    for result in all_results:
        for radius, counts in result['surrounding_cell_types'].items():
            row = {
                'center_y': result['center_y'],
                'center_x': result['center_x'],
                'radius': radius
            }
            row.update(counts)
            csv_data.append(row)

    df_csv = pd.DataFrame(csv_data)
    df_csv.to_csv(f'./{sample}_surrounding_cell_types.csv', index=False)