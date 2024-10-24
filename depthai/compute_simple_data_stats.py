import pickle
import argparse
from tqdm import tqdm
import numpy as np

def compute_stats(file_path):
    # Load the data
    with open(file_path, 'rb') as f:
        items = pickle.load(f)

    # Convert to DataFrame
    data = []
    for item in tqdm(items):
        token, points, info = item
        for point in points:
            x, y, z, intensity, time_lag = point
            data.append([x, y, z, intensity, time_lag])

    # Compute statistics (using numpy for simplicity)
    stats = {}
    data = np.array(data, dtype=np.float32)
    for i, col in enumerate(['x', 'y', 'z', 'intensity', 'time_lag']):
        stats[f'{col}_mean'] = np.mean(data[:, i])
        stats[f'{col}_std'] = np.std(data[:, i])
        stats[f'{col}_min'] = np.min(data[:, i])
        stats[f'{col}_max'] = np.max(data[:, i])

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute statistics from exported data.')
    parser.add_argument('file_path', type=str, help='Path to the pickled data file.')
    args = parser.parse_args()

    stats = compute_stats(args.file_path)
    for col in ['x', 'y', 'z', 'intensity', 'time_lag']:
        print(f'{col}:')
        for key, value in stats.items():
            if col in key:
                print(f'  {key}: {value}')
