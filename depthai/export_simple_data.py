from onnx_infer import build_data_pipeline
import pickle
import numpy as np


def export_simple_data(args):
    print(f"Exporting simple data to {args.output}")

    dataset, data_loader = build_data_pipeline(args.data_dir)

    items = []
    for i, data_batch in enumerate(data_loader):
        token = data_batch['metadata'][0]['token']
        points = data_batch['points'][:, 1:].cpu().numpy().astype(np.float32)
        info = dataset._nusc_infos[i]
        items.append((token, points, info))

    # Save the data
    with open(args.output, 'wb') as f:
        pickle.dump(items, f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Export data for simple evaluation.')
    parser.add_argument('data_dir', type=str, help='Path to the dataset directory.')
    parser.add_argument('output', type=str, help='Path to the output file.')
    args = parser.parse_args()

    export_simple_data(args)
