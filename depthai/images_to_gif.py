import os
import imageio
import argparse
from tqdm import tqdm


def png_to_gif(input_dir, output_file_name='output.gif', fps=10):
    images = []
    image_dir = os.path.join(input_dir, 'images')
    for file_name in tqdm(sorted(os.listdir(image_dir)), desc='Building GIF'):
        if file_name.endswith('.png'):
            file_path = os.path.join(image_dir, file_name)
            images.append(imageio.imread(file_path))
    imageio.mimsave(output_file_name, images, fps=fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert exported PNG files into a GIF.')
    parser.add_argument('-i', '--input_dir', type=str, required=True, help="Path to the directory containing the images subdirectory "
                                                                           "(output_dir of the inference script).")
    parser.add_argument('-o', '--output', type=str, default='output.gif', help='Output file name. Default is output.gif.')
    parser.add_argument('-f', '--fps', type=int, default=10, help='Frames per second. Default is 10.')
    args = parser.parse_args()

    png_to_gif(args.input_dir, args.output, args.fps)
