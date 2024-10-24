import os
import argparse
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


def png_to_gif(input_dirs, output_file, fps):
    # Get list of image files from the first directory (assuming filenames are the same in all dirs)
    image_filenames = sorted(os.listdir(input_dirs[0]))

    # Ensure all directories have the same number of images
    for directory in input_dirs:
        if len(os.listdir(directory)) != len(image_filenames):
            raise ValueError(f"Directory {directory} does not contain the same number of images as {input_dirs[0]}")

    frames = []
    for filename in tqdm(image_filenames, desc="Converting images to GIF"):
        # Open images from each directory
        images = [Image.open(os.path.join(dir, filename)) for dir in input_dirs]

        # Concatenate images side by side with headers
        widths, heights = zip(*(i.size for i in images))
        total_width = sum(widths)
        max_height = max(heights)

        # Add space for the header text (e.g., 30 pixels high)
        header_height = 30
        new_img = Image.new('RGB', (total_width, max_height + header_height), (255, 255, 255))

        # Draw headers (input directory names) above each sub-image
        draw = ImageDraw.Draw(new_img)
        try:
            # Use a default font, but this may vary by system
            font = ImageFont.load_default(size=70)
        except:
            font = None

        x_offset = 0
        for img, directory in zip(images, input_dirs):
            # Paste the image
            new_img.paste(img, (x_offset, header_height))

            # Draw the directory name as the header
            header_text = os.path.dirname(directory)
            text_bbox = draw.textbbox((0, 0), header_text, font=font)  # Get the bounding box of the text
            text_width = text_bbox[2] - text_bbox[0]  # Calculate the width of the text
            text_height = text_bbox[3] - text_bbox[1]  # Calculate the height of the text
            text_x = x_offset + (img.width - text_width) // 2  # Center the text above the image
            draw.text((text_x, (header_height - text_height) // 2), header_text, font=font, fill=(0, 0, 0))

            x_offset += img.width

        frames.append(new_img)

    # Save the frames as a GIF
    frames[0].save(output_file, format='GIF', append_images=frames[1:], save_all=True, duration=1000 // fps, loop=0)
    print(f"GIF saved as {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert exported PNG files from multiple directories into a side-by-side comparison GIF.')
    parser.add_argument('-i', '--input_dirs', type=str, nargs='+', required=True, help="Paths to the directories containing the images subdirectories "
                                                                                      "(output_dir of the inference script).")
    parser.add_argument('-o', '--output', type=str, default='output.gif', help='Output file name. Default is output.gif.')
    parser.add_argument('-f', '--fps', type=int, default=10, help='Frames per second. Default is 10.')
    args = parser.parse_args()

    # Append 'images' to each directory path
    args.input_dirs = [os.path.join(dir, 'images') for dir in args.input_dirs]

    png_to_gif(args.input_dirs, args.output, args.fps)
