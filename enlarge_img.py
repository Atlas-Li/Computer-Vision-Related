from PIL import Image
import numpy as np
import argparse


def enlarge_img(path, target_size, background):
    img = Image.open(path)
    img_np = np.asarray(img)
    h, w, c= img_np.shape

    # th = 1080
    # tw = 1920
    th, tw = target_size
    if background == "black" or background == "b" or background == "B":
        out = np.zeros([th, tw, c], dtype=np.uint8)

    if background == "white" or background == "w" or background == "W":
        out = np.ones([th, tw, c], dtype=np.uint8)
        out = out * 255

    left = int(tw/2-w/2)
    right = int(tw/2+w/2)
    up = int(th/2-h/2)
    bottom = int(th/2+h/2)

    if c == 4:  # RGBD or RGBA
        out[up:bottom, left:right, 0] = img_np[:, :, 0]
        out[up:bottom, left:right, 1] = img_np[:, :, 1]
        out[up:bottom, left:right, 2] = img_np[:, :, 2]
        out[up:bottom, left:right, 3] = img_np[:, :, 3]
    if c == 3:  # RGB
        out[up:bottom, left:right, 0] = img_np[:, :, 0]
        out[up:bottom, left:right, 1] = img_np[:, :, 1]
        out[up:bottom, left:right, 2] = img_np[:, :, 2]
    if c == 1:  # gray image
        out[up:bottom, left:right, 0] = img_np[:, :, 0]
    
    return out


def get_argumets():
    """
        Parse arguments from command line
    """

    parser = argparse.ArgumentParser(description='Increase the size of an image without changing its center')

    parser.add_argument('--imagePath',
                        '-p', 
                        type=str, 
                        default='our_data/calibration/group10/kinect_ir_colorized/1.png',
                        help='file path of the image.')
    parser.add_argument('--targetHeight',
                        '-t',
                        type=int, 
                        default=1080,
                        help='Height of the output image.')
    parser.add_argument('--targetWidth',
                        '-w',
                        type=int, 
                        default=1920,
                        help='Width of the output image.')
    parser.add_argument('--background',
                        '-b',
                        type=str, 
                        default='white',
                        choices=['white', 'w', 'W', 'black', 'b', 'B'],
                        help='color of increased area.')

    return parser.parse_args()


if __name__ == "__main__":

    args = get_argumets()

    path = args.imagePath
    out_size = [args.targetHeight, args.targetWidth]
    bg = args.background
    out = enlarge_img(path, out_size, 'b')

    out = Image.fromarray(out)
    out.show()
