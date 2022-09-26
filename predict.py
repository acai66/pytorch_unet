import argparse
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
# from unet import UNet_3Plus as UNet
from unet import UNet as UNet
from utils.utils import plot_img_and_mask

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    resized_img = full_img.resize((320, 320))
    img = torch.from_numpy(BasicDataset.preprocess(resized_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.ToTensor()
        ])

        #full_mask = tf(probs[1].cpu()).squeeze()
        #return (full_mask > out_threshold).numpy()
        full_mask = probs.cpu()

    if net.n_classes == 1:
        out = (full_mask > out_threshold).numpy()
        mask_img = mask_to_image(tf(out))
        return mask_img
    else:
        out = F.one_hot(full_mask.argmax(dim=0), net.n_classes)
        mask_img = mask_to_image(out.permute(2, 0, 1).numpy())
        mask_img = mask_img.resize((full_img.size[0], full_img.size[1]))
        return mask_img



def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--n_classes', type=int, default=2, help='Number of output channals')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', default='demo', metavar='OUTPUT',
                        help='Specify the directory in which the outputs is stored')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')

    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        split = os.path.splitext(fn)
        return f'{split[0]}_OUT{split[1]}'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))


if __name__ == '__main__':
    allowed_formats = ('.jpg', '.png', '.JPG', '.PNG')
    args = get_args()
    in_files = args.input
    output_path = args.output
    if os.path.exists(output_path):
        if os.path.isfile(output_path):
            logging.error(f'--output {output_path} is not a directory!')
            sys.exit(3)
    else:
        os.mkdir(output_path)

    net = UNet(n_channels=3, n_classes=args.n_classes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))
    
    img = Image.new('RGB', (256, 256), (0, 0, 0))
    img = img.convert("RGB")
    mask = predict_img(net=net,
                       full_img=img,
                       scale_factor=args.scale,
                       out_threshold=args.mask_threshold,
                       device=device)

    logging.info('Model loaded!')

    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        if os.path.isdir(filename):
            imgs_path = [os.path.join(filename, i) for i in os.listdir(filename) if i[-4:] in allowed_formats]
        else:
            imgs_path = [filename]
        for img_path in imgs_path:
            img = Image.open(img_path)
            img = img.convert("RGB")

            start_time = time.time()
            mask = predict_img(net=net,
                               full_img=img,
                               scale_factor=args.scale,
                               out_threshold=args.mask_threshold,
                               device=device)
            end_time = time.time()
            time_use = end_time - start_time
            logging.info('Predict time: %.4f s.' % (time_use))

            if not args.no_save:
                out_filename = os.path.join(output_path, os.path.splitext(os.path.split(img_path)[1])[0] + '_OUT.png')
                result = mask    # mask_to_image(mask)
                result.save(out_filename)
                logging.info(f'Mask saved to {out_filename}')

            if args.viz:
                logging.info(f'Visualizing results for image {img_path}, close to continue...')
                plot_img_and_mask(img, mask)
