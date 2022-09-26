import argparse
import torch
import os
import sys
import cv2
import numpy as np
import onnx
import onnxsim
from PIL import Image
from pathlib import Path

from unet import UNet
from utils.data_loading import BasicDataset
        

def file_size(path):
    # Return file/dir size (MB)
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / 1E6
    elif path.is_dir():
        return sum(f.stat().st_size for f in path.glob('**/*') if f.is_file()) / 1E6
    else:
        return 0.0

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--output', '-o', default='MODEL.onnx', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--n_classes', type=int, default=2, help='Number of output channals')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    
    return parser.parse_args()



def main(args):
    current_dir = os.path.dirname(__file__)
    if current_dir == "":
        current_dir = "./"
    # model_path = current_dir+'/models/mlsd_tiny_512_fp32.pth'
    # model = MobileV2_MLSD_Tiny().cuda().eval()
    
    model_path = args.model
    model = UNet(n_channels=3, n_classes=args.n_classes)

    device = 'cpu'    # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device=device)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)

    img = Image.new('RGB', (320, 320), (255, 255, 255))
    img = torch.from_numpy(BasicDataset.preprocess(img, args.scale, is_mask=False))
    img = img.unsqueeze(0)
    batch_image = img.to(device=device, dtype=torch.float32)
    
    outputs = model(batch_image)
    
    onnx_path = args.output
    opset = 12
    train = False
    dynamic = False
    verbose = False
    
    torch.onnx.export(model, batch_image, onnx_path, verbose=verbose, opset_version=opset,
                          training=torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,
                          do_constant_folding=not train,
                          input_names=['images'],
                          output_names=['output'],
                          dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # shape(1,3,640,640)
                                        'output': {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)
                                        } if dynamic else None)
    
    model_onnx = onnx.load(onnx_path)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model
    model_onnx, check = onnxsim.simplify(
                    model_onnx,
                    dynamic_input_shape=dynamic,
                    input_shapes={'images': list(batch_image.shape)} if dynamic else None)
    assert check, 'assert check failed'
    onnx.save(model_onnx, onnx_path)
    
    return    # no TensorRT
    
    prefix = 'TensorRT:'
    workspace = 12
    try:
        import tensorrt as trt
        opset = (12, 13)[trt.__version__[0] == '8']  # test on TensorRT 7.x and 8.x
        assert os.path.exists(onnx_path), f'failed to export ONNX file: {onnx}'

        print(f'\n{prefix} starting export with TensorRT {trt.__version__}...')
        f = 'test.engine'  # TensorRT engine file
        logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            logger.min_severity = trt.Logger.Severity.VERBOSE

        builder = trt.Builder(logger)
        config = builder.create_builder_config()
        config.max_workspace_size = workspace * 1 << 30

        flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        network = builder.create_network(flag)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(onnx_path)):
            raise RuntimeError(f'failed to load ONNX file: {onnx_path}')

        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]
        print(f'{prefix} Network Description:')
        for inp in inputs:
            print(f'{prefix}\tinput "{inp.name}" with shape {inp.shape} and dtype {inp.dtype}')
        for out in outputs:
            print(f'{prefix}\toutput "{out.name}" with shape {out.shape} and dtype {out.dtype}')
            
        half = True
        
        half &= builder.platform_has_fast_fp16
        print(f'{prefix} building FP{16 if half else 32} engine in {f}')
        if half:
            config.set_flag(trt.BuilderFlag.FP16)
        with builder.build_engine(network, config) as engine, open(f, 'wb') as t:
            t.write(engine.serialize())
        print(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')

    except Exception as e:
        print(f'\n{prefix} export failure: {e}')
if __name__ == '__main__':
    args = get_args()
    main(args)
