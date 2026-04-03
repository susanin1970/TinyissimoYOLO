import argparse

from ultralytics import YOLO
from ultralytics.utils.offline_tiling import Tiler


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a TinyissimoYOLO model with offline tiling (DSORT-MCU).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--tiling-config', default='tiling_config.yaml',
        help='Path to the tiling configuration YAML file',
    )
    parser.add_argument(
        '--model', default='tinyissimo-v1-small.yaml',
        help='Model YAML config or .pt checkpoint path',
    )
    parser.add_argument(
        '--data', default='CARPK_tiling.yaml',
        help='Dataset YAML path (passed to YOLO.train)',
    )
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--imgsz', type=int, default=256)
    parser.add_argument(
        '--single-cls', action=argparse.BooleanOptionalAction, default=True,
        help='Treat all objects as a single class',
    )
    parser.add_argument(
        '--no-wandb', action='store_true',
        help='Disable Weights & Biases logging',
    )
    parser.add_argument(
        '--wandb-project', default='ultralytics-test',
        help='W&B project name (ignored when --no-wandb is set)',
    )
    parser.add_argument('--no-export', action='store_true', help='Skip ONNX export after training')
    parser.add_argument('--opset', type=int, default=12, help='ONNX opset version for export')
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.no_wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project)
        except ImportError:
            print('WARNING: wandb is not installed — logging disabled. '
                  'Install it with: pip install wandb, or pass --no-wandb to suppress this message.')

    # Tile the dataset according to the config
    tiler = Tiler(args.tiling_config)
    tiler.get_split_dataset()

    # Load model and train
    model = YOLO(args.model)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        single_cls=args.single_cls,
    )

    num_layers = sum(1 for _ in model.model.model.modules()) - 1
    num_params = sum(p.numel() for p in model.model.model.parameters())
    print(f'Number of layers    : {num_layers}')
    print(f'Number of parameters: {num_params}')

    if not args.no_export:
        model.export(format='onnx', imgsz=[args.imgsz, args.imgsz], opset=args.opset)


if __name__ == '__main__':
    main()
