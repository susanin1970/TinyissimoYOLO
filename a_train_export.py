import argparse
import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train and export a TinyissimoYOLO model.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--version', default='v8', choices=['v1', 'v1-small', 'v5', 'v8'],
        help='Model version. NOTE: v1 requires manual edit of '
             'ultralytics/nn/modules/head.py — set self.reg_max=16 in Detect.__init__',
    )
    parser.add_argument(
        '--load', action='store_true',
        help='Resume training from the last checkpoint (requires --exp-id)',
    )
    parser.add_argument('--exp-id', default='exp1', help='Experiment ID used to locate checkpoints')
    parser.add_argument('--data', default='coco.yaml', help='Dataset YAML path (passed to YOLO.train)')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch', type=int, default=512)
    parser.add_argument('--imgsz', type=int, default=256)
    parser.add_argument('--optimizer', default='SGD')
    parser.add_argument('--project', default='results', help='Directory where results are saved')
    parser.add_argument('--name', default='exp', help='Sub-directory name inside --project')
    parser.add_argument(
        '--device', default='cuda',
        help='Device to use for training, e.g. "cuda", "cuda:0", "cpu"',
    )
    parser.add_argument('--no-export', action='store_true', help='Skip ONNX export after training')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.version == 'v1':
        print(
            'ERROR: TinyissimoYOLOv1 requires a manual source edit before training.\n'
            '  File : ultralytics/nn/modules/head.py\n'
            '  Class: Detect.__init__\n'
            '  Change: self.reg_max = 1  →  self.reg_max = 16\n'
            'Re-run this script after making the change.'
        )
        raise SystemExit(1)

    device = torch.device(args.device)

    if args.load:
        model_name = f'./results/{args.exp_id}/weights/last.pt'
        print(f'Resuming from checkpoint: {model_name}')
    else:
        model_name = f'./ultralytics/cfg/models/tinyissimo/tinyissimo-{args.version}.yaml'
        print(f'Starting new training with model: {model_name}')

    model = YOLO(model_name)

    model.train(
        data=args.data,
        project=args.project,
        name=args.name,
        optimizer=args.optimizer,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=device,
    )

    if not args.no_export:
        model.export(
            format='onnx',
            project=args.project,
            name=args.name,
            imgsz=[args.imgsz, args.imgsz],
        )


if __name__ == '__main__':
    main()
