from lib.config import load_config
from nets.resnet_v2 import ResNetV2
from parse_args import parse_args


def main(args):
    cfg = load_config(args.cfg_name)
    model = ResNetV2(cfg, converter.num_classes)
    self.model.create_architecture()
    pass


if __name__ == "__main__":
    args = parse_args(infer=True)
    main(args)
