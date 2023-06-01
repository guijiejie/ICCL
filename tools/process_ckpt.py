import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description='This script extracts backbone weights from a checkpoint')
    parser.add_argument('--checkpoint', default="test.pth", type=str, help='checkpoint file')
    parser.add_argument('--output', type=str, help='destination file name')
    args = parser.parse_args()
    return args

def bankeys(key):
    if 'attnpool_crop' in key:
        return False
    if 'model.transformer' in key:
        return False
    if 'token_embedding' in key:
        return False
    if 'ln_final' in key:
        return False
    if 'model.positional_embedding' in key:
        return False
    if 'text' in key:
        return False
    
    return True

def main():
    args = parse_args()
    assert args.output.endswith(".pth")
    print(f"trans {args.checkpoint} to {args.output}")
    ck = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    ck = ck['state_dict'] if 'state_dict' in ck else ck
    output_dict = dict(state_dict=dict(), author="myselfsup")
    has_backbone = False
    for key, value in ck.items():
        if key.startswith('module.'):
            key = key[len('module.'):]
        if key.startswith('backbone.') and bankeys(key):
            # if "bn" in key:
            #     key = key.replace("bn", "norm")
            output_dict['state_dict'][key] = value
            has_backbone = True
        # elif key.startswith('head.img_projector'):
        #     key = key.replace("head", "backbone")
        #     output_dict['state_dict'][key] = value
    if not has_backbone:
        raise Exception("Cannot find a backbone module in the checkpoint.")
    torch.save(output_dict, args.output)


if __name__ == '__main__':
    main()
