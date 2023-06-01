import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument('--workdir', default="result/resnet50", type=str)

if __name__ == "__main__":
    args = parser.parse_args()
    if not os.path.isdir(args.workdir):
        raise ValueError(f'workdir={args.workdir} does not exists.')
    
    fns = os.listdir(args.workdir)
    jsonfn = None
    for fn in fns:
        if "json" in fn:
            jsonfn = os.path.join(args.workdir, fn)
            break
    
    if jsonfn is None:
        raise ValueError(f'{args.workdir} does not contain json log.')

    with open(jsonfn, "r") as f:
        lines = f.readlines()
        best_acc = 0
        best_line = ""
        for l in lines:
            record = json.loads(l)
            if record['mode'] == "val":
                if '0-level' in record and record['0-level'] > best_acc:
                    best_acc = record['0-level']
                    best_line = l
                if 'top-1' in record and record['top-1'] > best_acc:
                    best_acc = record['top-1']
                    best_line = l
        
        print(best_line, end='')
    
    with open(jsonfn, "a") as f:
        f.write(f'{best_line}')