import os
import json

def load_hparams(args):
    with open(os.path.join(args.trained_folder, 'hparam.json')) as f:
        write_args = json.load(f)
    
    for key, value in write_args.items():
        setattr(args, key, value)
    return args