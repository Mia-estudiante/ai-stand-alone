import os
import json

#load hyperparameters(with json file)
def load_hparams(args):
    with open(os.path.join(args.trained_folder, "hparam.json"), "r") as f:
        hparams = json.load(f)  #dictionary

    for key, value in hparams.items():
        setattr(args, key, value)

    return args