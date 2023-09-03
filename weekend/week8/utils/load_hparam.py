import os
import json

def load_hparams(args):
    #load hyperparameters(with json file)
    with open(os.path.join(args.trained_folder, "hparam.json"), "r") as f:
        hparams = json.load(f)  #dictionary

    for key, value in hparams.items():
        setattr(args, key, value)

    return args

# img_size = int(hparams[0].strip())  
# hidden_size = int(hparams[1].strip())
# num_classes = int(hparams[2].strip())
# batch_size = int(hparams[3].strip())
# lr = float(hparams[4].strip())
# epochs = int(hparams[5].strip())
# results_folder = hparams[6].strip()