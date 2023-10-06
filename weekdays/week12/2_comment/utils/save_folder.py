import os
import json

def save_path_name(args):
    if not os.path.exists(args.results_folder):
        os.makedirs(args.results_folder)

    #Step1. 하위 저장 폴더 생성
    target_folder_name = max([0]+[int(e) for e in os.listdir(args.results_folder)])+1
    save_path = os.path.join(args.results_folder, str(target_folder_name))
    os.makedirs(save_path)

    #Step2. json파일에 하이퍼파라미터 저장
    with open(os.path.join(save_path, 'hparam.json'), 'w') as f:
        write_args = args.__dict__.copy()
        del write_args['device']
        json.dump(write_args, f, indent=4)        

    return save_path
