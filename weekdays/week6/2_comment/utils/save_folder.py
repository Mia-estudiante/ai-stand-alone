import os
import json

def get_save_path(args):   
    #저장
    #상위 저장 폴더를 만들어야 함
    if not os.path.exists(args.results_folder):
        os.makedirs(args.results_folder)
    
    # 내가 저장을 할 하위 폴더를 만들어야 함 (하위 폴더가 앞으로 사용될 타겟 폴더가 됨) 
    target_folder_name = max([0]+[int(e) for e in os.listdir(args.results_folder)])+1
    save_path = os.path.join(args.results_folder, str(target_folder_name))
    os.makedirs(save_path)

    # 타겟 폴더 밑에 hparam 저장(json 파일 형태)
    with open(os.path.join(save_path, 'hparam.json'), 'w') as f:
        write_args = args.__dict__.copy()  #namespace -> dict
        del write_args['device']           #자료형으로 나타낼 수 없는 것은 json 파일에 쓰지 못하므로
        json.dump(write_args, f, indent=4)

    return save_path