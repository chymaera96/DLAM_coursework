import os
import torch
import numpy as np
import json
import glob
import shutil

def load_index(data_dir, ext=['wav','mp3'], max_len=4000):
    dataset = {}

    print(f"=>Loading indices from {data_dir}")
    json_path = os.path.join(data_dir, data_dir.split('/')[-1] + ".json")

    if not os.path.exists(json_path):
        for idx,fpath in enumerate(glob.iglob(os.path.join(data_dir,'**/*.*'), recursive=True)):
            if fpath.split('.')[-1] in ext and idx < max_len: 
                dataset[str(idx)] = fpath
            
        
        with open(json_path, 'w') as fp:
            json.dump(dataset, fp)
    
    else:
        with open(json_path, 'r') as fp:
            dataset = json.load(fp)

    assert len(dataset) > 0
    return dataset

def get_frames(y, frame_length, hop_length):
    # frames = librosa.util.frame(y.numpy(), frame_length, hop_length, axis=0)
    frames = y.unfold(0, size=frame_length, step=hop_length)
    return frames

def qtile_normalize(y, q, eps=1e-8):
    return y / (eps + torch.quantile(y,q=q))


def load_ckp(checkpoint_fpath, model, optimizer, scheduler):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    return model, optimizer, scheduler, checkpoint['epoch'], checkpoint['loss']

def save_ckp(state,epoch,model_name,model_folder):
    if not os.path.exists(model_folder): 
        print("Creating checkpoint directory...")
        os.mkdir(model_folder)
    torch.save(state, "{}/model_{}_epoch_{}.pth".format(model_folder,model_name,epoch))


def create_train_set(data_dir, size=8000):
    dest = os.path.join(data_dir, f'fma_{size}')
    if not os.path.exists(dest):
        os.mkdir(dest)

    for ix,fname in enumerate(os.listdir(data_dir)):
        fpath = os.path.join(data_dir, fname)
        if ix <= 8000 and fpath.endswith('mp3'):
            shutil.move(fpath,dest)
        if len(os.listdir(dest)) >= 8000:
            return dest
    # source_dir = os.path.join(data_dir, f'fma_{size}')
    # target_dir = data_dir
        
    # file_names = os.listdir(source_dir)
        
    # for file_name in file_names:
    #     shutil.move(os.path.join(source_dir, file_name), target_dir)

    
    return dest