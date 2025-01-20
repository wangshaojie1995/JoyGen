import os
import os.path as osp
import torch
from torch.hub import download_url_to_file, get_dir
from urllib.parse import urlparse

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def download_pretrained_models(file_ids, save_path_root):
    import gdown
    
    os.makedirs(save_path_root, exist_ok=True)

    for file_name, file_id in file_ids.items():
        file_url = 'https://drive.google.com/uc?id='+file_id
        save_path = osp.abspath(osp.join(save_path_root, file_name))
        if osp.exists(save_path):
            user_response = input(f'{file_name} already exist. Do you want to cover it? Y/N\n')
            if user_response.lower() == 'y':
                print(f'Covering {file_name} to {save_path}')
                gdown.download(file_url, save_path, quiet=False)
                # download_file_from_google_drive(file_id, save_path)
            elif user_response.lower() == 'n':
                print(f'Skipping {file_name}')
            else:
                raise ValueError('Wrong input. Only accepts Y/N.')
        else:
            print(f'Downloading {file_name} to {save_path}')
            gdown.download(file_url, save_path, quiet=False)
            # download_file_from_google_drive(file_id, save_path)



def load_file_from_url(url, model_dir=None, progress=True, file_name=None):
    """Ref:https://github.com/1adrianb/face-alignment/blob/master/face_alignment/utils.py
    """
    if model_dir is None:
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, 'checkpoints')

    os.makedirs(os.path.join(ROOT_DIR, model_dir), exist_ok=True)

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.abspath(os.path.join(ROOT_DIR, model_dir, filename))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        download_url_to_file(url, cached_file, hash_prefix=None, progress=progress)
    return cached_file

