import os
import yaml 
import torch
import numpy as np

from PIL import Image
from munch import munchify
from torchvision import transforms


#------------------------
def load_conf_file(config_file_path):
    '''
        Load the conf file containing all the parameters with the dataset paths

        input:
            - config_file_path (str): path to the configuration file

        output
            - a dictionary containing the conf parameters
    '''
    with open(config_file_path) as fhandle:
        cfg = yaml.safe_load(fhandle)
        
    cfg = munchify(cfg)
    return cfg

#------------------------
def load_image(cfg, path, img_name, xchest=True):
    '''
    return the corresponding image (tensor and numpy): => (batch_size, C, H, W).

        Parameters:
            - path (str): image path location
        
        Returns:
            - PIL normalize (in [0, 1]) with shape (b, C,H,W)  
            - np image unormalize image
    '''
    pil_img = Image.open(os.path.join(path, img_name)) 

    if xchest:
        pil_img = pil_img.convert('RGB')
    
    normalization = [
        transforms.Resize((cfg.data.input_size)), 
        transforms.ToTensor(),
        transforms.CenterCrop(cfg.data.input_size),
        transforms.Normalize(cfg.data.mean, cfg.data.std)]
    
    test_preprocess_norm = transforms.Compose(normalization)
    
    ts_img = torch.unsqueeze(test_preprocess_norm(pil_img), dim=0)   
    pil_img = pil_img.resize((cfg.data.input_size, cfg.data.input_size))
    np_img = np.array(pil_img)                                
    
    return ts_img, np_img