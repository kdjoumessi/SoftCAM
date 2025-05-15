import os
import cv2
import torch
import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from typing import Any
from .func import load_conf_file
from .basic_tools import load_image
from .fully_convnet_tools import get_patch_location, topK_patches

from PIL import Image
from tqdm import tqdm
from torch.distributions import Categorical
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import accuracy_score, roc_auc_score
from torchcam import methods
from pytorch_grad_cam import ScoreCAM, LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from captum.attr import GuidedBackprop, IntegratedGradients, LayerGradCam, NoiseTunnel, LayerAttribution

############## General ###############
######################################

def simple_plot(fnames, grades, path_folder, size=(3,3), fs=12):
    fig = plt.figure(figsize=size)
    n = len(fnames)
    j = 0
    
    for i in range(n):
        j += 1
        ax = fig.add_subplot(1, n, j)
        img = plt.imread(os.path.join(path_folder, fnames[i]))
        ax.imshow(img)
        ax.set_title(f"{fnames[i].split('.')[0], grades[i]}", loc="center", fontsize=fs)
        ax.axis('off')
    plt.show()


#------------------------------------
def get_pred_with_heatmap(cfg, model, df, f_path, res=False, cname='conv_res', s=60, att=True, bar=True):
    ''' 
       description: run inference on each image and output predictions and corresponding activation maps
    '''    
    ypred, yhat = [], []
    activations_healthy, activations, all_activation = {}, {}, {}
    
    dat = df.copy()
    filenames   = df['filename'].tolist()

    iterate = tqdm(filenames) if bar else filenames
    
    for fname in iterate:
        ts_img, np_img = load_image(cfg, f_path, fname)
        ts_img, model = ts_img.to('cuda'), model.to('cuda')
        
        if res:
            pred = model(ts_img)
        elif att:
            pred, acti, _ = model(ts_img)
        else:
            pred, acti = model(ts_img)
        pred = pred.data.cpu()

        y_prob = torch.nn.functional.softmax(pred, dim=1)
        y_class = np.argmax(y_prob, axis = 1)
        
        ypred.append(y_prob.numpy()[0][1])
        yhat.append(y_class.item())

        if not res:
            acti = acti[0].detach().cpu().numpy()
            act = cv2.resize(acti[1], dsize=(s,s), interpolation=cv2.INTER_CUBIC)
            h_acti = cv2.resize(acti[0], dsize=(s,s), interpolation=cv2.INTER_CUBIC)
            
            activations[fname] = act
            all_activation[fname] = acti
            activations_healthy[fname] = h_acti

            
        #break
    
    dat[f'{cname}_pred'] = yhat
    dat[f'{cname}_conf'] =  np.round(np.array(ypred), 3)
    
    return dat, activations, activations_healthy, all_activation, yhat

#------------------------------------
def get_overlay_img(img, activation, alpha = 0.6, res_conv=True):
    '''
        alpha: Transparency factor
        only normalize for conv evidence map
    '''

    # Normalize the heatmap data to the range [0, 1]
    heatmap = (activation - np.min(activation)) / (np.max(activation) - np.min(activation)) if res_conv else activation
    
    # Apply a colormap to the heatmap
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    #print('heatmap_colored: ', heatmap_colored.shape)
    #print('image: ', img.shape)
    
    # Blend the image and heatmap
    overlay_image = cv2.addWeighted(img, 1 - alpha, cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB), alpha, 0)
    return overlay_image

#------------------------------------
def get_overlay_img_v2(img, activation, alpha = 0.6, res_conv=True):
    heatmap = (activation - np.min(activation)) / (np.max(activation) - np.min(activation)) if res_conv else activation
    heatmap = plt.cm.hot(heatmap)  
    heatmap = np.delete(heatmap, 3, 2)         # Remove alpha channel
    
    # overlay the heatmap on the original image using alpha blending
    overlay_image = heatmap[..., :3] * alpha + img / 255.0 * (1 - alpha)
    overlay_image = np.clip(overlay_image, 0, 1)
    return overlay_image

#------------------------
def model_eval(cfg, network, fnames, f_path, act=False):  
    '''
        (RSNA) Inference only 
    '''
    y_probs = np.zeros((0, 2), np.float32)
    
    torch.set_grad_enabled(False)
    
    for fname in tqdm(fnames):
        ts_img, np_img = load_image(cfg, f_path, fname)
        ts_img = ts_img.to('cuda')

        if act:
            y_pred, _ = network(ts_img)
        else:
            y_pred = network(ts_img)
           
        y_prob = torch.nn.functional.softmax(y_pred, dim=1)
        y_probs = np.concatenate([y_probs, y_prob.detach().cpu().numpy()]) 

    y_class_pred = np.argmax(y_probs, axis = 1)
    
    dic = { 'predict_proba': y_probs, # np.around(bag_y_probs, decimals=3)
            'predict_class': y_class_pred}   
    return dic

############### Plots ################
######################################
#------------------------------------
def plot_img_heat_att(imgs, dic, size, fs=10): 
    '''
       plot image, heatmap, and heatmap overlay on the image
       imgs: [np_img, heatmap, overlay] => list of images
    '''
    fig = plt.figure(figsize=size, layout='constrained')
    n = len(imgs)
    j = 0
    
    for i, img in enumerate(imgs):
        j += 1
        ax = fig.add_subplot(1, n, j)
        img = ax.imshow(img, cmap='viridis')
        ax.set_title(dic[i], loc="center", fontsize=fs)
        ax.axis('off')
        #plt.colorbar(img, ax=ax)
    #plt.show()

#------------------------------------
def plot_img_heat_att_v3(imgs, dic_overlay, size, fs=10): 
    '''
       plot image, heatmap overlay on the image
       imgs: [np_img] => list of input images
       dic_overlay: {key: overlay}
    '''
    fig = plt.figure(figsize=size, layout='constrained')
    nrow = len(imgs)
    ncol = len(dic_overlay) + 1
    j = 0
    
    for i, img_ in enumerate(imgs):
        j += 1
        ax = fig.add_subplot(nrow, ncol, j)
        img = ax.imshow(img_, cmap='viridis')
        ax.axis('off')

        for key, img_ in dic_overlay.items():
            j += 1
            ax = fig.add_subplot(nrow, ncol, j)
            img = ax.imshow(img_, cmap='viridis')
            if i ==0:
                ax.set_title(key, loc="center", fontsize=fs)
                ax.axis('off')
                

#------------------------------------
def get_posthoc_explanation(cfg, filenames, xai, f_path, s=16):
    '''
        description: run inference and return the attribution map for each method.

        params:
            - filenames (list)
            - xai (dic): dictionary where keys are saliency name and value the corresponding model

        output:
            - dic[dic2]: dic.key = attributions, dic2.key=image_name, val = attribution
    '''
    dic_attrib = {name: {} for name in xai.keys()}
    
    for fname in tqdm(filenames):
        ts_img, np_img = load_image(cfg, f_path, fname)
        ts_img.requires_grad_() 
        ts_img = ts_img.to('cuda')

        for xai_name, explainer in xai.items():  
            d_attrib = explainer.attribute(ts_img, target=1)
            d_attrib = d_attrib.squeeze().cpu().detach().numpy()

            h_attrib = explainer.attribute(ts_img, target=0)
            h_attrib = h_attrib.squeeze().cpu().detach().numpy()
            
            if xai_name =='GradCam':
                d_attrib = cv2.resize(d_attrib, dsize=(s,s), interpolation=cv2.INTER_CUBIC)
                h_attrib = cv2.resize(h_attrib, dsize=(s,s), interpolation=cv2.INTER_CUBIC)
                
            elif xai_name =='Smooth_GradCam':
                d_attrib = cv2.resize(d_attrib, dsize=(s,s), interpolation=cv2.INTER_CUBIC)       
                h_attrib = cv2.resize(h_attrib, dsize=(s,s), interpolation=cv2.INTER_CUBIC) 
            else:
                d_attrib = np.transpose(d_attrib, (1,2,0)) 
                d_attrib = np.sum(d_attrib, axis=2)                
                d_attrib = cv2.resize(d_attrib, dsize=(s,s), interpolation=cv2.INTER_CUBIC)

                h_attrib = np.transpose(h_attrib, (1,2,0)) 
                h_attrib = np.sum(h_attrib, axis=2) 
                h_attrib = cv2.resize(h_attrib, dsize=(s,s), interpolation=cv2.INTER_CUBIC)

            dic_attrib[xai_name][fname] = (h_attrib, d_attrib)
        #break     
    return dic_attrib


#------------------------------------
def get_posthoc_explanation_v3(cfg, model, filenames, xai, f_path, s=16, nclass=2):
    '''
        description: run inference and return the attribution map for each method.

        params:
            - filenames (list)
            - xai (dic): dictionary where keys are saliency name and value the corresponding model

        output:
            - dic[dic2]: dic.key = attributions, dic2.key=image_name, val = attribution
    '''
    dic_attrib = {name: {} for name in xai.keys()}
    
    for fname in tqdm(filenames):
        tmp_cam = {}
        ts_img, np_img = load_image(cfg, f_path, fname)
        ts_img = ts_img.to('cuda')

        for xai_name, explainer in xai.items():  
            
            if xai_name == 'ScoreCAM': # torchcam' ['ScoreCAM', 'LayerCAM']:
                with torch.no_grad():
                    output = model(ts_img)
                    
                for i in range(nclass):
                    cam_map = explainer(i)[0]
                    cam_map = cam_map.squeeze().cpu().numpy()
                    tmp_cam[i] = cv2.resize(cam_map, dsize=(s,s), interpolation=cv2.INTER_CUBIC)  
                    
            elif xai_name == 'LayerCAM':
                output = model(ts_img)
                #class_idx = torch.argmax(output).item()

                for i in range(nclass):                        
                    targets = [ClassifierOutputTarget(i)]
                    # Generate heatmap using ScoreCAM
                    grayscale_cam = explainer(input_tensor=ts_img, targets=targets)

                    # Convert the grayscale cam output (first entry) to a 2D heatmap
                    grayscale_cam = grayscale_cam[0]
                    tmp_cam[i] = cv2.resize(grayscale_cam, dsize=(s,s), interpolation=cv2.INTER_CUBIC) 
                    
                    #raise NotImplementedError('Attribution method not implemented.')
            else:  
                ts_img.requires_grad_(True)

                for i in range(nclass):
                    attrib = explainer.attribute(ts_img, target=i)
                    attrib = attrib.squeeze().cpu().detach().numpy()
                
                    if xai_name =='GradCAM':
                        attrib = cv2.resize(attrib, dsize=(s,s), interpolation=cv2.INTER_CUBIC)                        
                    elif xai_name =='Smooth_GradCam':
                        attrib = cv2.resize(attrib, dsize=(s,s), interpolation=cv2.INTER_CUBIC)    
                    else:
                        attrib = np.transpose(attrib, (1,2,0)) 
                        attrib = np.sum(attrib, axis=2)                
                        attrib = cv2.resize(attrib, dsize=(s,s), interpolation=cv2.INTER_CUBIC)
                        
                    tmp_cam[i] = attrib

            dic_attrib[xai_name][fname] = (tmp_cam[0], tmp_cam[1])
        #break     
    return dic_attrib

#------------------------------------
def get_attributions(cfg, dic_models, model_name, attrib, f_path_name, normalized=False, relu=False, ss=512):
    model = dic_models[model_name]
    if 'Vgg' in model_name:
        layer = model.features[29]  # last feature map before the classifier layer
    elif 'ResNet' in model_name:
        layer = model.layer4[-1]  # Use the last residual block's convolutional layer
    
    attribution_maps = {}

    for (f_path, fname) in tqdm(f_path_name):
        print(f_path, fname)
        input_tensor, np_img = load_image(cfg, f_path, fname)
        input_tensor.requires_grad_()  # input_tensor.requires_grad = True 
        input_tensor = input_tensor.to('cuda')        
    
        for attrib_name in attrib:
            if attrib_name == 'Score-CAM':
                #name = f'{attrib_name}_{fname}'
                cam_extractor = ScoreCAM(model, target_layer=layer)
            elif attrib_name == 'Layer-CAM':
                cam_extractor = LayerCAM(model, target_layer=layer)
            elif attrib_name == 'Grad-CAM':
                cam_extractor = LayerGradCam(model, layer)
            elif attrib_name == 'GuidedBP':
                cam_extractor = GuidedBackprop(model)
            else:
                raise NotImplementedError('Not implemented method.')

            output = model(input_tensor)
            pred_class = torch.argmax(output).item() # output.argmax(dim=1) 

            if attrib_name in ['Grad-CAM', 'GuidedBP']:
                # Compute Grad-CAM attributions for the predicted class
                heatmap = cam_extractor.attribute(input_tensor, target=pred_class)
            else:                        
                heatmap = cam_extractor(pred_class, output)[0]  # Get ScoreCAM heatmap

            # Convert to NumPy and Normalize
            np_heatmap = heatmap.squeeze().cpu().detach().numpy()

            if attrib_name == 'GuidedBP':
                np_heatmap = np.transpose(np_heatmap, (1,2,0))
                np_heatmap = np.sum(np_heatmap, axis=2)
                #attributions = (attributions * 255).astype(np.uint8)

            if relu:
                np_heatmap = np.maximum(np_heatmap, 0)  # Apply ReLU to remove negative values

            if normalized:       
                #print(np_heatmap.min(), np_heatmap.max())
                # attributions = np_heatmap / np_heatmap.max()  # Normalize
                #heatmap = (np_heatmap - np_heatmap.min()) / (np_heatmap.max() - np_heatmap.min()) if normalized else np_heatmap
                pass
                
            attribution_maps[attrib_name] = np_heatmap
                
    return attribution_maps

#------------------------------------
def plot_precision(dic_score, title, bb=30, fs=12, lfs=8, out=False, size=(11, 8)):
    '''
        description: plot topk precision curve
    '''
    fig, ax = plt.subplots(figsize=size, facecolor= "white", dpi=200, layout="constrained")
    plt.rcParams['font.size'] = f'{fs-2}'
    #colors = 'bgrcmy'
    
    xval = range(1, bb)
    for i, (xai_name, score) in enumerate(dic_score.items()):
        ax.plot(xval, score, label=xai_name, linewidth=0.8) # color=colors[i]
        #print(xai_name)
    ax.set_title(title)
    ax.set_xlabel('Top-K patches')
    ax.set_ylabel('Precision')
    ax.grid()

    if out:
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0., fontsize=lfs)
    else:
        ax.legend(loc="upper right", fontsize=lfs)
        
#------------------------------------
def plot_deletion_curve(performance, marker='o', label='Deletion Curve', k=30, frac=True, fs=12, lfs=8, size=(11, 8)):
    '''
        description: area under the deletion curve
    '''
    fig, ax = plt.subplots(figsize=size, facecolor= "white", dpi=200, layout="constrained")
    plt.rcParams['font.size'] = f'{fs-2}'

    if frac:
        # Normalize x-axis (deletion fraction)
        x = np.linspace(0, 1, k)
        xtitle = 'Fraction of Features Deleted'
    else:
        x = range(0, k)
        xtitle = 'Number of patch Deleted'

    del_fraction = np.linspace(0, 1, k)

    print('Area under the deletion curve')
    for key, yval in performance.items():        
        auc_deletion = np.trapz(yval, x=del_fraction) 

        if key=='GradCam':
            print(f'{key}: \t\t {auc_deletion}')  
        elif key in ['GuidedBackprop', 'IntegratedGradients']:
            print(f'{key}: \t {auc_deletion}') 
        else:
            print(f'{key}: \t\t\t {auc_deletion}')  

        plt.plot(x, yval, marker=marker, label=key, markersize=1, linewidth=0.5)
        
    ax.set_xlabel(xtitle) #
    ax.set_ylabel('Model Performance')
    ax.set_title('Sensitivity Analysis: Deletion Curve')
    ax.legend(fontsize=lfs)

#------------------------------------
def plot_deletion_curve_v2(dic_perform, bb=16, marker='o', label='Deletion Curve', frac=True, ls=1, ms=2, fs=12, lfs=8, size=(11, 8)):
    '''
        description: area under the deletion curve
    '''
    fig, ax = plt.subplots(figsize=size, facecolor= "white", dpi=200, layout="constrained")
    plt.rcParams['font.size'] = f'{fs-2}'
    
    if frac:
        # Normalize x-axis (deletion fraction)
        x = np.linspace(0, 1, bb)
        xtitle = 'Fraction of Features Deleted'
    else:
        x = range(0, bb)
        xtitle = 'Number of patch Deleted'

    for i, (name, score) in enumerate(dic_perform.items()):
        del_fraction = np.linspace(0, 1, bb)
        auc_deletion = np.trapz(score, x=del_fraction) 
        print(f'Area under the deletion curve. {name}: {auc_deletion}')    
        plt.plot(x, score, marker=marker, label=name, markersize=ms, linewidth=ls)
        
    ax.set_xlabel(xtitle)
    ax.set_ylabel('Model Performance')
    ax.set_title('Sensitivity Analysis: Deletion Curve')
    ax.legend(fontsize=lfs)

#------------------------------------
def plot_img_heat_att_v2(imgs, dic, size, df=None, file=None, fs=10, l=1): 
    '''
        RSNA plot with bounding boxes
    '''
    fig = plt.figure(figsize=size, layout='constrained')
    n = len(imgs)
    j = 0
    
    for i, img in enumerate(imgs):
        j += 1
        ax = fig.add_subplot(1, n, j)
        img = ax.imshow(img, cmap='viridis')
        ax.set_title(dic[i], loc="center", fontsize=fs)
        ax.axis('off')

    tmp = df[df.patientId==file[0].split('.')[0]]
    for _, row in tmp.iterrows():
        rect = patches.Rectangle((row['x'], row['y']), row['width'], row['height'], linewidth=l, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

#------------------------------------
def plot_xchest_with_bb(data, dat, path_folder, size=(20,5), fs=12, l=1):
    '''
        plot RSNA images with bb 
    '''
    fig = plt.figure(figsize=size, layout='constrained')
    n = len(data)

    fnames = data.filename.tolist()
    
    for idx, fname in enumerate(fnames):
        ax = fig.add_subplot(1, n, idx+1)
        img = Image.open(os.path.join(path_folder, fname)).convert('RGB')
        img = np.asarray(img.resize((512,512)))
        ax.imshow(img)

        tmp = dat[dat.patientId==fname[:-4]]
        for _, row in tmp.iterrows():
            rect = patches.Rectangle((row['x'], row['y']), row['width'], row['height'], linewidth=l, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        
        ax.axis('off')
    plt.show()


######### Precision analysis #########
######################################
#------------------------------------
def extract_patch(fname, heatmap, path, mean=False, ps=[33, 33], size=(512,512)):
    '''
        description: from a given image with the corresponding heatmap, split the image into non-overlapping patches, compute the energy falling into each patch as the max of mean activation from the heatmap.

        output [(i, j), energy]: a list of patch locations with the corresponding energy

        params:
             - fname (str)
             - heatmap (np.array): heatmap in high resolution  
             - path (str)
             - mean (bool): if True compute the energy as the mean value within the heatmap, otherwise the max value is used
             - ps ([patch_size, patch_size])
    '''

    image = cv2.imread(os.path.join(path, fname))
    image = cv2.resize(image, size)
    src = cv2.GaussianBlur(image, (3, 3), 0)
    imgray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _, bw_img = cv2.threshold(imgray, 80, 255, cv2.THRESH_BINARY)
    indices = cv2.findNonZero(bw_img)

    # Get the image dimensions
    height, width, _ = image.shape
    mask = np.ones((height, width), dtype=np.uint8)

    # Calculate the maximum coordinates for patch placement
    max_x = width 
    min_y = np.max(np.min(np.squeeze(indices), axis=0))
    max_y = np.max(np.max(np.squeeze(indices), axis=0))
    #print(f'max_x: {max_x}, min_y: {min_y}, max_y: {max_y}')

    num_patches_row = (((max_y - min_y) // ps[1]) + 1)
    num_patches_col = ((max_x // ps[0]) + 1)
    #print(f'num_patches row: {num_patches_row}, col: {num_patches_col}')

    # Initialize a list to store the average values and indices of each patch
    patch_averages = []
    
    for i in range(min_y, max_y, ps[1]):
        for j in range(0, max_x, ps[0]):
            patch = heatmap[i:i+ps[1], j:j+ps[0]]
            average_value = np.mean(patch) if mean else np.max(patch)
            patch_averages.append(((i, j), average_value))
    
    # Sort the patches by average value in descending order
    # this will be the list to be saved.
    sorted_patches = sorted(patch_averages, key=lambda x: x[1], reverse=True)

    return sorted_patches

#------------------------------------
def get_precision(acts, a_path, pred, thresh=1, ps=33, s=8, act_size=60, xchest=False, bar=True):
    '''
        description: compute the precision given a threshold

        params:
            - acts (dic): dictionary where keys are filename and values low dimension activations for the disease class
            - a_path (str)

        outputs:
            - df: dataframe containing for each image the location of positive activation from the heatmap, the patch scores as well as whether the corresponding patch contains ground truth annotation or not
            - score (scalar)
    '''
    n = act_size*act_size
    fnames, coords, labels, patch_score, predict, min_max = [], [], [], [], [], []

    iterate = tqdm(enumerate(acts.items())) if bar else enumerate(acts.items())
    for jj, (fname, act) in iterate:
        heatmap = cv2.resize(act, dsize=(act_size,act_size), interpolation=cv2.INTER_CUBIC)
        flat_heatmap = torch.from_numpy(heatmap.flatten())
        mask = plt.imread(os.path.join(a_path, fname))
        max_, min_ = heatmap.max(), heatmap.min()

        top_indices, top_scores, _, _ = topK_patches(flat_heatmap.numpy(), k=n, dx=-2, dy=2, max_=n, threshold=thresh, round_=False)
        loc_coords = get_patch_location(top_indices, s=s, size=act_size)

        for j, (y,x) in enumerate(loc_coords):
            patch_mask = mask[y:y+ps, x:x+ps] if xchest else mask[x:x+ps, y:y+ps]
                
            has_positive_activation = int(patch_mask.max())

            fnames.append(fname); coords.append((x,y))
            labels.append(int(has_positive_activation))
            patch_score.append(round(top_scores[j], 3))
            min_max.append((round(min_, 2), round(max_, 2)))
            predict.append(pred[jj])
                
    result = pd.DataFrame({'filename':fnames, 'coords':coords, 'scores':patch_score, 'min_max':min_max, 'labels':labels,
                          'predict': predict})
    score = round(result.labels.mean(), 3)
    return result, score

#------------------------------------
def get_non_overlap_patches_scores(activations, path, bar=True, size=(512, 512)):
    '''
        description: get non-overlapping patches from each image
    '''
    img_patch = {}

    iterate = tqdm(activations.items()) if bar else activations.items()
    
    for fname, heat in iterate:
        heat = cv2.resize(heat, dsize=size, interpolation=cv2.INTER_CUBIC)
        img_patch[fname] = extract_patch(fname, heat, path, size=size)
        
    return img_patch

#------------------------------------
def get_non_overlap_precision(acts, pred, path, a_path, thresh=1, ps=33, xchest=False, size=(512,512)):
    '''
        description: compute the precision from non-overlapping patches
    '''
    fnames, coords, labels, patch_score, predict = [], [], [], [], []

    act_patches = get_non_overlap_patches_scores(acts, path, bar=False, size=size)

    for jj, fname in enumerate(tqdm(act_patches.keys())):
        mask = plt.imread(os.path.join(a_path, fname))

        coord_score = act_patches[fname]

        for j, (xy, score) in enumerate(coord_score):
            y, x = xy
            if score > thresh:
                patch_mask = mask[y:y+ps, x:x+ps] if xchest else mask[x:x+ps, y:y+ps]                
                has_positive_activation = int(patch_mask.max())

                fnames.append(fname); coords.append((x,y))
                labels.append(int(has_positive_activation))
                patch_score.append(round(score, 3))
                predict.append(pred[jj])

    result = pd.DataFrame({'filename':fnames, 'coords':coords, 'scores':patch_score, 'labels':labels, 'predict':predict})
    score = round(result.labels.mean(), 3)
    
    return result, score, act_patches

#------------------------------------
def get_topk_precision(dic_act, pred, a_path, thresh=1, k=30, ps=33, s=8, act_size=60, xchest=False, posthoc=None, bar=True):
    '''
    description: get the topk precision

    params:
        - dic_act (dic): dictionary where keys are filename and values low dimension activations for the disease class
    '''
    n = act_size*act_size
    scores, score_tp = [], []
    dic_result = {}

    acts = dic_act[posthoc] if posthoc else dic_act

    iterate = tqdm(range(1, k)) if bar else range(1, k)
    for i in iterate:
        fnames, coords, labels, patch_score, min_max, predict = [], [], [], [], [], []
        
        for jj, (fname, heatmap) in enumerate(acts.items()):
            heatmap = cv2.resize(heatmap, dsize=(act_size,act_size), interpolation=cv2.INTER_CUBIC)
            flat_heatmap = torch.from_numpy(heatmap.flatten())
            mask = plt.imread(os.path.join(a_path, fname))
            max_, min_ = heatmap.max(), heatmap.min()
    
            top_indices, top_scores, _, _ = topK_patches(flat_heatmap.numpy(), k=i, dx=-2, dy=2, max_=n, threshold=thresh, round_=False)
            loc_coords = get_patch_location(top_indices, s=s, size=act_size)
    
            for j, (y,x) in enumerate(loc_coords):
                patch_mask = mask[y:y+ps, x:x+ps] if xchest else mask[x:x+ps, y:y+ps]
                
                has_positive_activation = int(patch_mask.max())
    
                fnames.append(fname); coords.append((x,y))
                labels.append(int(has_positive_activation))
                patch_score.append(round(top_scores[j], 3))
                min_max.append((round(min_, 2), round(max_, 2)))
                predict.append(pred[jj])
                
        result = pd.DataFrame({'filename':fnames, 'coords':coords, 'scores':patch_score, 'min_max':min_max, 'labels':labels, 'predict':predict})
        scores.append(round(result.labels.mean(), 3))
        score_tp.append(round(result[result.predict==1]['labels'].mean(), 3))
        dic_result[i] = result
    return dic_result, scores, score_tp


#------------------------------------
def get_topk_non_overlap_precision(dic_act, pred, path, a_path, thresh=1, k=30, ps=33, xchest=False, size=(512,512), posthoc=None):
    '''
        description: compute the topk precision from non-overlapping patches
    '''
    scores, score_tp = [], []
    dic_result = {}

    acts = dic_act[posthoc] if posthoc else dic_act
    act_patches = get_non_overlap_patches_scores(acts, path, bar=False, size=size)

    for i in tqdm(range(1, k)):
        fnames, coords, labels, patch_score, predict = [], [], [], [], []

        for jj, fname in enumerate(act_patches.keys()):
            mask = plt.imread(os.path.join(a_path, fname))
    
            coord_score = act_patches[fname][:i]
    
            for j, (xy, score) in enumerate(coord_score):
                y, x = xy
                if score > thresh:
                    patch_mask = mask[y:y+ps, x:x+ps] if xchest else mask[x:x+ps, y:y+ps]                
                    has_positive_activation = int(patch_mask.max())
    
                    fnames.append(fname); coords.append((x,y))
                    labels.append(int(has_positive_activation))
                    patch_score.append(round(score, 3))
                    predict.append(pred[jj])

        result = pd.DataFrame({'filename':fnames, 'coords':coords, 'scores':patch_score, 'labels':labels, 'predict':predict})
        dic_result[i] = result
        scores.append(round(result.labels.mean(), 3))
        score_tp.append(round(result[result.predict==1]['labels'].mean(), 3))
    
    return dic_result, scores, score_tp


######## Sensitivity analysis ########
######################################
#------------------------------------
def get_deletion_analysis(cfg, network, patch_score, img_path, k=15, ps=33, bar=True, act=False):
    '''
        description: Simulate a deletion curve

        params:
            - network: torch model
            - patch_score (dic): key = image name, val = list [(i,j) score]
    ''' 
    perform = [1]   
    dic_result, dic_img = {}, {}

    iterate = tqdm(range(1, k)) if bar else range(1, k)
    for i in iterate:
        imgs = {}
        files, occ_pred, occ_hat, preds, hats, label = [], [], [], [], [], []
        
        for fname in patch_score.keys():
            ts_img, np_img = load_image(cfg, img_path, fname)
            np_img = np.transpose(np_img, (2, 0, 1))
            np_img = np.expand_dims(np_img, axis=0) 
            ts_img = ts_img.to('cuda')

            pred, hat = get_prediction_v2(network, ts_img, act=act)
            loc_scores = patch_score[fname][:i]

            for j, (xy, score) in enumerate(loc_scores): 
                x, y = xy
                ts_img[0, :, x:x+ps, y:y+ps] = 0
                np_img[0, :, x:x+ps, y:y+ps] = 0

            imgs[fname] = np.transpose(np_img[0], (1,2,0))
            pred_, hat_ = get_prediction_v2(network, ts_img, act=act) 

            files.append(fname); occ_pred.append(pred_); occ_hat.append(hat_)
            preds.append(pred); hats.append(hat); label.append(1)
            
        result = pd.DataFrame({'filename':files, 'label':label, 'preds':preds, 'hats':hats, 'occ_pred':occ_pred, 'occ_hat':occ_hat})
        
        dic_img[i] = imgs
        dic_result[i] = result
        perform.append(round(accuracy_score(result.label, result.occ_hat), 4))
        
    return dic_result, perform, dic_img

#------------------------------------
def get_precision_sensitivity(acts, thresh=1, ps=33, s=8, act_size=60, bar=True):
    '''
        description: get dataframe with scores and patch location similar to the bagnet with soft overlapping
    '''
    n = act_size*act_size
    fnames, coords, patch_score, min_max = [], [], [], []

    iterate = tqdm(enumerate(acts.items())) if bar else enumerate(acts.items())
    for jj, (fname, act) in iterate:
        heatmap = cv2.resize(act, dsize=(act_size,act_size), interpolation=cv2.INTER_CUBIC)
        flat_heatmap = torch.from_numpy(heatmap.flatten())
        max_, min_ = heatmap.max(), heatmap.min()

        top_indices, top_scores, _, _ = topK_patches(flat_heatmap.numpy(), k=n, dx=-2, dy=2, max_=n, threshold=thresh, round_=False)
        loc_coords = get_patch_location(top_indices, s=s, size=act_size)

        for j, (y,x) in enumerate(loc_coords):
            #patch_mask = mask[y:y+ps, x:x+ps]
            fnames.append(fname); 
            coords.append((y,x))
            patch_score.append(round(top_scores[j], 3))
            min_max.append((round(min_, 2), round(max_, 2)))
                
    result = pd.DataFrame({'filename':fnames, 'coords':coords, 'scores':patch_score, 'min_max':min_max})
    return result

#------------------------------------
def get_overlap_patches_scores(data):
    '''
        description: get location-scores faithful to the bagnet with soft overlapping
    '''
    img_patches = {}
    
    fnames = data.filename.tolist()
    for fname in tqdm(fnames):
        patch = []
        tmp = data[data.filename==fname].reset_index(drop=True)

        for _, row in tmp.iterrows():
            patch.append((row['coords'], row['scores']))
            
        img_patches[fname] = patch
        
    return img_patches

#------------------------------------
def get_random_activation(fnames, low=-10.0, high =10.0, shape=(60,60)):
    random_activation = {}

    for seed, ff in enumerate(fnames):
        np.random.seed(seed)
        random_activation[ff] = np.random.uniform(low, high, size=shape)

    return random_activation
    

########## Model selection ###########
######################################

#------------------------------------
def run_eval(cfg, df, model_surf, pref, model_p, img_path, anot_p, thresh=1, s=60, k=30, xchest=False):
    overall_prec = []
    topk_prec = {}
    
    for i, surf in tqdm(enumerate(model_surf)):
        cmodel = torch.load(os.path.join(model_p, surf + pref), weights_only=False)
        cmodel = cmodel.to('cuda')
        cmodel.eval()
        print(i)
        
        _, acti, _, _, pred_ = get_pred_with_heatmap(cfg, cmodel, df, img_path, cname='conv_res', s=s, res=False, att=False, bar=False) 
        _, score = get_precision(acti, anot_p, pred_, thresh=thresh, bar=False, xchest=xchest, act_size=s)
        overall_prec.append(score)

        _, topk_conv_scores, topk_conv_scores_tp = get_topk_precision(acti, pred_, anot_p, thresh=thresh, k=k, bar=False, xchest=xchest, act_size=s)
        topk_prec[surf] = (topk_conv_scores, topk_conv_scores_tp)
        #break
    return overall_prec, topk_prec

#------------------------------------
def plot_overall_precision(reg_prec, marker='o', i=0, fs=12, size=(11, 8)):
    '''
        description: area under the deletion curve
    '''
    fig, ax = plt.subplots(figsize=size, facecolor= "white", dpi=200, layout="constrained")
    plt.rcParams['font.size'] = f'{fs-2}'

    reg = [elt[0] for elt in reg_prec]
    prec = [elt[1] for elt in reg_prec]      

    ax.plot(reg, prec, marker=marker, markersize=2)
    ax.scatter(reg[i], prec[i], c='r')
    #plt.xscale('log')
    ax.set_xlabel('Regularization') #
    ax.set_ylabel('Overall precision')
    ax.set_title('Precision Analysis')

#------------------------------------
def plot_topk_reg(topk_dic, title, i=0, marker='o', fs=12, lfs=8, out=True, size=(11, 8)):
    '''
        description: area under the deletion curve
    '''
    fig, ax = plt.subplots(figsize=size, facecolor= "white", dpi=200, layout="constrained")
    plt.rcParams['font.size'] = f'{fs-2}'
    xx = range(0, len(topk_dic['0'][0]))

    for key, val in topk_dic.items():
        ax.plot(xx, val[i], marker=marker, markersize=1, linewidth=0.5, label=key)
        
    ax.set_xlabel('Topk patches') 
    ax.set_ylabel('Precision')
    ax.set_title(title)
    if out:
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0., fontsize=lfs)
    else:
        ax.legend(fontsize=lfs)

#------------------------------------
def plot_overall_accuracy(data, precs, i=0, j=0, marker='o', k=0, fs=12, lfs=6, log_ax=False, size=(11, 8)):
    '''
        description: accuracy

        params:
            data (list): [(ACC, AUC, reg, 'reg')]
            i (int): idx for yval
            j (int): idx for xaxis
            k (int): trade-off
    '''
    fig, ax = plt.subplots(figsize=size, facecolor= "white", dpi=200, layout="constrained")
    plt.rcParams['font.size'] = f'{fs-2}'

    y_val = [elt[i] for elt in data]
    x_axis = [elt[j] for elt in data]      

    #ax.scatter(x_axis, y_val, s=1, label='Accuracy')
    ax.plot(x_axis, y_val, marker=marker, markersize=1, linewidth=0.5, label='Accuracy')
    ax.scatter(x_axis[k], y_val[k], c='c', s=4)

    #ax.scatter(x_axis, precs, s=1, label='Precision')
    ax.scatter(x_axis[k], precs[k], c='c', s=4)
    ax.plot(x_axis, precs, marker=marker, markersize=1, linewidth=0.5, label='Precision')
    
    if log_ax:
        plt.xscale('log')
    ax.set_xlabel('Regularization') #
    ax.set_ylabel('Accuracy - Precision')
    ax.set_title('Accuracy - Precision Analysis')
    ax.legend(fontsize=lfs)


##### RSNA activation precision ######
######################################
#------------------------------------
def signed_min_max_scale(array, range_min=-1, range_max=1):
    '''
        Separate positive and negative values
    '''
    pos_mask = array > 0
    neg_mask = array < 0

    # Scale positive values
    if np.any(pos_mask):  # Check if there are positive values
        pos_min = np.min(array[pos_mask])
        pos_max = np.max(array[pos_mask])
        array[pos_mask] = range_min + (array[pos_mask] - pos_min) * (range_max - range_min) / (pos_max - pos_min)

    # Scale negative values
    if np.any(neg_mask):  # Check if there are negative values
        neg_min = np.min(array[neg_mask])
        neg_max = np.max(array[neg_mask])
        array[neg_mask] = -range_min - (array[neg_mask] - neg_max) * (range_max - range_min) / (neg_max - neg_min)

    return array

#------------------------------------
def get_activation_precision(acts, path, a_path, scale=True, n=1, thresh=0, test=False, size=(512,512)):
    '''
        ToDo
    '''

    scores = []
    dic_acti, dic_sign_acti, dic_mask, dic_img, dic_score = {}, {}, {}, {}, {}

    fnames = list(acts.keys())
    
    if test:
        fnames  = random.sample(fnames, n)
        print(fnames)
    
    for fname in fnames: 
        image = cv2.imread(os.path.join(path, fname))
        image = cv2.resize(image, size)

        mask = plt.imread(os.path.join(a_path, fname))
        #mask = cv2.imread(os.path.join(a_path, fname))
        #mask_float32 = mask.astype(np.float32)
        #img_nor = mask_float32 / 255.0

        acti = cv2.resize(acts[fname], dsize=size, interpolation=cv2.INTER_CUBIC)
        
        sign_acti = acti.copy()
        
        if scale:
            sign_acti = signed_min_max_scale(sign_acti) # uniformaize activation in the same range 
        
        #sign_acti[sign_acti <= thresh] = 0.
        #sign_acti[sign_acti > thresh] = 1.

        sign_acti = sign_acti > thresh

        act_pres = (mask * sign_acti).sum() / sign_acti.sum()
        scores.append(round(act_pres, 4))

        dic_img[fname] = image
        dic_mask[fname] = mask
        dic_acti[fname] = acti
        dic_score[fname] = act_pres
        dic_sign_acti[fname] = sign_acti
        
    return np.array(scores), dic_score, dic_sign_acti, dic_acti, dic_mask, dic_img

#------------------------------------
def get_activation_precision_v3(acts, a_path, size=(512,512)):

    scores = []
    scores_sens = []
    dic_acti, dic_sign_acti, dic_mask, dic_img, dic_score = {}, {}, {}, {}, {}

    fnames = list(acts.keys())
    
    for fname in fnames: 
        mask = plt.imread(os.path.join(a_path, fname))

        acti = cv2.resize(acts[fname], dsize=size, interpolation=cv2.INTER_CUBIC)
        sign_acti = np.where(acti > 0, acti, 0)
        sign_acti = scale_image(sign_acti, 1)
        accuracy = calculate_mass_within(sign_acti, mask)

        scores.append(round(accuracy[0], 4))
        scores_sens.append(round(accuracy[1], 4))
        
    return np.array(scores), np.array(scores_sens)

#### Weighting Game Accuracy #####
######################################
def scale_image(img: Any, max: int = 1) -> Any:
    ''' Binarized the input'''
    eps = np.finfo(float).eps
    out = (img - img.min()) * (1/((img.max() - img.min()) + eps) * max)
    return out


#------------------------------------
def calculate_mass_within(saliency_map: np.array, class_mask: np.array) -> float:
    eps = np.finfo(float).eps
    mass = saliency_map.sum() + eps
    
    mass_within = (saliency_map * class_mask).sum()
    act_prec = (mass_within / mass).item()

    mass_sens = class_mask.sum() + eps
    act_sens = (mass_within / mass_sens).item()
    
    return act_prec, act_sens


#------------------------------------
## compute the weighting game accuracy
def get_weighting_game_accuracy(dic_acts, a_path, ss=512):
    acc_scores, acc_score_list, saliency_map, dic_img_acc = {}, {}, {}, {}

    for name, act in tqdm(dic_acts.items()):
        scores = []
        s_map, s_img_acc = {}, {}
        ffnames = list(act.keys())

        for fname in ffnames: 
            mask = plt.imread(os.path.join(a_path, fname))
            saliency = cv2.resize(act[fname], dsize=(ss,ss), interpolation=cv2.INTER_CUBIC)
            saliency = np.where(saliency > 0, saliency, 0)
            saliency = scale_image(saliency, 1)
            accuracy = calculate_mass_within(saliency, mask)
            scores.append(round(accuracy, 4))
            s_map[fname] = saliency
            s_img_acc[fname] = accuracy
        
        acc_scores[name] = np.array(scores).mean()
        acc_score_list[name] = np.array(scores)
        dic_img_acc[name] = s_img_acc
        saliency_map[name] = s_map
            
    return acc_scores, acc_score_list, saliency_map, dic_img_acc   

#------------------------------------ False
def get_weighting_game_accuracy_v2(dic_acts, a_path, scale=True, thresh=1, size=(512,512)):
    
    acc_scores, acc_score_list, saliency_map, dic_img_acc = {}, {}, {}, {}

    for name, acts in tqdm(dic_acts.items()):
        scores = []
        s_img_acc, s_map = {}, {}
        fnames = list(acts.keys())

        for fname in fnames: 
            mask = plt.imread(os.path.join(a_path, fname))
            acti = cv2.resize(acts[fname], dsize=size, interpolation=cv2.INTER_CUBIC)
            sign_acti = acti.copy()
            
            if scale:
                sign_acti = signed_min_max_scale(sign_acti) # uniformaize activation in the same range 

            sign_acti = sign_acti > thresh

            act_pres = (mask * sign_acti).sum() / sign_acti.sum()
            scores.append(round(act_pres, 4))  
            s_img_acc[fname] = scores[-1] 
            s_map[fname] = sign_acti

        acc_scores[name] = np.array(scores).mean() 
        acc_score_list[name] = np.array(scores)
        saliency_map[name] = s_map
        dic_img_acc[name] = s_img_acc

    return acc_scores, acc_score_list, saliency_map, dic_img_acc

#------------------------------------
def get_weighting_game_accuracy_v4(dic_acts, a_path, r=4, ss=512):
    prec_sens_score, list_prec_sens_score, dic_saliency_map, dic_img_prec_sens  = {}, {}, {}, {}

    for name, act in tqdm(dic_acts.items()):
        tmp_prec, tmp_prec_sens = [], []
        img_prec, img_prec_sens, saliency_map = {}, {}, {}
        
        ffnames = list(act.keys())

        for fname in ffnames: 
            mask = plt.imread(os.path.join(a_path, fname))
            activation = cv2.resize(act[fname], dsize=(ss,ss), interpolation=cv2.INTER_CUBIC)
            saliency = np.where(activation > 0, activation, 0)
            saliency = scale_image(saliency, 1)
            accuracy = calculate_mass_within(saliency, mask)

            tmp_prec.append(round(accuracy[0], r))
            tmp_prec_sens.append(round(accuracy[1], r))

            img_prec[fname] = accuracy[0]
            img_prec_sens[fname] = accuracy[1]
            saliency_map[fname] = saliency
            
        prec_sens_score[name] = (round(np.array(tmp_prec).mean(), r), round(np.array(tmp_prec_sens).mean(), r))
        list_prec_sens_score[name] = (np.array(tmp_prec), np.array(tmp_prec_sens))
        dic_img_prec_sens[name] = (img_prec, img_prec_sens)
        dic_saliency_map[name] = saliency_map
        
    return prec_sens_score, list_prec_sens_score, dic_saliency_map, dic_img_prec_sens     

#------------------------------------
def plot_img_heatmap_v3(dat, fnames, dic_heatmap, size, fs=10, ncol=6): 
    '''
       plot image, heatmap, and heatmap overlay on the image
       imgs: [np_img, heatmap, overlay] => list of images
    '''
    fig = plt.figure(figsize=size, layout='constrained')
    nrow = len(fnames)
    j = 0
    
    for i, fname in enumerate(fnames):
        tmp = dat[dat.patientId==fname[:-4]]
        
        for ii, (key, act) in enumerate(dic_heatmap.items()):
            j += 1
            ax = fig.add_subplot(nrow, ncol, j)
            ax.imshow(act[fname], cmap='viridis')

            if i==0:
                ax.set_title(key, loc="center", fontsize=fs)
            
            for _, row in tmp.iterrows():
                rect = patches.Rectangle((row['x'], row['y']), row['width'], row['height'], linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                    
            ax.axis('off')




#### local to global sensitivity #####
######################################
#------------------------------------
def get_pointing_game_sensitivity_score(dic_act, dic_health, seed=0, bar=True, thresh=0.003):
    '''
        Description: Compute the pointing game metric from 2x2 grid where only one entry contains the disease

        Params:
            - dic_act (dic): key=filename, value=disease_activation
            - dic_health (dic): key=filename, value=healthy_activation
    '''
    numerator, denominator = 0, 0
    fname_healthy = list(dic_health.keys())

    iterator = tqdm(dic_act.items()) if bar else dic_act.items()
    
    for idx, (fname, acti) in enumerate(iterator): 
        #d_condition = acti > thresh
        saliency_map = np.where(acti > thresh, 1, 0)
            
        numerator += saliency_map.sum()  # d_condition
        denominator += saliency_map.sum()  # d_condition

        random.seed(idx) 
        sampled_values = random.sample(fname_healthy, 3)
        #print(sampled_values)
        
        for hfile in sampled_values:
            h_acti = dic_health[hfile]
            #h_condition = h_acti > thresh
            h_saliency_map = np.where(h_acti > thresh, 1, 0)
            denominator += h_saliency_map.sum()

    sens_score =  numerator / denominator
    
    return sens_score
    
#------------------------------------
def get_pointing_game(dic_act):
    pg_acts = {}
    for name, actt in dic_act.items(): 
        heathy_act, disease_act = actt
        pg_score = get_pointing_game_sensitivity_score(disease_act, heathy_act, bar=False)
        pg_acts[name] = pg_score
        print(f'{name}\t\t\t| {pg_score}')
    return pg_acts


#------------------------------------
def get_pointing_game_sensitivity_score_v2(cfg, network, d_df, h_df, path, act=True, conv=False):
    numerator, denominator = 0, 0
    
    for idx, row in tqdm(d_df.iterrows()): 
        ts_img, _ = load_image(cfg, img_path, row['filename'])
        ts_img = ts_img.to('cuda')
        neg_act = 0

        if conv:
            prediction, activation = conv_model(ts_img)
            activation = activation[0].detach().cpu().numpy()
            activation = activation[1]
            activation[activation < 0] = 0
            
        numerator += activation.sum()
        denominator += activation.sum()

        df_tmp = h_df.sample(3, random_state=idx)
        for _, row_ in  df_tmp.iterrows():
            ts_img_, _ = load_image(cfg, img_path, row_['filename'])
            ts_img_ = ts_img_.to('cuda')

            if conv:
                prediction, acti = conv_model(ts_img_)
                acti = acti[0].detach().cpu().numpy()
                action = acti[1]
                acti[acti < 0] = 0
                denominator += acti.sum()

    sens_score =  numerator / denominator
        #break
    return sens_score

#------------------------------------
def get_local_to_global_positive(dic_act, disease_imgs, thresh=0):
    '''
        proportion of positive activations from disease images
    '''
    print(f'Proportion of positive activations')
    for key, actt in dic_act.items():
        h_act, d_act = actt
        
        t_all_act = 0
        t_post_act = 0
        for fname in disease_imgs: 
            tmp_act = d_act[fname]
    
            # Boolean array (True or False)
            condition = tmp_act > thresh
    
            # Integer array (1 for True, 0 for False)
            integer_array = condition.astype(int)
    
            t_all_act += tmp_act.size
            t_post_act += condition.sum()
            
        prop = t_post_act / t_all_act
        
        if key in ['GuidedBackprop', 'GradCam']:
            print(f'{key} \t\t| {round(prop, 4)}')
        elif key in ['IntegratedGradients']:
            print(f'{key} \t| {round(prop, 4)}')
        else:
            print(f'{key} \t\t\t| {round(prop, 4)}')
        
    return prop

#------------------------------------
def get_local_to_global_positive_v2(dic_act, disease_imgs, thresh=0):
    '''
        proportion of positive activations from disease images
    '''
    print(f'Proportion of positive activations')
    for key, actt in dic_act.items():
        h_act, d_act = actt
        
        t_all_act = 0
        t_post_act = 0
        for fname in disease_imgs: 
            tmp_act = d_act[fname]
    
            # Boolean array (True or False)
            saliency_map = np.where(tmp_act > thresh, 1, 0)
    
            t_all_act += saliency_map.size
            t_post_act += saliency_map.sum()
            
        prop = t_post_act / t_all_act
        
        if key in ['GuidedBackprop', 'GradCam']:
            print(f'{key} \t\t| {round(prop, 4)}')
        elif key in ['IntegratedGradients']:
            print(f'{key} \t| {round(prop, 4)}')
        else:
            print(f'{key} \t\t\t| {round(prop, 4)}')
        
    return prop

#------------------------------------
def get_local_to_global_negative(dic_act, healthy_imgs, thresh=0):
    '''
        proportion of negative activations from healthy images
    '''
    print(f'Proportion of negative activations')
    for key, actt in dic_act.items():
        h_act, d_act = actt
        
        t_all_act = 0
        t_post_act = 0
        for fname in healthy_imgs: 
            tmp_act = h_act[fname]
    
            # Boolean array (True or False)
            condition = tmp_act < thresh
    
            # Integer array (1 for True, 0 for False)
            integer_array = condition.astype(int)
    
            t_all_act += tmp_act.size
            t_post_act += condition.sum()
            
        prop = t_post_act / t_all_act

        if key in ['GuidedBackprop', 'GradCam']:
            print(f'{key} \t\t| {round(prop, 4)}')
        elif key in ['IntegratedGradients']:
            print(f'{key} \t| {round(prop, 4)}')
        else:
            print(f'{key} \t\t\t| {round(prop, 4)}')
            
    return prop

#------------------------------------
def get_local_to_global_negative_v2(dic_act, healthy_imgs, thresh=0):
    '''
        proportion of negative activations from healthy images
    '''
    print(f'Proportion of negative activations')
    for key, actt in dic_act.items():
        h_act, d_act = actt
        
        t_all_act = 0
        t_post_act = 0
        for fname in healthy_imgs: 
            tmp_act = h_act[fname]
            
            saliency_map = np.where(tmp_act < thresh, 1, 0)
    
            t_all_act += saliency_map.size
            t_post_act += saliency_map.sum()
            
        prop = t_post_act / t_all_act

        if key in ['GuidedBackprop', 'GradCam']:
            print(f'{key} \t\t| {round(prop, 4)}')
        elif key in ['IntegratedGradients']:
            print(f'{key} \t| {round(prop, 4)}')
        else:
            print(f'{key} \t\t\t| {round(prop, 4)}')
            
    return prop


############ RSNA dataset ############
######################################
#------------------------------------
def save_xchest_masks(df, data, dser_dir, save_dir, s=512):
    '''
        save masks
    '''
    fnames = df.filename.tolist()
    
    for fname in tqdm(fnames):        
        tmp = data[data.patientId==fname[:-4]]
        mask = np.zeros((512, 512), dtype=np.uint8)
        
        for idx, row in tmp.iterrows():            
            top_left = (int(row['x']), int(row['y']))
            bottom_right = (int(row['x'] + row['width']), int(row['y'] + row['height']))
            cv2.rectangle(mask, top_left, bottom_right, 255, -1)
    
        cv2.imwrite(f'{save_dir}/{fname}', mask)

#------------------------------------
def save_dilated_img(a_path, d_path):
    files = os.listdir(a_path)

    kernel = np.ones((9,9), np.uint8)  # Define kernel explicitly
    mask_dilation = partial(cv2.dilate, kernel=kernel, iterations=1)

    for file in files:
        mask = plt.imread(os.path.join(a_path, file))
        image = (mask * 255).clip(0, 255).astype(np.uint8) ## Sample binary image
        dilated_mask = mask_dilation(image)  # Now correctly applies dilation

        cv2.imwrite(f'{d_path}/{file}', dilated_mask)

#------------------------------------
def plot_grid(df, ax):
     for idx, row in df.iterrows():
         rect = patches.Rectangle((row['x'], row['y']), row['width'], row['height'], linewidth=1, edgecolor='r', facecolor='none')
         ax.add_patch(rect)

def plot_grid2(ax, coord, ps=33):
     for (x, y) in coord:
         rect = patches.Rectangle((x, y), ps, ps, linewidth=1, edgecolor='r', facecolor='none')
         ax.add_patch(rect)

def scatter_plot(ax, xy, text=False):
    for i, (x, y) in enumerate(xy):
        ax.scatter(x,y, c='g')
        if text:
            ax.text(x+10,y, i)

#------------------------------------
def plot_test(all_df, df_precision, overall_prec, activation, img_path, save_dir, k=6, size=(8, 4)): 

    rr = random.sample(df_precision.filename.tolist(), k)
    aa = rr[0]
    #rr[:2]

    aaa = overall_prec[overall_prec.filename==aa]
    coords = aaa.coords.tolist()
    #aaa.head(2)

    tmp = all_df[all_df.patientId==aa[:-4]]
    
    n =2
    fig = plt.figure(figsize=size, layout='constrained')
    
    img = Image.open(os.path.join(img_path, aa)).convert('RGB').resize((512,512))
    img = np.asarray(img)
    
    bb = activation[aa]
    bb = cv2.resize(bb, dsize=(512,512), interpolation=cv2.INTER_CUBIC)
    overlay = get_overlay_img(img, bb)
    
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(img)
    plot_grid(tmp, ax)
    scatter_plot(ax, coords[:n])
    
    ax = fig.add_subplot(1, 3, 2)
    img = plt.imread(save_dir + '/' + aa)
    ax.imshow(img)
    plot_grid2(ax, coords[:n])
    scatter_plot(ax, coords[:n], text=True)
    
    ax = fig.add_subplot(1, 3, 3)
    ax.imshow(overlay)
    plot_grid(tmp, ax)
    scatter_plot(ax, coords[:n])
    
######################################

#------------------------------------
def get_activations_(file, path, dic_acti, s=512, xchest=False, attrib=[]):
    pil_img = Image.open(os.path.join(path, file)) 
    if xchest:
        pil_img = pil_img.resize((512, 512))
    
    if xchest:
        pil_img = pil_img.convert('RGB')

    np_img = np.array(pil_img) 

    acti = {file: np_img}
    
    for key, (_, d_act) in dic_acti.items():
        act = cv2.resize(d_act[file], dsize=(s,s), interpolation=cv2.INTER_CUBIC)
        if key in attrib:
            acti[key] = get_overlay_img_v2(np_img, act)
        else:
            #print(key)
            acti[key] = get_overlay_img(np_img, act)
        
    return acti 

#------------------------------------
def get_activations(file, path, dic_acti, s=512, v2=False, norm=True, xchest=False, alpha=None):
    pil_img = Image.open(os.path.join(path, file)) 

    alhp = alpha if alpha else 0.6
    
    if xchest:
        pil_img = pil_img.convert('RGB')
        pil_img = pil_img.resize((512,512))

    np_img = np.array(pil_img) 
    #print(np_img.shape)

    acti = {file: np_img}
    
    for idx, (key, h_d_act) in enumerate(dic_acti.items()):
        act = cv2.resize(h_d_act[file][1], dsize=(s,s), interpolation=cv2.INTER_CUBIC)
        if idx in [3,4]:
            if v2:
                acti[key] = get_overlay_img_v2(np_img, act, alpha=alhp, res_conv=norm)
            else:
                acti[key] = get_overlay_img(np_img, act, alpha=alhp, res_conv=norm) #
        else:            
            acti[key] = get_overlay_img(np_img, act)
        #print(idx, key)
        
    return acti, np_img

#------------------------------------
def get_activations_v2(files, path, dic_acti, s=512, xchest=False, attrib=[]):
    actis = {}
    for file in files:
        pil_img = Image.open(os.path.join(path, file)) 
        if xchest:
            pil_img = pil_img.resize((512, 512))
            pil_img = pil_img.convert('RGB')            
    
        np_img = np.array(pil_img) 
    
        acti = {file: np_img}
        
        for key, (_, d_act) in dic_acti.items():
            act = cv2.resize(d_act[file], dsize=(s,s), interpolation=cv2.INTER_CUBIC)
            #print('activation: ', act.shape)
            if key in attrib:
                acti[key] = get_overlay_img_v2(np_img, act)
            else:
                acti[key] = get_overlay_img(np_img, act)
        actis[file] = acti
        
    return actis 

#------------------------------------
def get_activations_v41(file, path, dic_acti, s=512, xchest=False, norm=True):
    pil_img = Image.open(os.path.join(path, file)) 
    
    if xchest:
        pil_img = pil_img.convert('RGB')
        pil_img = pil_img.resize((512,512))

    np_img = np.array(pil_img) 

    acti = {file: np_img}
    
    for idx, (key, h_d_act) in enumerate(dic_acti.items()):
        activation = cv2.resize(h_d_act[file][1], dsize=(s,s), interpolation=cv2.INTER_CUBIC)
        x_min, x_max = np.min(activation), np.max(activation)
        #print(key, x_min, x_max)

        if norm:            
            if x_max == x_min: # Avoid division by zero if all values are the same
                heatmap = np.zeros_like(activation)
            else:
                heatmap = 2 * (activation - x_min) / (x_max - x_min) - 1
        else: 
            heatmap = activation
            
        #heatmap = (activation - np.min(activation)) / (np.max(activation) - np.min(activation)) 

        acti[key] = heatmap
        
    return acti, np_img


#------------------------------------
def plot_img_heatmap(dic_imgs, size, s=30, fs=10, xchest=False, fname=None, a_path=None, dset='fundus', dat=None): 
    '''
       plot image, heatmap, and heatmap overlay on the image
       imgs: [np_img, heatmap, overlay] => list of images
    '''
    fig = plt.figure(figsize=size, layout='constrained')
    n = len(dic_imgs)
    j = 0
    
    for i, (key, img) in enumerate(dic_imgs.items()):
        j += 1
        ax = fig.add_subplot(1, n, j)
        img = ax.imshow(img, cmap='viridis')

        if (i==0) and (dset=='fundus'):
            X, Y = [], []
            mask = plt.imread(os.path.join(a_path, fname))
            for ii in range(512):
                for jj in range(512):
                    if mask[ii,jj] != 0:
                        X.append(ii)
                        Y.append(jj)
            ax.scatter(X, Y, c='r', s=s)
            
        if (i==0) and (dset=='oct'):
            r = 496/512
            df_ = a_path
            file = fname.split('.')[0].split('/')[-1]
            df_tmp = df_[df_.filename==file].reset_index(drop=True)

            for jj in range(len(df_tmp)):
                row = df_tmp.iloc[jj]
                ax.scatter(row['x']*r, row['y']*r, c='red', s=s)        
        
        if xchest:
            if i>0:
                #ax.set_title(key.split('/')[-1], loc="center", fontsize=fs)
                pass
            else:
                tmp = dat[dat.patientId==fname[:-4]]
                print('okkkk')
                
                for _, row in tmp.iterrows():
                    rect = patches.Rectangle((row['x'], row['y']), row['width'], row['height'], linewidth=1, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
        else:
            if i==0:
                pass
            else:
                ax.set_title(key.split('/')[-1], loc="center", fontsize=fs)
        ax.axis('off')

#------------------------------------
def plot_img_heatmap_v2(dic_imgs, size, fs=10, ncol=6, xchest=False, fname=None, dat=None): 
    '''
       plot image, heatmap, and heatmap overlay on the image
       imgs: [np_img, heatmap, overlay] => list of images
    '''
    fig = plt.figure(figsize=size, layout='constrained')
    nrow = len(dic_imgs)
    j = 0
    
    for i, (key, dic) in enumerate(dic_imgs.items()):
        
        for ii, (key2, img) in enumerate(dic.items()):
            j += 1
            ax = fig.add_subplot(nrow, ncol, j)
            img = ax.imshow(img, cmap='viridis')

            if xchest:
                if i==0:
                    if ii > 0:
                        ax.set_title(key2, loc="center", fontsize=fs)
                        
                tmp = dat[dat.patientId==key[:-4]]
                #if ii == 0:
                 #   ax.set_title(tmp, loc="center", fontsize=fs)
                
                for _, row in tmp.iterrows():
                    rect = patches.Rectangle((row['x'], row['y']), row['width'], row['height'], linewidth=1, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
            else:
                if (i==0) or (ii==0):
                    ax.set_title(key2.split('/')[-1], loc="center", fontsize=fs)
                    
            ax.axis('off')

#------------------------------------
def plot_img_heatmap41(dic_imgs, size, s=30, fs=10, xchest=False, fname=None, a_path=None, dset='fundus', fcmap='RdBu_r', vmax=None,
                       dic_bagnet_ftrs=None, dat=None, alpha=0.5, alpha2=0.8): 
    '''
       plot image, heatmap, and heatmap overlay on the image
       imgs: [np_img, heatmap, overlay] => list of images
    '''
    fig = plt.figure(figsize=size, layout='constrained')
    n = len(dic_imgs)
    params = dic_bagnet_ftrs
    j = 0
    
    for i, (key, img) in enumerate(dic_imgs.items()):
        
        if i==0: 
            j += 1
            ax = fig.add_subplot(1, n, j)
            ax.imshow(img, cmap='viridis')
            
            if dset=='fundus':
                X, Y = [], []
                mask = plt.imread(os.path.join(a_path, fname))
                for ii in range(512):
                    for jj in range(512):
                        if mask[ii,jj] != 0:
                            X.append(ii)
                            Y.append(jj)
                ax.scatter(X, Y, c='r', s=s)
                
            if dset=='oct':
                r = 496/512
                df_ = a_path
                file = fname.split('.')[0].split('/')[-1]
                df_tmp = df_[df_.filename==file].reset_index(drop=True)
    
                for jj in range(len(df_tmp)):
                    row = df_tmp.iloc[jj]
                    ax.scatter(row['x']*r, row['y']*r, c='red', s=s) 
            if dset=='rsna':
                tmp = dat[dat.patientId==fname[:-4]]
                
                for _, row in tmp.iterrows():
                    rect = patches.Rectangle((row['x'], row['y']), row['width'], row['height'], linewidth=1, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
        else:
            j += 1
            ax = fig.add_subplot(1, n, j)
            vmax_ = vmax if vmax else np.max(img)
            if params: 
                ax.imshow(params['overlay'], extent=params['extent'], cmap=params['cmap_original'], interpolation='none', alpha=alpha)
            else:
                ax.imshow(dic_imgs[fname], alpha=alpha)
            
            axx = ax.imshow(img, interpolation='bilinear', cmap=fcmap, alpha=alpha2, vmin=-vmax_, vmax=vmax_,)
            
            ax.set_title(key.split('/')[-1], loc="center", fontsize=fs)
        ax.axis('off')
        
    cbar = fig.colorbar(axx, ax=ax, location='right', shrink=0.85)
    
    cbar.ax.set_yticklabels([]) # Remove colorbar tick labels
    cbar.ax.tick_params(size=0)  # Set tick size to zero ~ Remove tick marks (horizontal bars)


######################################
#------------------------------------
def get_posthoc_explanation_v2(cfg, filenames, xai, f_path, s=60, target=1):
    dic_attrib = {name: {} for name in xai.keys()}
    
    for fname in tqdm(filenames):
        ts_img, np_img = load_image(cfg, f_path, fname)
        ts_img = ts_img.to('cuda')

        for xai_name, explainer in xai.items():  
            attrib = explainer.attribute(ts_img, target=target)
            attrib = attrib.squeeze().cpu().detach().numpy()
            attrib = np.transpose(attrib, (1,2,0))
            
            attrib_np = cv2.resize(np.sum(attrib, axis=2), dsize=(s,s), interpolation=cv2.INTER_CUBIC)
            dic_attrib[xai_name][fname] = attrib_np
        #break     
    return dic_attrib

#------------------------------------
def get_single_posthoc_explanation(ts_img_, np_img_, explainer, v2=True, s=512, n=5):
    dic_attrib = {}
    dic_overlay = {}
    for i in range(n):  
        attrib = explainer.attribute(ts_img_, target=i)
        attrib = attrib.squeeze().cpu().detach().numpy()
        attrib = np.transpose(attrib, (1,2,0))
        
        attrib = cv2.resize(np.sum(attrib, axis=2), dsize=(s,s), interpolation=cv2.INTER_CUBIC)
        dic_attrib[i]  = attrib    
        dic_overlay[i] = get_overlay_img_v2(np_img_, attrib, alpha=0.6, res_conv=False) if v2 else get_overlay_img(np_img_, attrib, alpha=0.6, res_conv=False)
    return dic_attrib, dic_overlay

#------------------------------------
def plot_img_multi_heat_att(img, title, dic_overlay, dic_acti, ypred, size, nrow=2, fs=10):
    fig = plt.figure(figsize=size, layout='constrained')
    n = len(dic_overlay) + 1
    j = 1

    ax = fig.add_subplot(nrow, n, j)
    img = ax.imshow(img, cmap='viridis')
    ax.set_title(title, loc="center", fontsize=fs)
    ax.axis('off')
    
    for i, img_ in dic_overlay.items():
        j += 1
        ax = fig.add_subplot(nrow, n, j)
        img = ax.imshow(img_)
        ax.set_title(ypred[i], loc="center", fontsize=fs)
        ax.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.03)
    plt.colorbar(img, cax=cax, use_gridspec=False)
    
    j += 1   
    
    for i, acti in dic_acti.items():
        j += 1
        ax = fig.add_subplot(nrow, n, j)
        img = ax.imshow(acti)
        ax.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.03)
    plt.colorbar(img, cax=cax, use_gridspec=False)
    
    #plt.show()

#------------------------------------
# oct
def plot_annotations(df, dser_dir, s=5, fs=12, nrow=1, ncol=6, size=(10,10)):
    fig = plt.figure(layout='constrained', figsize=size)
    plt.rcParams['font.size'] = f'{fs}'
    
    filenames = df.filename.unique().tolist()
    filenames = random.sample(filenames, ncol) 
    r = 496/512
    
    for idx, fname in enumerate(filenames):        
        tmp_df = df[df.filename==fname].reset_index(drop=True)
        
        img_path = os.path.join(dser_dir, f'{fname}.png')
        img = plt.imread(img_path)

        ax = fig.add_subplot(nrow, ncol, idx+1)
        ax.imshow(img)
        ax.set_title(fname)
    
        for j in range(len(tmp_df)):
            row = tmp_df.iloc[j]
            ax.scatter(row['x']*r, row['y']*r, c='red', s=s)
        ax.axis('off')


#------------------------------------
def get_oct_masks(df, dser_dir, save_dir, s=58):
    
    filenames = df.filename.unique()
    masks = {}
    r = 496/512
    
    for idx, fname in enumerate(filenames):
        
        tmp_df = df[df.filename==fname].reset_index(drop=True)
    
        img_path = os.path.join(dser_dir, f'{fname}.png')
        img = plt.imread(img_path)
        
        mask = np.zeros((496, 496))
        for j in range(len(tmp_df)):
            row = tmp_df.iloc[j]
            x, y = int(row['x']*r), int(row['y']*r)
            mask[x,y] = 1

        mask = np.asarray(mask.astype(int)) * 255
        masks[fname] = mask
        
        cv2.imwrite(f'{save_dir}/{fname}.png', mask)
    return masks


#------------------------------------

#------------------------------------
def get_prediction(network, img, act=False, verbose=True):
    activation = None
    if act:
        prediction, activation = network(img)
        activation = activation[0].detach().cpu().numpy()
        if verbose:
            print(activation.shape) # , att_weight.shape
    else:
        prediction = network(img)
    
    prediction = prediction.data.cpu()
    #att_weight = att_weight[0].detach().cpu().numpy()

    y_prob = torch.nn.functional.softmax(prediction, dim=1)
    y_prob = np.round(y_prob.numpy(), 3)
    y_class = np.argmax(y_prob, axis = 1)
    
    dist = Categorical(logits=prediction)
    pred, yhat = torch.topk(dist.probs, 1)

    if verbose:
        print('Logit distribution: \t ', y_prob[0])
        print(f'Probabilities: \t\t {round(pred.item(), 4)}, \t Class: {yhat.item()}')
    return y_prob, activation


def get_prediction_v2(network, img, act=False):
    if act:
        prediction, activation = network(img)
        activation = activation[0].detach().cpu().numpy()
    else:
        prediction = network(img)
    
    prediction = prediction.data.cpu()

    y_prob = torch.nn.functional.softmax(prediction, dim=1)
    y_prob = np.round(y_prob.numpy(), 3)
    y_class = np.argmax(y_prob, axis = 1)

    dist = Categorical(logits=prediction)
    pred, yhat = torch.topk(dist.probs, 1)
    pred = round(pred.item(), 4)
    return pred, yhat.item()

'''
#prediction, activation, _ = conv_model(ts_img)
prediction, activation = conv_model(ts_img)
prediction = prediction.data.cpu()
activation = activation[0].detach().cpu().numpy()
#att_weight = att_weight[0].detach().cpu().numpy()
print(activation.shape) # , att_weight.shape

dist = Categorical(logits=prediction)
pred, yhat = torch.topk(dist.probs, 1)
print('Logit distribution: \t ', prediction.numpy()[0])
print(f'Probabilities: \t\t {round(pred.item(), 4)}, \t Class: {yhat.item()}')
'''

######################################
######### Plot #######################
######################################

#------------------------------------
def plot_img_heat_att_v4(dic_overlay, imgs, img_names, anot_paths, size, ncol=8, fs=10, s=1, c='c', trans=True, txt_kwargs=None): 
    '''
       plot image, heatmap overlay on the image
       imgs: [np_img] => list of input images
       dic_overlay: {key: overlay}
    '''
    fig = plt.figure(figsize=size, facecolor= "white", dpi=200, layout='constrained')
    #plt.rcParams['font.size'] = f'{fs}'
    nrow = len(imgs)
    j, k, kk = 0, 0, 0

    for k, (dset, dic_map) in enumerate(dic_overlay.items()):
        
        for i, (key, img_) in enumerate(dic_map.items()):
            j += 1
            ax = fig.add_subplot(nrow, ncol, j)                
            ax.imshow(img_, cmap='viridis') 
    
            if k==0:
                if i > 0:
                    if 'Guided' in key:
                        ax.set_title('Guided BP', loc="center", **txt_kwargs)
                    elif 'Integra' in key:
                        ax.set_title('Itgd Grad. ', loc="center", **txt_kwargs)
                    else:
                        ax.set_title(key, loc="center", **txt_kwargs)

            if (dset=='fundus') and (i==0):
                X, Y = [], []
                mask = plt.imread(os.path.join(anot_paths[0], img_names[0]))

                for ii in range(512):
                    for jj in range(512):
                        if mask[ii,jj] != 0:
                            X.append(ii)
                            Y.append(jj)
                            
                ax.scatter(X, Y, c=c, s=s)

            if (dset=='oct') and (i==0):
                r = 496/512
                df_ = anot_paths[1]
                file = img_names[1].split('.')[0].split('/')[-1]
                df_tmp = df_[df_.filename==file].reset_index(drop=True)

                for jj in range(len(df_tmp)):
                    row = df_tmp.iloc[jj]
                    ax.scatter(row['x']*r, row['y']*r, c=c, s=s)

            if (dset=='rsna') and (i==0):
                dat = anot_paths[2]
                file = img_names[2][:-4]
                df_tmp = dat[dat.patientId==file]
                
                for _, row in df_tmp.iterrows():
                    rect = patches.Rectangle((row['x'], row['y']), row['width'], row['height'], linewidth=2, edgecolor=c, facecolor='none')
                    ax.add_patch(rect)
                
            ax.axis('off')

#------------------------------------
def plot_img_heat_att_v41(dic_overlay, imgs, img_names, anot_paths, size, ncol=8, fs=10, s=1, c='c', params=None, alpha=0.3, alpha2=0.8, lw=1.7,
                         ytitle=['', '', ''], vmax=None, fcmap='viridis', trans=True, txt_kwargs=None): 
    '''
       plot image, heatmap overlay on the image
       imgs: [np_img] => list of input images
       dic_overlay: {key: overlay}
    '''
    fig = plt.figure(figsize=size, facecolor= "white", dpi=200, layout='constrained')
    #plt.rcParams['font.size'] = f'{fs}'
    nrow = len(imgs)
    j, k, kk = 0, 0, 0

    for k, (dset, dic_map) in enumerate(dic_overlay.items()):
        
        for i, (key, img_) in enumerate(dic_map.items()):

            if i==0:
                j += 1
                ax = fig.add_subplot(nrow, ncol, j)                
                ax.imshow(img_, cmap='viridis') 
                ax.set_ylabel(ytitle[k], **txt_kwargs)
                ax.set_yticks([])
                ax.set_xticks([])
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                
                if dset=='fundus':
                    X, Y = [], []
                    mask = plt.imread(os.path.join(anot_paths[0], img_names[0]))
    
                    for ii in range(512):
                        for jj in range(512):
                            if mask[ii,jj] != 0:
                                X.append(ii)
                                Y.append(jj)                                
                    ax.scatter(X, Y, c=c, s=s)
                    #pass
                elif dset=='oct':
                    r = 496/512
                    df_ = anot_paths[1]
                    file = img_names[1].split('.')[0].split('/')[-1]
                    df_tmp = df_[df_.filename==file].reset_index(drop=True)
    
                    for jj in range(len(df_tmp)):
                        row = df_tmp.iloc[jj]
                        ax.scatter(row['x']*r, row['y']*r, c=c, s=s)
                    # pass
                else:
                    dat = anot_paths[2]
                    file = img_names[2][:-4]
                    df_tmp = dat[dat.patientId==file]
                    
                    for _, row in df_tmp.iterrows():
                        rect = patches.Rectangle((row['x'], row['y']), row['width'], row['height'], linewidth=lw, edgecolor=c, facecolor='none')
                        ax.add_patch(rect)
            else:
                j += 1
                ax = fig.add_subplot(nrow, ncol, j) 

                vmax_ = vmax if vmax else np.max(img_)
                
                if dset=='fundus':
                    #ax.imshow(params['overlay'], extent=params['extent'], cmap=params['cmap_original'], interpolation='none', alpha=alpha)
                    ax.imshow(dic_map[img_names[k]], alpha=alpha)
                else:
                    ax.imshow(dic_map[img_names[k]], alpha=alpha)
                
                axx = ax.imshow(img_, interpolation='bilinear', cmap=fcmap, alpha=alpha2, vmin=-vmax_, vmax=vmax_,)
                ax.axis('off')
                
            if k==0:
                if i > 0:
                    if 'Guided' in key:
                        ax.set_title('Guided BP', loc="center", **txt_kwargs)
                    elif 'Integra' in key:
                        ax.set_title('Itgd Grad. ', loc="center", **txt_kwargs)
                    else:
                        ax.set_title(key, loc="center", **txt_kwargs)
                
            
        cbar = fig.colorbar(axx, ax=ax, location='right', shrink=0.85)
        cbar.ax.set_yticklabels([]) # Remove colorbar tick labels
        cbar.ax.tick_params(size=0)  # Set tick size to zero ~ Remove tick marks (horizontal bars)


#------------------------------------
def get_inference_all_cam(fpath, keys, reg, dset):
    with open(fpath, 'rb') as file:  
        save_all_inference = pickle.load(file)
    cres_all = save_all_inference[f'{dset}_cres_all_act']           # (2, 16, 16)
    h_d_attrib = save_all_inference[f'{dset}_h_d_attrib_dic']       # (60,60), (60,60)

    all_conv_cam = cres_all | h_d_attrib
    #print(all_conv_cam.keys())
    
    for key in keys:
        del all_conv_cam[key]
    
    all_conv_cam = move_dict_element(all_conv_cam, '0', 6)
    all_conv_cam = move_dict_element(all_conv_cam, reg, 6)

    key_mapping = {'0': 'dense SoftCAM', reg: 'sparse SoftCAM'}
    all_conv_cam = rename_keys(all_conv_cam, key_mapping)
    return all_conv_cam

#------------------------------------
def rename_keys(d, key_mapping):
    """
    Rename keys in a dictionary based on a given mapping.

    :param d: Original dictionary
    :param key_mapping: Dictionary mapping old keys to new keys
    :return: New dictionary with renamed keys
    """
    return {key_mapping.get(k, k): v for k, v in d.items()}

#------------------------------------
def move_dict_element(d, key_to_move, new_position):
    items = list(d.items())
    item = items.pop([k for k, v in items].index(key_to_move))
    items.insert(new_position, item)
    return dict(items)

#------------------------------------
def get_precision_sensitivity_val(fpath, keys, reg, dset):
    with open(os.path.join(fpath, f'{dset}_all_precision.pkl'), 'rb') as file:  
        save = pickle.load(file)

    with open(os.path.join(fpath, f'{dset}_all_sensitivity.pkl'), 'rb') as file:  
        save2 = pickle.load(file)

    dic_topk_prec = save[f'dic_topk_prec_{dset}']
    dic_overall_prec = save[f'dic_overall_prec_{dset}']

    dic_all_sensitivity = save2[f'all_sensitivity_performance_{dset}']

    #print(dic_topk_prec.keys())
    for key in keys:
        del dic_topk_prec[key]
        del dic_all_sensitivity[key]
    
    dic_topk_prec = move_dict_element(dic_topk_prec, '0', 6)
    dic_topk_prec = move_dict_element(dic_topk_prec, reg, 6)

    dic_all_sensitivity = move_dict_element(dic_all_sensitivity, '0', 6)
    dic_all_sensitivity = move_dict_element(dic_all_sensitivity, reg, 6)

    key_mapping = {'0': 'dense SoftCAM', reg: 'sparse SoftCAM'}
    dic_topk_prec = rename_keys(dic_topk_prec, key_mapping)
    dic_all_sensitivity = rename_keys(dic_all_sensitivity, key_mapping)

    all_topk_tp_prec = {key: val[1] for key, val in dic_topk_prec.items()}
    
    return all_topk_tp_prec, dic_all_sensitivity

#------------------------------------
def plot_img_heatmap_v33(dat, fnames, fpath, dic_heatmap, size, fs=10, ncol=8): 
    '''
       plot image, heatmap, and heatmap overlay on the image
       imgs: [np_img, heatmap, overlay] => list of images
    '''
    fig = plt.figure(figsize=size, layout='constrained')
    nrow = len(fnames)
    j = 0
    
    for i, fname in enumerate(fnames):
        tmp = dat[dat.patientId==fname[:-4]]

        j += 1
        ax = fig.add_subplot(nrow, ncol, j)
        img = Image.open(os.path.join(fpath, fname)) 
        img = np.array(img.resize((512, 512)))
        ax.imshow(img)

        for _, row in tmp.iterrows():
            rect = patches.Rectangle((row['x'], row['y']), row['width'], row['height'], linewidth=1.5, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        ax.axis('off')
        
        for ii, (key, act) in enumerate(dic_heatmap.items()):
            j += 1
            ax = fig.add_subplot(nrow, ncol, j)
            ax.imshow(act[fname], cmap='viridis')

            if i==0:
                ax.set_title(key, loc="center", fontsize=fs)
            
            for _, row in tmp.iterrows():
                rect = patches.Rectangle((row['x'], row['y']), row['width'], row['height'], linewidth=1.5, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                    
            ax.axis('off')

#------------------------------------
def plot_img_heatmap_v333(dat, fnames, fpath, dic_heatmaps, size, fs=10, ncol=8, kwargs=None): 
    '''
       plot image, heatmap, and heatmap overlay on the image
       imgs: [np_img, heatmap, overlay] => list of images
    '''
    fig = plt.figure(figsize=size, layout='constrained')
    nrow = len(fnames) * 2
    j = 0
    
    for i, fname in enumerate(fnames):
        tmp = dat[dat.patientId==fname[:-4]]

        j += 1
        ax = fig.add_subplot(nrow, ncol, j)
        img = Image.open(os.path.join(fpath, fname)) 
        img = np.array(img.resize((512, 512)))
        ax.imshow(img)

        for _, row in tmp.iterrows():
            rect = patches.Rectangle((row['x'], row['y']), row['width'], row['height'], linewidth=1.5, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        ax.axis('off')

        for idx, (model, dic_heatmap_) in enumerate(dic_heatmaps.items()): 
            dic_heatmap, dic_img_acc = dic_heatmap_[0], dic_heatmap_[1]
            k = 0
            
            if idx!= 0:
                j += 1
                
            for ii, (key, act) in enumerate(dic_heatmap.items()):
                j += 1
                ax = fig.add_subplot(nrow, ncol, j)
                #print(key, fname)
                score = round(dic_img_acc[key][fname], 3)
                ax.imshow(act[fname], cmap='viridis')
                ax.text(390, 50, score, color='white', backgroundcolor='black', fontsize=fs, weight='bold')
    
                if idx==0:
                    ax.set_title(key, loc="center", **kwargs)
                    
                for _, row in tmp.iterrows():
                    rect = patches.Rectangle((row['x'], row['y']), row['width'], row['height'], linewidth=1.5, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)

                if k%8 == 0:
                    ax.set_ylabel(model, fontsize=kwargs['fontsize'])
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:
                    ax.axis('off')
                k += 1

#------------------------------------
def plot_img_heatmap_v444(dat, fnames, fpath, dic_heatmaps, size, lw=1.5, fs=10, ncol=8, cmap=None, kwargs=None): 
    '''
       plot image, heatmap, and heatmap overlay on the image
       imgs: [np_img, heatmap, overlay] => list of images
    '''
    fig = plt.figure(figsize=size, layout='constrained')
    nrow = len(fnames) * 2
    j = 0
    cmapp = cmap if cmap else 'viridis'
    
    for i, fname in enumerate(fnames):
        tmp = dat[dat.patientId==fname[:-4]]

        j += 1
        ax = fig.add_subplot(nrow, ncol, j)
        img = Image.open(os.path.join(fpath, fname)) 
        img = np.array(img.resize((512, 512)))
        ax.imshow(img, cmap='gray')

        for _, row in tmp.iterrows():
            rect = patches.Rectangle((row['x'], row['y']), row['width'], row['height'], linewidth=lw, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        ax.axis('off')

        for idx, (model, dic_heatmap_) in enumerate(dic_heatmaps.items()): 
            dic_heatmap, dic_img_acc = dic_heatmap_[0], dic_heatmap_[1]
            k = 0
            
            if idx!= 0:
                j += 1
                
            for ii, (key, act) in enumerate(dic_heatmap.items()):
                j += 1
                ax = fig.add_subplot(nrow, ncol, j)
                #print(key, fname)
                score = round(dic_img_acc[key][0][fname], 3)
                score_sens = round(dic_img_acc[key][1][fname], 3)
                axx = ax.imshow(act[fname], cmap=cmapp)
                ax.text(60, 60, score, color='white', backgroundcolor='black', fontsize=fs, weight='bold')
                ax.text(340, 60, score_sens, color='white', backgroundcolor='black', fontsize=fs, weight='bold')
    
                if (i==0) and (idx==0):
                    if 'Guided' in key:
                        ax.set_title('Guided BP', loc="center", **kwargs)
                    elif 'Integra' in key:
                        ax.set_title('Itgd Grad. ', loc="center", **kwargs)
                    else:
                        ax.set_title(key, loc="center", **kwargs)
                    #ax.set_title(key, loc="center", **kwargs)
                    
                for _, row in tmp.iterrows():
                    rect = patches.Rectangle((row['x'], row['y']), row['width'], row['height'], linewidth=lw, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)

                if k%8 == 0:
                    ax.set_ylabel(model, **kwargs) # fontsize=kwargs['fontsize']
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:
                    ax.axis('off')
                k += 1

            cbar = fig.colorbar(axx, ax=ax, location='right', shrink=0.85)
            cbar.ax.set_yticklabels([]) # Remove colorbar tick labels
            cbar.ax.tick_params(size=0)
#------------------------------------
def plot_precision_with_sensitivity(dic_val, bb=30, fs=12, ms=1, lw=0.7, lfs=8, nrow=2, ncol=3, model=None, size=(16,3), txt_kwargs=None):
    '''
        description: plot topk precision curve
    '''
    fig, ax = plt.subplots(nrow, ncol, figsize=size, facecolor= "white", dpi=200, layout="constrained")
    plt.rcParams['font.size'] = f'{fs}'
    txt = 'abcdefghijk'

    xaxis = list(range(1, bb))
    ax = ax.flatten()
    k = 0
    
    yrange = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] 
    for idx, (dset, (dic_prec, _)) in enumerate(dic_val.items()): 
        for key, val in dic_prec.items():
            if 'Guided' in key:
                title = 'Guided BP'
            elif 'Int' in key:
                title = 'Itgt. Grad.'
            else:
                title = key
            ax[k].plot(xaxis, val, label=title, marker='o', markersize=ms, linewidth=lw)

        if k==0:
            ax[k].set_ylabel('Precision',  **txt_kwargs)
            ax[k].set_xlabel('Top-K patches')
            ax[k].legend(fontsize=lfs, framealpha=0) # loc="upper right"

        ax[k].set_title(f'{txt[k]}.   {dset}', loc='left',  **txt_kwargs)
        ax[k].spines['right'].set_visible(False)
        ax[k].spines['top'].set_visible(False)
        
        if dset == 'CXR':
            ax[k].set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
        elif dset=='OCT':
            ax[k].set_yticks([0.1, 0.3, 0.5, 0.7, 0.9])
        else:
            ax[k].set_yticks(yrange)
        k += 1 
        
    k = 3
    xaxis = range(0, bb)
    yrange = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for idx, (dset, (_, dic_sens)) in enumerate(dic_val.items()): 
        kk = k+idx
        
        for key, val in dic_sens.items():
            ax[kk].plot(xaxis, val, marker='o', markersize=ms, label=key, linewidth=lw)
            
        ax[kk].spines['right'].set_visible(False)
        ax[kk].spines['top'].set_visible(False)
        if dset=='CXR':
            if model=='vgg':
                ax[kk].set_yticks([0.7, 0.8, 0.9, 1.0])
            else:
                ax[kk].set_yticks([0.85, 0.9, 0.95, 1.0])
        elif dset=='Fundus':
            ax[kk].set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        else:
            ax[kk].set_yticks(yrange)            
        
        if idx==0:
            ax[kk].set_ylabel('Sensitivity', **txt_kwargs) # Average accuracy drop
            ax[kk].set_xlabel('Number of occluded patches')
            ax[kk].legend(fontsize=lfs, framealpha=0)

#--------------------------
def plot_precision_with_sensitivity_v2(dic_vals, bb=30, fs=12, lfs=8, nrow=2, ncol=3, model=None, size=(16,3), txt_kwargs=None):
    '''
        description: plot topk precision curve
    '''
    fig, ax = plt.subplots(nrow, ncol, figsize=size, facecolor= "white", dpi=200, layout="constrained")
    plt.rcParams['font.size'] = f'{fs}'
    txt = 'abcdefghijk'
    
    ax = ax.flatten()
     
    for idx_, (m_key, dic_val) in enumerate(dic_vals.items()):
        kkk = idx_
        k = idx_ * 6
        xaxis = list(range(1, bb+1))
        yrange = [0.2, 0.4, 0.6, 0.8]
        #print(xaxis)
        for idx, (dset, (dic_prec, _)) in enumerate(dic_val.items()): 
            for key, val in dic_prec.items():
                #if kkk==1:
                #    xaxis = list(range(1, bb))
                ax[k].plot(xaxis, val[:bb], label=key, linewidth=0.7)
                #print(k, len(xaxis), len(val))
    
            if idx==0:
                ax[k].set_ylabel(f'Precision: {m_key}', **txt_kwargs)
                ax[k].set_xlabel('Top-k patches', **txt_kwargs)
                
            if k == 0:
                ax[k].legend(fontsize=lfs) # loc="upper right"
                
            if idx_==0:
                ax[k].set_title(f'{txt[k]}.   {dset}', loc='left', **txt_kwargs)
                
            ax[k].spines['right'].set_visible(False)
            ax[k].spines['top'].set_visible(False)
            ax[k].set_xticks(range(0, bb+1, 4))
            
            if dset == 'CXR':
                ax[k].set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
            elif dset=='OCT':
                ax[k].set_yticks([0.1, 0.3, 0.5, 0.7, 0.9])
            else:
                ax[k].set_yticks(yrange)
            k += 1 
            
        #print('k before: ', k)  
        #k = kkk + 3
        xaxis = range(0, bb)
        yrange = [0.2, 0.4, 0.6, 0.8, 1.0]
        
        for idx, (dset, (_, dic_sens)) in enumerate(dic_val.items()): 
            kk = k+idx
            
            for key, val in dic_sens.items():
                ax[kk].plot(xaxis, val[:bb], marker='o', markersize=1, label=key, linewidth=0.7)
                
            ax[kk].spines['right'].set_visible(False)
            ax[kk].spines['top'].set_visible(False)
            ax[kk].set_xticks(range(0, bb+1, 4))
            
            if dset=='CXR':
                if idx_==1:
                    ax[kk].set_yticks([0.7, 0.8, 0.9, 1.0])
                else:
                    #ax[kk].set_yticks([0.7, 0.8, 0.9, 1.0])
                    ax[kk].set_yticks([0.9, 1.0])
            elif dset=='Fundus':
                ax[kk].set_yticks([0.4, 0.6, 0.8, 1.0])
            else:
                ax[kk].set_yticks(yrange)   
            
            if idx==0:
                ax[kk].set_ylabel(f'Sensitivity: {m_key}', **txt_kwargs) # Average accuracy drop
                ax[kk].set_xlabel('Top-k occluded patches', **txt_kwargs)
                
            if kk==3:
                ax[kk].legend(fontsize=lfs)
                
            #print('tmp ', (idx, idx_, kk))
            if kk < 6:
                ax[kk].set_title(f'{txt[kk]}.   {dset}', loc='left',  **txt_kwargs)
        #print('final k, kk', (k, kk))
        k = kk +1 


## ------------------------------------
def plot_precision_with_sensitivity_v22(dic_vals, bb=30, ts=12, lw=0.7, fs=12, lfs=8, ms=1, nrow=2, ncol=3, model=None, size=(16,3), txt_kwargs=None):
    '''
        description: plot topk precision curve
    '''
    fig, axs = plt.subplots(nrow, ncol, figsize=size, facecolor= "white", dpi=200, layout="constrained")
    #plt.tick_params(axis='both', labelsize=ts)
    plt.rcParams['font.size'] = f'{fs}'
    txt = 'abcdefghijk'

    xtick = range(0, bb+1, 5)
    yticks_sens = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    yranges_prec = {'Fundus': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], 'OCT': [0.1, 0.3, 0.5, 0.7, 0.9], 'RSNA': [0.5, 0.6, 0.7, 0.8, 0.9] }

    axs = axs.flatten()
    xaxis_prec = list(range(1, bb)) # for precision ~ top [1:n]
    xaxis_sens = range(0, bb)
    j=-1
     
    for idx, (model, dset_prec_sens) in enumerate(dic_vals.items()):
        
        for idxx, (dset, prec_sens) in enumerate(dset_prec_sens.items()):
            j += 1
            ax = axs[j]
            ax2 = axs[j+ncol]
            yticks = yranges_prec[dset]    
            #print(model, dset)
                 
            for cam_method, precision in prec_sens[0].items():
                sensitivity = prec_sens[1][cam_method]
                if 'Guided' in cam_method:
                    title = 'Guided BP'
                elif 'Integrat' in cam_method:
                    title = 'Itgd Grad.'
                else:
                    title = cam_method
                ax.plot(xaxis_prec, precision[:bb], label=title, linewidth=lw, marker='o', markersize=ms)
                ax2.plot(xaxis_sens, sensitivity[:bb], label=title, linewidth=lw, marker='o', markersize=ms)
                

            if j==0:
                ax.set_xlabel('Top-k patches', **txt_kwargs)
                ax2.set_xlabel('Top-k occluded patches', **txt_kwargs)
                #ax2.legend(fontsize=lfs, framealpha=0)
            if j==1: #0
                ax.legend(fontsize=lfs, framealpha=0, ncol=2)
                ax2.legend(fontsize=lfs, framealpha=0)
            if j<ncol:
                ax.set_title(f'{txt[j]}.   {dset}', loc='left', **txt_kwargs)
            if (j%2)==0:
                ax.set_ylabel(f'Precision: {model}', **txt_kwargs)
                ax2.set_ylabel(f'Sensitivity: {model}', **txt_kwargs)

            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['top'].set_visible(False)
            
            ax.set_xticks(xtick); ax.set_yticks(yticks)
            ax2.set_xticks(xtick); #ax2.set_yticks(yticks_sens)

#---------------------
def plot_precision_with_sensitivity_v222(dic_vals, bb=30, ts=12, lw=0.7, fs=12, lfs=8, ms=1, nrow=2, ncol=3, model=None, size=(16,3), txt_kwargs=None):
    '''
        description: plot topk precision curve
    '''
    fig, axs = plt.subplots(nrow, ncol, figsize=size, facecolor= "white", dpi=200, layout="constrained")
    #plt.tick_params(axis='both', labelsize=ts)
    plt.rcParams['font.size'] = f'{fs}'
    txt = 'abcdefghijk'

    xtick = range(0, bb+1, 5)
    yticks_sens = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    yranges = {'Fundus': [0.82, 0.85, 0.88, 0.92, 0.96, 1.0], 'OCT': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}

    axs = axs.flatten()
    xaxis_sens = range(0, bb)
    j=-1
     
    for idx, (model, dset_prec_sens) in enumerate(dic_vals.items()):
        
        for idxx, (dset, prec_sens) in enumerate(dset_prec_sens.items()):
            j += 1
            ax = axs[j]
            yticks = yranges[dset]    
            #print(model, dset)
                 
            for cam_method, sensitivity in prec_sens.items():
                score = [1] + list(sensitivity)
                #print(len(score), len(xaxis_sens))
                #print(score)
                #print(xaxis_sens)
                ax.plot(xaxis_sens, score, label=cam_method, linewidth=lw, marker='o', markersize=ms)               

            if j in [0]:
                ax.set_xlabel('Top-k occluded patches', **txt_kwargs)
                ax.legend(fontsize=lfs, framealpha=0, ncol=1)
                
            if j<ncol:
                ax.set_title(f'{txt[j]}.   {dset}', loc='left', **txt_kwargs)
            if (j%2)==0:
                ax.set_ylabel(f'Sensitivity: {model}', **txt_kwargs)

            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            
            ax.set_xticks(xtick) ; ax.set_yticks(yticks)

#---------------------
def get_local_to_global_positive_41(dic_act, disease_imgs, task=None, i=0, r=3, thresh=0, s=16):
    '''
        proportion of positive activations from disease images
    '''
    print(f'Proportion of {task} activations')
    prop_dic, dic_img_score = {}, {}
    for key, actt in dic_act.items():
        #h_act = cv2.resize(actt[0], dsize=(s,s), interpolation=cv2.INTER_CUBIC)

        img_prop = []
        img_score = {}
        for fname in disease_imgs: 
            
            tmp_act = cv2.resize(actt[fname][i], dsize=(s,s), interpolation=cv2.INTER_CUBIC)
            d_h_act = tmp_act / np.max(np.abs(tmp_act))
            #d_h_act_round = np.rint(d_h_act)
    
            # Boolean array (True or False)
            condition = (d_h_act > thresh) if i==1 else (d_h_act < thresh)           
            local_score = condition.sum() / condition.size
            img_prop.append(local_score)
            img_score[fname] = local_score
            
        mean = round(np.mean(img_prop), r)
        std = round(np.std(img_prop), r)
        prop_dic[key] = (mean, std)
        dic_img_score[key] = img_score
        
        if key in ['GuidedBackprop', 'GradCam']:
            print(f'{key} \t\t\t| {mean, std}')
        elif key in ['IntegratedGradients']:
            print(f'{key} \t\t| {mean, std}')
        else:
            print(f'{key} \t\t\t| {mean, std}')
        
    return prop_dic, dic_img_score

#-------------------
def plot_val_performances_v2(dic_metrics, points, met='AUC', fs=16, lfs=10, ymin=0.7, s=8, nrow=2, ncol=2, kwargs=None, size=(11,8)):
    
    fig, axs = plt.subplots(nrow, ncol, figsize=size, facecolor= "white", dpi=200, layout="constrained")
    plt.rcParams['font.size'] = f'{fs}'
    txt = 'abcdefghijk'
    axs = axs.flatten()

    k = -1
    for idx, (model, reg_metrics) in enumerate(dic_metrics.items()):
        for idx_, (reg_name, metrics) in enumerate(reg_metrics.items()):
            k += 1 
            ii = points[k]
            ax1 = axs[k]
            val_acc = [ll[0] for ll in metrics]
            val_auc = [ll[1] for ll in metrics]
            reg = [ll[2] for ll in metrics]

            if k in [0,2]:
                ax1.plot(reg, val_auc, 'b-', label=f'{met}') 
                ax1.plot(reg, val_acc, 'g-', label='Accuracy')
            else:
                ax1.plot(reg, val_auc, 'b-') 
                ax1.plot(reg, val_acc, 'g-')
                
            ax1.scatter(reg, val_auc, color = 'b', s=s)
            ax1.scatter(reg[ii], val_auc[ii], color = 'r', s=s+3)
 
            ax1.scatter(reg, val_acc, color = 'g', s=s, zorder=1)
            ax1.scatter(reg[ii], val_acc[ii], color = 'r', s=s+3, zorder=2, label=f'$\\lambda$ = {reg[ii]}')

            if idx_ == 0:
                ax1.set_xlabel('Regularization values $(\\lambda_1)$')
                ax1.set_ylabel(f'{model}', **kwargs)
            else:
                ax1.set_xlabel('Regularization values $(\\lambda_2)$')

            ax1.legend(fontsize=lfs)

            if idx==0:
                ax1.set_title(f'{txt[k]}.   {reg_name}', loc='left', **kwargs)                
                
            ax1.grid()
            if model=='VGG':
                ax1.set_ylim([0.86, 1.0])
            if k in [3]:
                xticks = list(np.round(np.linspace(0, max(reg), 7), 3))
                ax1.set_xticks(xticks)

            if k in [0, 2]:
                xticks = list(np.round(np.linspace(0, max(reg), 6), 4))
                #print(xticks)
                ax1.set_xticks(xticks)
            if k in [1]:
                yticks = list(np.round(np.linspace(0.88, 1, 6), 2))
                ax1.set_yticks(yticks)