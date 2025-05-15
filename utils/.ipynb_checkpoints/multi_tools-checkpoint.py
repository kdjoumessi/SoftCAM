
import cv2
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from .tools import *
from .basic_tools import load_image
import matplotlib.colors as mcolors
from skimage import feature, transform

from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget



#------------------------------------
def get_multi_overlay_img(img, activations, alpha = 0.6, res_conv=True):
    '''
        alpha: Transparency factor
        only normalize for conv evidence map
    '''
    dic_act = {}
    for i, activation in activations.items():
        # Normalize the heatmap data to the range [0, 1]
        heatmap = (activation - np.min(activation)) / (np.max(activation) - np.min(activation)) if res_conv else activation
        
        # Apply a colormap to the heatmap
        heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Blend the image and heatmap
        overlay_image = cv2.addWeighted(img, 1 - alpha, cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB), alpha, 0)
        
        dic_act[i] = overlay_image
        
    return dic_act

#------------------------------------
def plot_multi_img_heat_att(imgs, prob, title, size, fs=10, act=True, vmax=100, alpha=0.6): 
    '''
       plot image, heatmap, and heatmap overlay on the image
       imgs: [np_img, heatmap, overlay] => list of images
    '''
    fig = plt.figure(figsize=size, layout='constrained')
    overlay_imgs = imgs[1] if act else imgs[2]
    
    n = len(imgs) + len(overlay_imgs) -1
    
    j = 1
    input_img = imgs[0]
    ax = fig.add_subplot(1, n, j)
    ax.imshow(input_img)
    ax.set_title(title['label'], loc="center", fontsize=fs)
    ax.axis('off')

    for ii, tmp_img in overlay_imgs.items():
        j += 1
        ax = fig.add_subplot(1, n, j)
        if act:
            img = ax.imshow(input_img, vmin=-vmax, vmax=vmax) # cmap='RdBu_r'
            ax.imshow(tmp_img, vmin=-vmax, vmax=vmax, alpha=alpha)
        else:
            img = ax.imshow(tmp_img, vmin=-vmax, vmax=vmax) # cmap='viridis',
            
        ax.set_title(f'{title[ii]}: {round(prob[ii].item(), 3)}', loc="center", fontsize=fs)
        ax.axis('off')
    if act:
        cbar = fig.colorbar(img, ax=ax, location='right', shrink=0.85)

#------------------------------------
def get_multi_pred_with_heatmap(cfg, model, df, f_path, res=False, cname='conv_res', s=60, att=True, bar=True):
    ''' 
       description: run inference on each image and output predictions and corresponding activation maps
    '''    
    ypred, yhat = [], []
    all_activation = {}
    
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
            acti = acti.detach().cpu().numpy()
        else:
            pred, acti = model(ts_img)
            acti = acti.detach().cpu().numpy()
            
        pred = pred.data.cpu()

        y_prob = torch.nn.functional.softmax(pred, dim=1)
        y_class = np.argmax(y_prob, axis = 1)
        
        ypred.append(np.max(y_prob.numpy()[0]))
        yhat.append(y_class.item())

        if not res:
            acti = acti[0]
            acti_ = []
            for ii in range(len(acti)):
                act = cv2.resize(acti[ii], dsize=(s,s), interpolation=cv2.INTER_CUBIC)
                acti_.append(act)
            all_activation[fname] = np.stack(acti_)
    
    dat[f'{cname}_pred'] = yhat
    dat[f'{cname}_conf'] =  np.round(np.array(ypred), 3)
    
    return dat, all_activation

#------------------------------------
def get_multi_posthoc_explanation(cfg, model, filenames, xai, f_path, s=16, nclass=2):
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
        ts_img = ts_img.to('cuda')

        for xai_name, explainer in xai.items():  
            tmp_cam = []
            
            if xai_name == 'ScoreCAM': # torchcam' ['ScoreCAM', 'LayerCAM']:
                with torch.no_grad():
                    output = model(ts_img)
                    
                for i in range(nclass):
                    cam_map = explainer(i)[0]
                    cam_map = cam_map.squeeze().cpu().numpy()
                    tmp_cam.append(cv2.resize(cam_map, dsize=(s,s), interpolation=cv2.INTER_CUBIC))
                    
            elif xai_name == 'LayerCAM':
                output = model(ts_img)

                for i in range(nclass):                        
                    targets = [ClassifierOutputTarget(i)]
                    # Generate heatmap using ScoreCAM
                    grayscale_cam = explainer(input_tensor=ts_img, targets=targets)

                    # Convert the grayscale cam output (first entry) to a 2D heatmap
                    grayscale_cam = grayscale_cam[0]
                    tmp_cam.append(cv2.resize(grayscale_cam, dsize=(s,s), interpolation=cv2.INTER_CUBIC))
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
                        
                    tmp_cam.append(attrib)

            dic_attrib[xai_name][fname] = np.stack(tmp_cam) 
    
    return dic_attrib

#------------------------------------
def get_multi_non_overlap_patches_scores(activations, path, bar=True, size=(512, 512)):
    '''
        description: get non-overlapping patches from each image
    '''
    img_patch = {}

    iterate = tqdm(activations.items()) if bar else activations.items()
    
    for fname, heat in iterate:
        patch_class = {}
        
        for ii in range(len(heat)):
            heat_ = heat[ii]
            heat_ = cv2.resize(heat_, dsize=size, interpolation=cv2.INTER_CUBIC)
            patch_class[ii] = extract_patch(fname, heat_, path, size=size)
        
        img_patch[fname] = patch_class
        
    return img_patch

#------------------------------------
def plot_multi_deletion_curve(dic_perform, bb=16, marker='o', r=3, label='Deletion Curve', ytitle='', pos=1, out=False, ls=1, ms=2, fs=12, lfs=8, size=(11, 8)):
    '''
        description: area under the deletion curve
    '''
    fig, ax = plt.subplots(figsize=size, facecolor= "white", dpi=200, layout="constrained")
    plt.rcParams['font.size'] = f'{fs-2}'
    
    x = range(1, bb)
    print(len(x), list(x))
    xtitle = 'Number of patch Deleted'

    for i, (name, score) in enumerate(dic_perform.items()):
        #print(len(score))
        #score = np.insert(score_, 0, 1)  
        del_fraction = np.linspace(0, 1, bb-1)
        auc_deletion = np.trapz(score, x=del_fraction)
        
        print(f'AUC-Del. {name}: {round(auc_deletion, r)}')    
        plt.plot(x, score, marker=marker, label=name, markersize=ms, linewidth=ls)
        
    ax.set_xlabel(xtitle)
    ax.set_ylabel('Sensitivity Analysis: Deletion Curve')
    ax.set_title(ytitle) # 'Sensitivity Analysis: Deletion Curve'
    #ax.set_xlim(1, 15)

    if out:
        ax.legend(loc='upper left', bbox_to_anchor=(pos, 1), borderaxespad=0., fontsize=lfs)
    else:
        #ax.legend(loc="upper right", fontsize=lfs)
        ax.legend(fontsize=lfs)

#------------------------------------
def plot_multi_deletion_curve_v2(dic_perform, bb=16, marker='o', label='Deletion Curve', ytitle='', pos=1, out=False, ls=1, ms=2, fs=12, lfs=8, size=(11, 8)):
    '''
        description: area under the deletion curve
    '''
    fig, ax = plt.subplots(figsize=size, facecolor= "white", dpi=200, layout="constrained")
    plt.rcParams['font.size'] = f'{fs-2}'
    
    x = range(bb)
    print(len(x), list(x))
    xtitle = 'Number of patch Deleted'

    for i, (name, score_) in enumerate(dic_perform.items()):
        #print(len(score))
        score = np.insert(score_, 0, 1)  
        del_fraction = np.linspace(0, 1, bb)
        auc_deletion = np.trapz(score, x=del_fraction)
        
        print(f'AUC-Del. {name}: {round(auc_deletion, 4)}')    
        plt.plot(x, score, marker=marker, label=name, markersize=ms, linewidth=ls)
        
    ax.set_xlabel(xtitle)
    ax.set_ylabel(ytitle)
    ax.set_title('Sensitivity Analysis: Deletion Curve')
    #ax.set_xlim(1, 15)

    if out:
        ax.legend(loc='upper left', bbox_to_anchor=(pos, 1), borderaxespad=0., fontsize=lfs)
    else:
        #ax.legend(loc="upper right", fontsize=lfs)
        ax.legend(fontsize=lfs)

#------------------------------------
def get_multi_deletion_analysis(cfg, network, patch_score, df_data, img_path, k=15, kk=4, ps=33, act=False):
    '''
        description: Simulate a deletion curve

        params:
            - network: torch model
            - patch_score (dic): key = image name, val = list [(i,j) score]
    ''' 
    perform = [1]   
    class_sensitivity = {}
    #dic_result, dic_img = {}, {}

    for ii in tqdm(range(1, kk)): # loop over classes
        dic_result = {}
        sens_fnames = df_data[df_data.level==ii]['filename'].tolist()  ## class selection
        #print(len(sens_fnames))
        #print('fname: \n', sens_fnames)
        class_patch_scores = {key: patch_score[key][ii] for key in sens_fnames} # retain only images correctly predicted as class ii

        for i in range(1,k): # loop over topk patches
            files, occ_pred, occ_hat, preds, hats, label, num, drop, relative_drop = [], [], [], [], [], [], [], [], []
        
            for fname in class_patch_scores.keys():            
                ts_img, np_img = load_image(cfg, img_path, fname)
                np_img = np.transpose(np_img, (2, 0, 1))
                np_img = np.expand_dims(np_img, axis=0) 
                ts_img = ts_img.to('cuda')
    
                pred, hat = get_prediction_v2(network, ts_img, act=act)
                loc_scores = class_patch_scores[fname][:i]
    
                for j, (xy, score) in enumerate(loc_scores): 
                    x, y = xy
                    ts_img[0, :, x:x+ps, y:y+ps] = 0
    
                pred_, hat_ = get_prediction_v2(network, ts_img, act=act) 
                pred__ = min(pred, pred_)
                rel_drop = pred__ / pred

                files.append(fname); occ_pred.append(pred_); occ_hat.append(hat_); drop.append(pred-pred_)
                preds.append(pred); hats.append(hat); label.append(1); num.append(j+1); relative_drop.append(rel_drop)
            
            result = pd.DataFrame({'filename':files, 'label':label, 'preds':preds, 'hats':hats, 'occ_pred':occ_pred, 'occ_hat':occ_hat, 
                                   'conf_drop': drop, 'relative_drop': relative_drop, 'num_patches': num})
        
            dic_result[i] = result
        
        class_sensitivity[ii] = dic_result
        
    return class_sensitivity

#------------------------------------
def multi_run_eval(cfg, df, model_surf, pref, model_p, img_path, anot_p, thresh=1, s=60, k=30, fundus=True):
    overall_prec = []
    topk_prec = {}
    
    for i, surf in tqdm(enumerate(model_surf)):
        cmodel = torch.load(os.path.join(model_p, surf + pref), weights_only=False)
        cmodel = cmodel.to('cuda')
        cmodel.eval()
        print(i)
        
        _, all_act = get_multi_pred_with_heatmap(cfg, cmodel, df, img_path, s=s, cname='', res=False, att=False, bar=False)        

        if fundus:
            acti = {fname: np.mean(act_[1:3], axis=0) for fname, act_ in all_act.items()}
        else:
            acti = {fname: act_[1] for fname, act_ in all_act.items()}

        pred_ = [1]*len(df)
        _, score = get_precision(acti, anot_p, pred_, thresh=thresh, bar=False, xchest=False, act_size=s)
        overall_prec.append(score)

        _, topk_conv_scores, topk_conv_scores_tp = get_topk_precision(acti, pred_, anot_p, thresh=thresh, k=k, bar=False, xchest=False, act_size=s)
        topk_prec[surf] = (topk_conv_scores, topk_conv_scores_tp)
        #break
    return overall_prec, topk_prec
    

# ---------- Multi plot --------
#------------------------------------
def get_multi_dict(multi_tmp_modal, models, fpath, ffname, label, mapping=False):
    reg = models[-1]
    tmp_img = plt.imread(fpath)
    #print(models)
    dic_tmp = {key: multi_tmp_modal[key][ffname] for key in models}

    if mapping:
        key_mapping = {'0': 'dense SoftCAM', reg: 'sparse SoftCAM'}
        dic_tmp = rename_keys(dic_tmp, key_mapping)
        edges = get_bagnet_heatmap(tmp_img, dic_tmp['dense SoftCAM'])
    else:
        edges = get_bagnet_heatmap(tmp_img, dic_tmp[models[0]])
    
    out = [tmp_img, label, edges]

    return out, dic_tmp
    
#------------------------------------
def get_cmap():
    # Colormap with transparency
    transp = np.concatenate( 
                (plt.cm.binary(np.linspace(0.99, 1, 32))[:, :-1], 
                 np.linspace(0, 0.01, 32).reshape(-1, 1) ), 
                axis=1)
    
    # Individual colormaps 
    clr_blues = plt.cm.Blues(np.linspace(0.5, 1, 96)) # Greens
    clr_red = plt.cm.Reds(np.linspace(0.5, 1, 96)) # plt.cm.viridis, Reds
    
    # Individual colormaps with transparency
    clrs_bl_t = np.vstack((transp, clr_blues))[::-1]
    clrs_rd_t = np.vstack((transp, clr_red))
    
    # Combine
    clrs_bl_t_rd = np.vstack((clrs_bl_t, clrs_rd_t))
    
    # Complete colormaps
    map_bl_t_rd = mcolors.LinearSegmentedColormap.from_list('map_bl_t_rd', clrs_bl_t_rd)
    return map_bl_t_rd

#------------------------------------
def get_single_overlay(img, activation, alpha = 0.6, res_conv=True):
    '''
        alpha: Transparency factor
        only normalize for conv evidence map
    '''
    # Normalize the heatmap data to the range [0, 1]
    heatmap = (activation - np.min(activation)) / (np.max(activation) - np.min(activation)) if res_conv else activation
        
    # Apply a colormap to the heatmap
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Ensure input image is uint8
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)

    # Blend the image and heatmap
    overlay_image = cv2.addWeighted(img, 1 - alpha, cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB), alpha, 0)
    
    return overlay_image

#------------------------------------
def get_single_overlay_v2(img, activation, alpha = 0.6, res_conv=True):
    heatmap = (activation - np.min(activation)) / (np.max(activation) - np.min(activation)) if res_conv else activation
    heatmap = plt.cm.hot(heatmap)  
    heatmap = np.delete(heatmap, 3, 2)         # Remove alpha channel
    
    # overlay the heatmap on the original image using alpha blending
    overlay_image = heatmap[..., :3] * alpha + img / 255.0 * (1 - alpha)
    overlay_image = np.clip(overlay_image, 0, 1)
    return overlay_image

#------------------------------------
def plot_multi_class_visualization(img, dics, prob, labels, size, vmax=None, oct=False, nrow=3, ncol=6, fs=10, overlay=False, custom_map=True, cmap=None, alpha=0.6, alpha2=0.8, s=512, kwargs=None): 
    '''
       plot image, heatmap, and heatmap overlay on the image
       imgs: [np_img, heatmap, overlay] => list of images
    '''
    fig = plt.figure(figsize=size, layout='constrained')
    
    j = 1
    ax = fig.add_subplot(nrow, ncol, j)
    ax.imshow(img[0])
    ax.set_title(f'Label: {img[1]}', loc="center", **kwargs)
    ax.axis('off')

    name = list(dics.keys())[-1]
    fcmap = cmap if custom_map else 'RdBu_r'
    ncol, idxs = (5, [6, 11]) if oct else (ncol, [7, 13])       
    #print(ncol, idxs)
        
    for idx, (key, acts) in enumerate(dics.items()):
        if key not in ['dense SoftCAM', 'sparse SoftCAM']:
            vmax_ = max(abs(acts.max()), abs(acts.min()))
        else:
            #vmax_ = max(abs(dics[name].max()), abs(dics[name].min()))
            #vmax = max(abs(dics['sparse SoftCAM'].max()), abs(dics['sparse SoftCAM'].min()))
            vmax_ = dics['sparse SoftCAM'].max()
            vmax_ = vmax if vmax else vmax_ 
        #vmax = vmax if vmax else vmax_
        
        for ii in range(len(acts)):
            j += 1
            
            if j in idxs:
                j += 1
            
            ax = fig.add_subplot(nrow, ncol, j)                
            act = cv2.resize(acts[ii], dsize=(s,s), interpolation=cv2.INTER_CUBIC)

            if overlay: 
                img_overlay = get_single_overlay(np.array(img[0]), act)
                axx = ax.imshow(img_overlay, vmin=-vmax, vmax=vmax, cmap=fcmap) # ,  interpolation='bilinear', cmap='viridis'
            else: 
                 parms = img[2]
                 #ax.imshow(img[0], alpha=alpha) # vmin=-vmax, vmax=vmax  
                 if oct:
                    ax.imshow(img[0], alpha=alpha)
                 else:
                     ax.imshow(parms['overlay'], extent=parms['extent'], cmap=parms['cmap_original'], interpolation='none', alpha=alpha) 
                 axx = ax.imshow(act, vmin=-vmax_, vmax=vmax_, alpha=alpha2, interpolation='bilinear', cmap=fcmap)

            if idx==0:
                ax.set_title(labels[ii], loc="center", **kwargs)

            if ii==0:
                ax.set_ylabel(key, **kwargs)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines['left'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
            else:
                ax.axis('off')
                
        cbar = fig.colorbar(axx, ax=ax, location='right', shrink=0.85)

#------------------------------------
def get_bagnet_feature(np_img, np_act, thresh=0.5, ed=5):   
    
    edges = get_bagnet_heatmap(np_img, np_act, ii=None, thresh=thresh, ed=ed)

    return edges

#------------------------------------
def get_bagnet_heatmap(img, np_logits, ii=2, dx=0.05, dy=0.05, dilation=0.5, percentile=99, thresh=0.5, ed=5):
    '''
        description: compute the high-resolution heatmap
        
        parameters:
            np_img (list): list of input numpy images
            ld_fmap (np.array): feature maps from the forward pass
    ''' 
    img_params = {}
    np_logit = np_logits[ii] if ii else np_logits
    
    xx = np.arange(0.0, img.shape[1], dx)
    yy = np.arange(0.0, img.shape[0], dy)
    xmin, xmax, ymin, ymax = np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)
    extent = xmin, xmax, ymin, ymax
    cmap_original = plt.get_cmap('Greys_r').copy()
    cmap_original.set_bad(alpha=0)
    overlay = None
    
    original_greyscale = np.mean(img, axis=-1)
    in_image_upscaled = transform.rescale(original_greyscale, dilation, mode='constant', channel_axis=False, anti_aliasing=True)
    edges = feature.canny(in_image_upscaled).astype(float)
    edges[edges < thresh] = np.nan
    edges[:ed, :] = np.nan
    edges[-ed:, :] = np.nan
    edges[:, :ed] = np.nan
    edges[:, -ed:] = np.nan
    overlay = edges

    abs_max = np.percentile(np.abs(np_logits), percentile)
    abs_min = np.percentile(np.abs(np_logits), 1)
    img_params = {'overlay': overlay, 'extent': extent, 'cmap_original': cmap_original, 'vmax': abs_max, 'vmin': abs_min} 
        
    return img_params

#------------------------------------
def get_activation_precision_v2(acts, a_path, scale=True, n=1, thresh=1, size=(512,512)):
    '''
        ToDo
    '''

    scores = []
    dic_acti, dic_sign_acti, dic_mask, dic_img, dic_score = {}, {}, {}, {}, {}

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
        
    return np.array(scores)

