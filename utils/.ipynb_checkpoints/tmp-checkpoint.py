from utils.fully_convnet_tools import get_patch_location, topK_patches_v2


def get_prediction_v2(network, img, act=False):
    if act:
        prediction, activation = network(img)
        activation = activation[0].detach().cpu().numpy()
    else:
        prediction = network(img)
    #print(prediction)
    prediction = prediction.data.cpu()

    y_prob = torch.nn.functional.softmax(prediction, dim=1)
    y_prob = np.round(y_prob.numpy(), 5)
    y_class = np.argmax(y_prob, axis = 1)
    return y_prob[0][1], y_class[0]


def topK_patches_v2(np_heatmap, size=60, dx=-2, dy=2, p=4, k=5, max_=None):
    '''
        - one patch of size 33x33 can overlap with 3 other patches in each direction 
        input:
            - np_heatmap (np array)
            - k (int): possible number of selected patches
            - dx, dy (int, int): how to select patches
            - max_ (int): max_ number of patches
            - threshold (int)
    '''
    scores = torch.from_numpy(np_heatmap.flatten())
    values, idx = torch.topk(scores, len(scores))
    
    values = values.tolist()
    idx    = idx.int().tolist()
    dic_index = {}
    dic_val = {}
    
    top_patch_idx = []
    top_patch_val = []
    
    i = 0
    overlapping_idx = []
    while (len(top_patch_idx) < k):
        #print(i, len(top_patch_idx))
        val = values[i]
        index = idx[i]
        #print(f'index {i}, idx = {index}')
        if index not in overlapping_idx:
            top_patch_idx.append(index)
            top_patch_val.append(values[i])
        
        # how to manage extrem cases where after adding dc or dy the final index is out of bounds ?
        row_idx = (index // size) + dx
        col_idx = (index % size) + dy
        coord = (row_idx, col_idx)
        
        coords = [(coord[0]+i, coord[1]-j) for i in range(p) for j in range(p)] # increase x index and decrease y index
        #print('coords: \n', coords)
        tmp_1D_idx_from_coord = [(elt[0] * size) + elt[1] for elt in coords]
        dic_index[index] = tmp_1D_idx_from_coord
        dic_val[index] = val
        
        overlapping_idx += tmp_1D_idx_from_coord
        i += 1        
        dic_index['overlap'] = overlapping_idx
        
        if (i == max_) | (len(top_patch_idx) == k): # len(top_patch_idx)
            break    
    return top_patch_idx, top_patch_val, dic_index, dic_val



def get_topk_sensitivity(cfg, network, dic_act, img_path, cname='conv_res_conf', act=False, k=30, ps=33, s=8, act_size=60, 
                         posthoc=None):
    n = act_size * act_size
    avg_drop = []
    dic_result, dic_img, loc_coords, min_max = {}, {}, {}, {}

    acts = dic_act[posthoc] if posthoc else dic_act

    for fname, heatmap in acts.items():
        flat_heatmap = torch.from_numpy(heatmap.flatten())
        max_, min_ = heatmap.max(), heatmap.min()

        top_indices, top_scores, _, _ = topK_patches_v2(flat_heatmap.numpy(), k=30, dx=-2, dy=2, max_=1000) # 1000
        loc_coord = get_patch_location(top_indices, s=s, size=act_size)
        loc_coords[fname] = loc_coord
        min_max[fname] = (round(min_, 2), round(max_, 2))
    
    for i in tqdm(range(1, k)):
        fnames, coords, preds, yhats, rdrop, y_preds, hats = [], [], [], [], [], [], []
        dic_img_ = {}
        
        for fname, loc_coordi in loc_coords.items():
            coord = []
            loc_coord = loc_coordi[:i]

            ts_img, np_img = load_image(cfg, img_path, fname)
            np_img = np.transpose(np_img, (2, 0, 1))
            np_img = np.expand_dims(np_img, axis=0) 
            ts_img = ts_img.to('cuda')
            
            y_pred, hat = get_prediction_v2(network, ts_img, act=act)
            
            for j, (y,x) in enumerate(loc_coord):                
                ts_img[0, :, y:y+ps, x:x+ps] = 0
                np_img[0, :, y:y+ps, x:x+ps] = 0
                coord.append((y,x))
                #print('test coord: ', loc_coord)
        
            pred, yhat = get_prediction_v2(network, ts_img, act=act)  
            #print('rdrop : ', pred/y_pred)

            rdrop.append(round(pred/y_pred, 4)); preds.append(pred); yhats.append(yhat)
            fnames.append(fname); y_preds.append(y_pred); hats.append(hat); coords.append(coord)

            dic_img_[fname] = (np_img, ts_img.detach().cpu())
            
        
        result = pd.DataFrame({'filename':fnames, 'coords':coords, 'occ_pred':preds, 'occ_yhat':yhats, 
                               'y_pred':y_preds, 'y_hat':hats, 'rdrop': rdrop})
        
        dic_result[i] = result
        dic_img[i] = dic_img_
        avg_drop.append(round(result.rdrop.mean(), 4))

    return dic_result, avg_drop, dic_img, loc_coords

####
dic_result, avg_drop, dic_img, dic_coord = get_topk_sensitivity(cfg, conv_model, activation_, img_path, cname='conv_res_conf', 
                                                      act=True, k=30, act_size=60, posthoc=None)

ll = [len(ii) for ii in dic_coord.values()]
#print(ll)