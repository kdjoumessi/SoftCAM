import torch


def topK_patches(np_heatmap, size=60, dx=-2, dy=2, p=4, k=5, max_=None, threshold=None, round_=True):
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
    
    values = values.int().tolist() if round_ else values.tolist()
    idx    = idx.int().tolist()
    dic_index = {}
    dic_val = {}
    
    top_patch_idx = []
    top_patch_val = []
    
    i = 0
    overlapping_idx = []
    while (len(top_patch_idx) < k) and (values[i] > threshold):
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
        
        if len(top_patch_idx) == max_:
            break    
    return top_patch_idx, top_patch_val, dic_index, dic_val


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
        
        if len(top_patch_idx) == max_:
            break    
    return top_patch_idx, top_patch_val, dic_index, dic_val



def get_patch_location(indices, s=1, size=512):    
    '''
        convert patch indices from the low resolution (60x60 idealy) to the high resolution (512x512 idealy) 
        NB: without patches
    '''
    idx = []
    
    for i in indices:
        row_idx = (i // size)     # row_idx = (i // 24)  # 512 = grid size
        start_row = row_idx * s   # start_row = row_idx * 8  # 1 = stride
        
        col_idx = i % size        # col_idx = (i - (24 * row_idx))
        start_col = col_idx * s   # start_col = col_idx * 8 
        
        idx.append((start_row, start_col))
    return idx