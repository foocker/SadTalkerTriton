import os
import numpy as np
import torch


def get_facerender_data(coeffs_pred_dict, crop_pic, first_coeff_dict,
                        batch_size, 
                        expression_scale=1.0, save_dir=None, save_flag=False):

    semantic_radius = 13
    data={}

    source_image = np.array(crop_pic).transpose((2, 0, 1)) / 255.0  #
    # np.set_printoptions(precision=5) 
    source_image = np.round(source_image, decimals=5)
    source_image_ts = torch.FloatTensor(source_image).unsqueeze(0) 
    source_image_ts = source_image_ts.repeat(batch_size, 1, 1, 1)
    data['source_image'] = source_image_ts
    
    # full
    source_semantics = first_coeff_dict['coeff_3dmm'][:1,:73]         #crop is [:1,:70] 
    generated_3dmm = coeffs_pred_dict['coeff_3dmm'][:,:70]

    source_semantics_new = transform_semantic_1(source_semantics, semantic_radius)
    source_semantics_ts = torch.FloatTensor(source_semantics_new).unsqueeze(0)
    source_semantics_ts = source_semantics_ts.repeat(batch_size, 1, 1)
    data['source_semantics'] = source_semantics_ts

    # target 
    generated_3dmm[:, :64] = generated_3dmm[:, :64] * expression_scale

    # TODO full is 73, crop is 70 
    generated_3dmm = np.concatenate([generated_3dmm, np.repeat(source_semantics[:,70:], generated_3dmm.shape[0], axis=0)], axis=1)
    
    # if still_mode:  # 直接默认为真
    generated_3dmm[:, 64:] = np.repeat(source_semantics[:, 64:], generated_3dmm.shape[0], axis=0)
        
    if save_dir is not None and save_flag:
        os.makedirs(save_dir, exist_ok=True)
        with open( os.path.join(save_dir, 'generated_3dmm.txt'), 'w') as f:
            for coeff in generated_3dmm:
                for i in coeff:
                    f.write(str(i)[:7]   + '  '+'\t')
                f.write('\n')

    target_semantics_list = [] 
    frame_num = generated_3dmm.shape[0]
    data['frame_num'] = frame_num
    for frame_idx in range(frame_num):
        target_semantics = transform_semantic_target(generated_3dmm, frame_idx, semantic_radius)
        target_semantics_list.append(target_semantics)

    remainder = frame_num%batch_size
    if remainder!=0:
        for _ in range(batch_size-remainder):
            target_semantics_list.append(target_semantics)

    target_semantics_np = np.array(target_semantics_list)             #frame_num 70 semantic_radius*2+1
    target_semantics_np = target_semantics_np.reshape(batch_size, -1, target_semantics_np.shape[-2], target_semantics_np.shape[-1])
    data['target_semantics_list'] = torch.FloatTensor(target_semantics_np)
    
    return data

def transform_semantic_1(semantic, semantic_radius):
    semantic_list =  [semantic for i in range(0, semantic_radius*2+1)]
    coeff_3dmm = np.concatenate(semantic_list, 0)
    return coeff_3dmm.transpose(1,0)

def transform_semantic_target(coeff_3dmm, frame_index, semantic_radius):
    num_frames = coeff_3dmm.shape[0]
    seq = list(range(frame_index- semantic_radius, frame_index + semantic_radius+1))
    index = [ min(max(item, 0), num_frames-1) for item in seq ] 
    coeff_3dmm_g = coeff_3dmm[index, :]
    return coeff_3dmm_g.transpose(1,0)
    
