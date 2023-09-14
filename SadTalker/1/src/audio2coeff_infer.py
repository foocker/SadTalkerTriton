import os 
import torch
import numpy as np
from yacs.config import CfgNode as CN
from scipy.signal import savgol_filter
from src.audio2pose_models.audio2pose import Audio2Pose

import onnxruntime
from tqdm import tqdm


class Audio2Coeff():

    def __init__(self, device):
        #load config
        fcfg_pose = open("/models/SadTalker/1/src/config/auido2pose.yaml")
        cfg_pose = CN.load_cfg(fcfg_pose)
        cfg_pose.freeze()
        self.device = device

        # load audio2pose_model
        self.audio2pose_model = Audio2Pose(cfg_pose, None, device=device)
        self.audio2pose_model = self.audio2pose_model.to(device)
        self.audio2pose_model.eval()
        
        self.audio2pose_model.load_state_dict(torch.load("/models/SadTalker/1/onnx_weights/audio2pose_simple.pth"))
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

        audio2exp = onnxruntime.InferenceSession('/models/SadTalker/1/onnx_weights/audio2exp_simple.onnx', sess_options, \
            providers=[('TensorrtExecutionProvider', {'trt_fp16_enable': True, 'trt_engine_cache_enable': True, 
                                                      'trt_engine_cache_path': '/models/SadTalker/1/cache'}), 'CUDAExecutionProvider'])
        
        self.audio2exp = audio2exp

        
    def audio2exp_onnx(self, batch, step=10):
        mel_input = batch['indiv_mels'].type(torch.FloatTensor).data.numpy()
        ref_input = batch['ref'].type(torch.FloatTensor).data.numpy()
        ratio_gt = batch['ratio_gt'].type(torch.FloatTensor).data.numpy()
        T = mel_input.shape[1]

        exp_coeff_pred = []
        for i in tqdm(range(0, T, step),'audio2exp:'): # every 10 frames
            current_mel_input = mel_input[:,i:i+step]
            audiox = current_mel_input.reshape(-1, 1, 80, 16)                  # bs*T 1 80 16
            ref = ref_input[:, :, :64][:, i:i+step]
            ratio = ratio_gt[:, i:i+step]                               #bs T
            ort_inputs = {"x":audiox, "ref":ref, "ratio":ratio}
            curr_exp_coeff_pred = self.audio2exp.run(None, ort_inputs)[0]
            exp_coeff_pred += [curr_exp_coeff_pred]
            
        # BS x T x 64
        results_dict = {
            'exp_coeff_pred': np.concatenate(exp_coeff_pred, axis=1)
            }
        return results_dict

    def audio2pose_onnx(self):
        pass

    def generate(self, batch, pose_style, coeff_save_dir=None, save_flag=False):
        results_dict_exp = self.audio2exp_onnx(batch, step=20)  
        exp_pred = torch.from_numpy(results_dict_exp['exp_coeff_pred']).to(self.device)

        with torch.no_grad():
            batch['class'] = torch.LongTensor([pose_style]).to(self.device)
            results_dict_pose = self.audio2pose_model.test(batch)
            pose_pred = results_dict_pose['pose_pred']                   # bs T 6

            pose_len = pose_pred.shape[1]
            if pose_len<13:
                pose_len = int((pose_len-1)/2)*2+1
                pose_pred = torch.Tensor(savgol_filter(np.array(pose_pred.cpu()), pose_len, 2, axis=1)).to(self.device)
            else:
                pose_pred = torch.Tensor(savgol_filter(np.array(pose_pred.cpu()), 13, 2, axis=1)).to(self.device) 
            
            coeffs_pred = torch.cat((exp_pred, pose_pred), dim=-1)            #bs T 70

            coeffs_pred_numpy = coeffs_pred[0].clone().detach().cpu().numpy()
            
            if coeff_save_dir is not None and save_flag:
                from scipy.io import savemat
                os.makedirs(coeff_save_dir, exist_ok=True)
                savemat(os.path.join(coeff_save_dir, '%s##%s.mat'%(batch['pic_name'], batch['audio_name'])),
                        {'coeff_3dmm': coeffs_pred_numpy})
                
            return {'coeff_3dmm': coeffs_pred_numpy}
    


