import numpy as np
import cv2, os, torch
from PIL import Image 
from src.face3d.util.preprocess import align_img
from src.face3d.util.load_mats import load_lm3d
from scipy.io import  savemat
from src.utils.croper import Preprocesser, landmark_98_to_68
import onnxruntime


def split_coeff(coeffs):
    """
    Return:
        coeffs_dict     -- a dict of torch.tensors

    Parameters:
        coeffs          -- torch.tensor, size (B, 256)
    """
    id_coeffs = coeffs[:, :80]
    exp_coeffs = coeffs[:, 80: 144]
    tex_coeffs = coeffs[:, 144: 224]
    angles = coeffs[:, 224: 227]
    gammas = coeffs[:, 227: 254]
    translations = coeffs[:, 254:]
    return {
        'id': id_coeffs,
        'exp': exp_coeffs,
        'tex': tex_coeffs,
        'angle': angles,
        'gamma': gammas,
        'trans': translations
    }

def resize_by_level(frame, levels):
    ''' write by yh'''
    assert len(levels) == 2, "levels list must equal 2 level"
    levels.sort(reverse=True) 
    h, w, _ = frame.shape
    max_hw = max(h, w)
    if max_hw < levels[-1]:
        return frame
    if max_hw >= levels[0]:
        level = levels[0]
    else:
        level = levels[1]
        
    radio_level = max_hw / level
    h = int(h/radio_level) 
    w = int(w/radio_level)
    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
    print("Shape after resize:{}".format(frame.shape))
    return frame

def resize_by_scale(frame, target_size):
    h, w = frame.shape[:2]
    w_scale = target_size[0] / w
    h_scale = target_size[1] / h
    scale = min(w_scale, h_scale)

    width = int(w * scale)
    height = int(h * scale)

    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA) 
    
    return resized


class CropAndExtract():
    def __init__(self, device):

        self.propress = Preprocesser() # 检测，关键点提取，对齐，crop
        self.lm3d_std = load_lm3d("/models/SadTalker/1/src/config")
        self.device = device
        
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

        face_recon = onnxruntime.InferenceSession('/models/SadTalker/1/onnx_weights/face_recon_wresnet50_simple.onnx', sess_options, \
            providers=[('TensorrtExecutionProvider', {'trt_fp16_enable': True, 'trt_engine_cache_enable': True, 
                                                      'trt_engine_cache_path': '/models/SadTalker/1/cache'}), 'CUDAExecutionProvider'])
        self.face_recon = face_recon
    
    def generate(self, full_frame, pic_size=256, save_flag=False, save_dir="/models/SadTalker/1/results", debug=False):
        # 传入RGB格式 cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if save_flag:
            landmarks_path =  os.path.join(save_dir, 'input_croped_landmarks.txt')
            coeff_path =  os.path.join(save_dir, 'input_croped_aligned_3dmm.mat')
            png_path =  os.path.join(save_dir, 'input_croped.png')

        # 检测，关键点，crop，对齐。 full_frame:原始图的对齐resize
        full_frame, crop, quad = self.propress.crop(full_frame, xsize=512)
        clx, cly, crx, cry = crop
        lx, ly, rx, ry = quad
        lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
        oy1, oy2, ox1, ox2 = cly+ly, cly+ry, clx+lx, clx+rx
        crop_info = ((ox2 - ox1, oy2 - oy1), crop, quad)

        frame_pil = Image.fromarray(cv2.resize(full_frame, (pic_size, pic_size), interpolation=cv2.INTER_AREA))
        if debug:
            frame_pil.save("frames_pil_256.png")

        # save crop info
        if save_flag:
            cv2.imwrite(png_path, cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR))

        # 2. get the landmark according to the detected face.
        lm = landmark_98_to_68(self.propress.predictor.facelandmark_infer_onnx(np.array(frame_pil)))
        
        if save_flag:
            np.savetxt(landmarks_path, lm)
        video_coeffs, full_coeffs = [], []
        W,H = frame_pil.size  # 256, 256 face image 
        lm1 = lm.reshape([-1, 2])
        if np.mean(lm1) == -1:
            lm1 = (self.lm3d_std[:, :2]+1)/2.
            lm1 = np.concatenate(
                [lm1[:, :1]*W, lm1[:, 1:2]*H], 1
            )
        else:
            lm1[:, -1] = H - 1 - lm1[:, -1]

        trans_params, im1, lm1, _ = align_img(frame_pil, lm1, self.lm3d_std)  
        # 和3d 人脸对齐， croped+resize(256)的人脸和在上面的关键点信息  同3d模型(input is 224,224) 做对齐
        if debug:
            print(trans_params, im1.size, lm1.size) #
            print(np.array(frame_pil))
            im1.save("im1_simple.png")
            print(im1.mode)

        trans_params = np.array([float(item) for item in np.hsplit(trans_params, 5)]).astype(np.float32)
        im_t = torch.tensor(np.array(im1)/255., dtype=torch.float32).permute(2, 0, 1).to(self.device).unsqueeze(0)
        
        im_np = im_t.data.cpu().numpy()
        if debug:
            cv2.imwrite('face_recon_input_simple.png', (im_np[0]*255).astype(np.uint8).transpose(1,2,0))
        ort_inputs = {self.face_recon.get_inputs()[0].name: im_np}
        full_coeff = self.face_recon.run(None, ort_inputs)[0]
        coeffs = split_coeff(full_coeff) 
        pred_coeff = {key:coeffs[key] for key in coeffs}

        pred_coeff = np.concatenate([
            pred_coeff['exp'], 
            pred_coeff['angle'],
            pred_coeff['trans'],
            trans_params[2:][None],
            ], 1)
        
        full_coeffs.append(full_coeff)
        video_coeffs.append(pred_coeff)
        semantic_npy = np.array(video_coeffs)[:,0]

        if save_flag:
            savemat(coeff_path, {'coeff_3dmm': semantic_npy, 'full_3dmm': np.array(full_coeffs)[0]})
        if debug:
            print({'coeff_3dmm': semantic_npy, 'full_3dmm': np.array(full_coeffs)[0]}, frame_pil, crop_info)
            
        return {'coeff_3dmm': semantic_npy, 'full_3dmm': np.array(full_coeffs)[0]}, frame_pil, crop_info
