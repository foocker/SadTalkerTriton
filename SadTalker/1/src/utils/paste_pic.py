import time
import cv2
import numpy as np
import multiprocessing as mp


def process_image_single(crop_frame, full_img, ox1, ox2, oy1, oy2):
    p = cv2.resize(crop_frame.astype(np.uint8), (ox2 - ox1, oy2 - oy1))
    # print(p.shape, 'process image single ')
    mask = 255 * np.ones(p.shape, p.dtype)
    location = ((ox1 + ox2) // 2, (oy1 + oy2) // 2)  
    gen_img = cv2.seamlessClone(p, full_img, mask, location, cv2.NORMAL_CLONE)
    return gen_img

def parallel_poisson_fusion(crop_frames, full_img, ox1, ox2, oy1, oy2, fps=25, reduce=False):
    
    if reduce:
        # 泊松融合数量减半，测试
        crop_frames_odd = crop_frames[1::2]
        crop_frames_even = crop_frames[0::2]
        crop_frames = crop_frames_even
    num_processes = len(crop_frames) // fps
    reminder = len(crop_frames) % fps 
    if reminder / fps > 0.55:
        num_processes += 2

    pool = mp.Pool(processes=num_processes)

    # 定义批次大小
    batch_size = len(crop_frames) // num_processes
    # 将crop_frames按批次分割
    crop_batches = [crop_frames[i:i + batch_size] for i in range(0, len(crop_frames), batch_size)]
    
    results = []

    for batch_idx, crop_batch in enumerate(crop_batches):
        for crop_frame in crop_batch:
            result = pool.apply_async(process_image_single, (crop_frame, full_img, ox1, ox2, oy1, oy2))
            results.append(result)

    pool.close()
    pool.join()
    
    # 获取处理后的图像
    processed_images = [result.get() for result in results]
    if reduce:
        processed_images = [item for item in processed_images for _ in range(2)]

    return processed_images

def encode_frames_bytes(frames):
    encoded_frames  = []
    frame_sizes = []
    for frame in frames:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        result, enc_frame = cv2.imencode('.jpg', frame, encode_param)
        if result:
            size = len(enc_frame)
            encoded_frames.append(enc_frame)
            frame_sizes.append(size)
            
    frame_sizes_arr = np.array(frame_sizes, dtype=np.int32)
    encoded_frames_bytes = b''.join(encoded_frames) 
            
    return frame_sizes_arr, encoded_frames_bytes

    
def decode_video_with_save(outputs, fps=25, save_name="/models/SadTalker/1/results/final_video.mp4"):
    frame_sizes = outputs[0]
    if frame_sizes.dtype != np.int32:
        frame_sizes = np.frombuffer(outputs[0], dtype=np.int32)
    encoded_frames = outputs[1]
    frame_cursor = 0
    
    for size in frame_sizes:
        frame_data = encoded_frames[frame_cursor:frame_cursor+size]
        frame = cv2.imdecode(np.frombuffer(frame_data, dtype='uint8'), cv2.IMREAD_COLOR)
        h, w = frame.shape[:2]
        break
    fourcc = cv2.VideoWriter_fourcc(*"mp4v") 
    out = cv2.VideoWriter(save_name, fourcc, fps, (w, h))
    
    for size in frame_sizes:
        frame_data = encoded_frames[frame_cursor:frame_cursor+size]
        frame = cv2.imdecode(np.frombuffer(frame_data, dtype='uint8'), cv2.IMREAD_COLOR)
        frame_cursor += size
        out.write(frame)
    out.release()


def paste_pic_func(crop_frames, full_img, crop_info, original_wav, fps=25, test_local=False, **kwargs):
    if len(crop_info) != 3:
        print("you didn't crop the image")
        return
    else:
        clx, cly, crx, cry = crop_info[1]
        lx, ly, rx, ry = crop_info[2]
        lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
        
        oy1, oy2, ox1, ox2 = cly + ly, cly + ry, clx + lx, clx + rx
    
    # 多进程
    # st = time.time()
    processed_images = parallel_poisson_fusion(crop_frames, full_img, ox1, ox2, oy1, oy2, fps=fps, reduce=False)
    # ed = time.time()
    # print(" x multiprocess poisson cost time", ed - st, "fps is ", 1/(ed - st) * len(crop_frames), 'not contain encode decode save')
    
    outputs = encode_frames_bytes(processed_images)
    # 本地测试
    if test_local:
        from src.utils.videoio import merge_video_with_audio
        from src.utils.audio import save_wav_sf
        import os
        video_save_dir="/models/SadTalker/1/results"
        audio_save_name="input_audio.wav"
        video_save_name="generated_video.mp4"
        video_audio_save_name="generated_video_audio.mp4"
        audio_save_path = os.path.join(video_save_dir, audio_save_name)
        video_save_path = os.path.join(video_save_dir, video_save_name)
        video_audio_save_path = os.path.join(video_save_dir, video_audio_save_name)
        save_wav_sf(original_wav, audio_save_path, sr=16000)
        decode_video_with_save(outputs, save_name=video_save_path)
        merge_video_with_audio(video_save_path, audio_save_path, video_audio_save_path)
    
    return outputs




