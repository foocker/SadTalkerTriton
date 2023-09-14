import torch
from torch import nn
from src.audio2pose_models.cvae import CVAE
from src.audio2pose_models.audio_encoder import AudioEncoder

class Audio2Pose(nn.Module):
    def __init__(self, cfg, wav2lip_checkpoint, device='cuda'):
        super().__init__()
        self.cfg = cfg
        self.seq_len = cfg.MODEL.CVAE.SEQ_LEN
        self.latent_dim = cfg.MODEL.CVAE.LATENT_SIZE
        self.device = device

        self.audio_encoder = AudioEncoder(wav2lip_checkpoint, device)
        self.audio_encoder.eval()
        for param in self.audio_encoder.parameters():
            param.requires_grad = False

        self.netG = CVAE(cfg)
        

    def test(self, x):
        # for k, v in x.items():
        #     try:
        #         print(k, v.shape)
        #     except:
        #         print(k, v)
        from tqdm import tqdm 
        import time
        batch = {}
        ref = x['ref']                            #bs 1 70
        batch['ref'] = x['ref'][:,0,-6:]  
        batch['class'] = x['class']  
        bs = ref.shape[0]
        
        indiv_mels= x['indiv_mels']               # bs T 1 80 16
        indiv_mels_use = indiv_mels[:, 1:]        # we regard the ref as the first frame
        num_frames = x['num_frames']
        num_frames = int(num_frames) - 1

        #  num_frames 和  seq_len 在一次编解码上数目一致。且为8的倍数
        div = num_frames//self.seq_len
        re = num_frames%self.seq_len
        pose_motion_pred_list = [torch.zeros(batch['ref'].unsqueeze(1).shape, dtype=batch['ref'].dtype, 
                                                device=batch['ref'].device)]
        # st = time.time()
        for i in tqdm(range(div), 'audio2pose:'):
            z = torch.randn(bs, self.latent_dim).to(ref.device)
            batch['z'] = z
            audio_emb = self.audio_encoder(indiv_mels_use[:, i*self.seq_len:(i+1)*self.seq_len,:,:,:]) #bs seq_len 512
            batch['audio_emb'] = audio_emb
            batch = self.netG.test(batch)
            pose_motion_pred_list.append(batch['pose_motion_pred'])  #list of bs seq_len 6
        # ed = time.time()
        # print('cost of audio2pose is ', ed - st, 'fps is ', 1/(ed - st))
        
        # st = time.time()
        if re != 0:
            for i in tqdm(range(1), 'audio2pose res:'):
                z = torch.randn(bs, self.latent_dim).to(ref.device)
                batch['z'] = z
                audio_emb = self.audio_encoder(indiv_mels_use[:, -1*self.seq_len:,:,:,:]) #bs seq_len  512
                if audio_emb.shape[1] != self.seq_len:
                    pad_dim = self.seq_len-audio_emb.shape[1]
                    pad_audio_emb = audio_emb[:, :1].repeat(1, pad_dim, 1) 
                    audio_emb = torch.cat([pad_audio_emb, audio_emb], 1) 
                batch['audio_emb'] = audio_emb
                batch = self.netG.test(batch)
                pose_motion_pred_list.append(batch['pose_motion_pred'][:,-1*re:,:])  
        # ed = time.time()
        # print('cost of audio2pose is ', ed - st, 'fps is ', 1/(ed - st), 'rest is', re)
        
        pose_motion_pred = torch.cat(pose_motion_pred_list, dim = 1)
        batch['pose_motion_pred'] = pose_motion_pred

        pose_pred = ref[:, :1, -6:] + pose_motion_pred  # bs T 6

        batch['pose_pred'] = pose_pred
        return batch
