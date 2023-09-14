import torch
from torch import nn
from src.audio2pose_models.res_unet import ResUnet

def class2onehot(idx, class_num):

    assert torch.max(idx).item() < class_num
    onehot = torch.zeros(idx.size(0), class_num).to(idx.device)
    onehot.scatter_(1, idx, 1)
    return onehot

class CVAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        decoder_layer_sizes = cfg.MODEL.CVAE.DECODER_LAYER_SIZES
        latent_size = cfg.MODEL.CVAE.LATENT_SIZE
        num_classes = cfg.DATASET.NUM_CLASSES
        audio_emb_in_size = cfg.MODEL.CVAE.AUDIO_EMB_IN_SIZE
        audio_emb_out_size = cfg.MODEL.CVAE.AUDIO_EMB_OUT_SIZE
        seq_len = cfg.MODEL.CVAE.SEQ_LEN

        self.latent_size = latent_size
        self.decoder = DECODER(decoder_layer_sizes, latent_size, num_classes,
                                audio_emb_in_size, audio_emb_out_size, seq_len)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def test(self, batch):
        '''
        class_id = batch['class']
        z = torch.randn([class_id.size(0), self.latent_size]).to(class_id.device)
        batch['z'] = z
        '''
        return self.decoder(batch)

class DECODER(nn.Module):
    def __init__(self, layer_sizes, latent_size, num_classes, 
                audio_emb_in_size, audio_emb_out_size, seq_len):
        super().__init__()

        self.resunet = ResUnet()
        self.num_classes = num_classes
        self.seq_len = seq_len

        self.MLP = nn.Sequential()
        input_size = latent_size + seq_len*audio_emb_out_size + 6
        for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            else:
                self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())
        
        self.pose_linear = nn.Linear(6, 6)
        self.linear_audio = nn.Linear(audio_emb_in_size, audio_emb_out_size)

        self.classbias = nn.Parameter(torch.randn(self.num_classes, latent_size))

    def forward(self, batch):

        z = batch['z']                                          #bs latent_size
        bs = z.shape[0]
        class_id = batch['class']
        ref = batch['ref']                             #bs 6
        audio_in = batch['audio_emb']                           # bs seq_len audio_emb_in_size
        #print('audio_in: ', audio_in[:, :, :10])

        audio_out = self.linear_audio(audio_in)                 # bs seq_len audio_emb_out_size
        #print('audio_out: ', audio_out[:, :, :10])
        audio_out = audio_out.reshape([bs, -1])                 # bs seq_len*audio_emb_out_size
        # audio_out = audio_out.reshape([-1, int(audio_out.numel()/bs)])
        class_bias = self.classbias[class_id]                   #bs latent_size

        z = z + class_bias
        x_in = torch.cat([ref, z, audio_out], dim=-1)
        x_out = self.MLP(x_in)                                  # bs layer_sizes[-1]
        x_out = x_out.reshape(bs, self.seq_len, -1)
        # x_out = x_out.reshape((-1, self.seq_len, int(x_out.numel()/(bs*self.seq_len))))
        x_out = x_out.unsqueeze(1)

        pose_emb = self.resunet(x_out)             #bs 1 seq_len 6
        pose_emb = pose_emb.squeeze(1)
        pose_motion_pred = self.pose_linear(pose_emb)       #bs seq_len 6

        batch.update({'pose_motion_pred':pose_motion_pred})
        return batch
