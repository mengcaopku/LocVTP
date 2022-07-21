import torch
import torch.nn as nn

class Shiftpredict(nn.Module):
    def __init__(self, projection_dim, out_dim, dropout=0.1):
        super(Shiftpredict, self).__init__()
        '''
        # Version 1
        self.projection = nn.Linear(2*projection_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc1 = nn.Linear(projection_dim, projection_dim)
        self.fc2 = nn.Linear(projection_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        #self.layer_norm = nn.LayerNorm(projection_dim)
        #self.outact = nn.Tanh() # For regression
        #self.outact = nn.Softmax(dim=-1) # For classification
        self.outact = nn.LeakyReLU() # For classification without normalization
        '''
        '''
        version 2
        self.f_reduce = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=128, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU()
        )
        self.f_cls = nn.Linear(1280, 10)
        '''
        self.f_cls = nn.Linear(1280, 5)
    
    def forward(self, svidEmbed, textEmbed):
        '''
        # Version 1
        projected = self.projection(torch.cat((svidEmbed, textEmbed), dim=1))
        out = self.gelu(projected)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.outact(out)
        #return out.squeeze(1) # For regression
        return out # For classification
        '''
        '''
        # Version 2
        batch_size = svidEmbed.shape[0]
        textEmbed = textEmbed.unsqueeze(1).expand(-1, 10, -1)
        catEmbed = torch.cat((textEmbed, svidEmbed), dim=2) # [batch_size, 10, 512]
        catEmbed = catEmbed.permute(0, 2, 1) # [batch_size, 512, 10]
        catEmbed = self.f_reduce(catEmbed) # [batch_size, 128, 10]
        catEmbed = catEmbed.reshape(batch_size, -1)
        out = self.f_cls(catEmbed) # TODO: check "self.outact"
        '''
        svidEmbed = svidEmbed.view(-1, 1280)
        out = self.f_cls(svidEmbed)
        #print("out[0]: {}".format(out[0]))
        return out
 


def build_shiftpredict(cfg):
    proj_dim = cfg.MODEL.CLIP.PROJECTION.DIM # Input feat dim
    #out_dim = len(cfg.INPUT.SHIFTRATIO)
    out_dim = cfg.INPUT.NUM_SEGMENTS
    return Shiftpredict(
        proj_dim, out_dim
    ) 