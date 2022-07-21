import torch
from torch import nn

class C3D(nn.Module):
    def __init__(self, nb_classes, feature_layer):
        super(C3D, self).__init__()

        self.feature_layer = feature_layer

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        #self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 7), padding=(1, 1, 1), stride=(3,3,1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, nb_classes)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()

        self.outPool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        bs = x.shape[0]
        h = self.relu(self.conv1(x))
        h = self.pool1(h)

        h = self.relu(self.conv2(h))
        h = self.pool2(h)

        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3b(h))
        h = self.pool3(h)

        h = self.relu(self.conv4a(h))
        h = self.relu(self.conv4b(h))
        h = self.pool4(h)

        h = self.relu(self.conv5a(h))
        h = self.relu(self.conv5b(h))
        h = self.pool5(h)

        #h = h.view(-1, 8192)
        h = h.view(bs, -1, 8192) # To deal with the multiple segments cases
        out = h if self.feature_layer == 5 else None
        h = self.relu(self.fc6(h))
        out = h if self.feature_layer == 6 and out == None else out
        h = self.dropout(h)
        h = self.relu(self.fc7(h))
        out = h if self.feature_layer == 7 and out == None else out
        h = self.dropout(h)
        logits = self.fc8(h)
        ## Pool all video segments within one video
        #out = self.outPool(out.permute(0, 2, 1)).squeeze(2) # Move to outside
        return logits, out


class VideoEmbed(nn.Module):
    def __init__(self, pretrained, trainable, nb_classes=487, feature_layer=6):
        super(VideoEmbed, self).__init__()
        #self.conv = nn.Conv1d(input_size, hidden_size, 1, 1)
        #self.pool = nn.AvgPool1d(kernel_size, stride)
        self.embednet = C3D(nb_classes, feature_layer)
        if pretrained:
            self.embednet.load_state_dict(torch.load("c3d.pickle"))

        for p in self.embednet.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        #return self.pool(self.conv(x.transpose(1, 2)).relu())
        #featList = []
        #for _x in x:
        #   featList.append(self.embednet(_x)[1])
        import ipdb; ipdb.set_trace()
        return self.embednet(x)[1]

def build_videoEncoder(cfg):
    #input_size = cfg.MODEL.TAN.FEATPOOL.INPUT_SIZE
    #hidden_size = cfg.MODEL.TAN.FEATPOOL.HIDDEN_SIZE
    #kernel_size = cfg.MODEL.TAN.FEATPOOL.KERNEL_SIZE
    #stride = cfg.INPUT.NUM_PRE_CLIPS // cfg.MODEL.TAN.NUM_CLIPS
    #return FeatAvgPool(input_size, hidden_size, kernel_size, stride)
    pretrained = cfg.MODEL.CLIP.VIDEOEMBED.PRETRAIN
    trainable = cfg.MODEL.CLIP.VIDEOEMBED.TRAINABLE
    nb_classes = cfg.MODEL.CLIP.VIDEOEMBED.NBCLASS
    feature_layer = cfg.MODEL.CLIP.VIDEOEMBED.LAYER
    return VideoEmbed(pretrained, trainable, nb_classes, feature_layer)
