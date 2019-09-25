import lightnet as ln
import torch
import numpy as np
import matplotlib.pyplot as plt
import brambox as bb
import dask as da
from datasets import BramboxPathFlowDataset
import argparse, pickle

# Settings
ln.logger.setConsoleLevel('ERROR')             # Only show error log messages
bb.logger.setConsoleLevel('ERROR')
# https://eavise.gitlab.io/lightnet/notes/02-B-engine.html

p=argparse.ArgumentParser()
p.add_argument('--num_classes',default=4,type=int)
p.add_argument('--patch_size',default=512,type=int)
p.add_argument('--patch_info_file',default='cell_info.db',type=str)
p.add_argument('--input_dir',default='inputs',type=str)

args=p.parse_args()
num_classes=args.num_classes+1
patch_size=args.patch_size
patch_info_file=args.patch_info_file
input_dir=args.input_dir
anchors=pickle.load(open('anchors.pkl','rb'))

annotation_file = 'annotations_bbox_{}.pkl'.format(patch_size)
annotations=bb.io.load('pandas',annotation_file)

model=ln.models.Yolo(num_classes=num_classes,anchors=anchors.tolist())

loss = ln.network.loss.RegionLoss(
    num_classes=model.num_classes,
    anchors=model.anchors,
    stride=model.stride
)

transforms = ln.data.transform.Compose([ln.data.transform.RandomHSV(
    hue=1,
    saturation=2,
    value=2
)])

# Create HyperParameters
params = ln.engine.HyperParameters(
    network=model,
    mini_batch_size=8,
    batch_size=64,
    max_batches=128
)
params.loss = ln.network.loss.RegionLoss(params.network.num_classes, params.network.anchors)
params.optim = torch.optim.SGD(params.network.parameters(), lr=1e-5)

post = ln.data.transform.Compose([
    ln.data.transform.GetBoundingBoxes(
        num_classes=params.network.num_classes,
        anchors=params.network.anchors,
        conf_thresh=0.5,
    ),

    ln.data.transform.NonMaxSuppression(
        nms_thresh=0.5
    ),

    ln.data.transform.TensorToBrambox(
        network_size=(patch_size,patch_size),
        # class_label_map=class_label_map,
    )
])

dataset=BramboxPathFlowDataset(input_dir,patch_info_file, patch_size, annotations, input_dimension=(patch_size,patch_size), class_label_map=None, identify=None, img_transform=transforms, anno_transform=None)

class CustomEngine(ln.engine.Engine):
    def start(self):
        """ Do whatever needs to be done before starting """
        self.params.to(self.device)  # Casting parameters to a certain device
        self.optim.zero_grad()       # Make sure to start with no gradients
        self.loss_acc = []           # Loss accumulator

    def process_batch(self, data):
        """ Forward and backward pass """
        data, target = data  # Unpack
        #print(target)
        data=data.permute(0,3,1,2).float()
        if torch.cuda.is_available():
            data=data.cuda()

        #print(data)

        output = self.network(data)
        #print(output)
        loss = self.loss(output, target)
        print(loss)
        loss.backward()

        self.loss_acc.append(loss.item())

    def train_batch(self):
        """ Weight update and logging """
        self.optim.step()
        self.optim.zero_grad()

        batch_loss = sum(self.loss_acc) / len(self.loss_acc)
        self.loss_acc = []
        self.log(f'Loss: {batch_loss}')

    def quit(self):
        if self.batch >= self.max_batches:  # Should probably save weights here
            print('Reached end of training')
            return True
        return False

dl = ln.data.DataLoader(
    dataset,
    batch_size = 2,
    collate_fn = ln.data.brambox_collate   # We want the data to be grouped as a list
)

# Create engine
engine = CustomEngine(
    params, dl,              # Dataloader (None) is not valid
    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
)

engine()
