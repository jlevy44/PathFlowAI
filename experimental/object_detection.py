import lightnet as ln
import torch
import numpy as np
import matplotlib.pyplot as plt
import brambox as bb
import dask as da
from datasets import BramboxPathFlowDataset

# Settings
ln.logger.setConsoleLevel('ERROR')             # Only show error log messages
bb.logger.setConsoleLevel('ERROR')
# https://eavise.gitlab.io/lightnet/notes/02-B-engine.html

num_classes=3
patch_size=256
patch_info_file='cell_info.db'
annotation_file = 'annotations_bbox_{}.pkl'.format(patch_size)
annotations=bb.io.load('pandas',annotation_file)

model=ln.models.Yolo(num_classes=num_classes)

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

dataset=BramboxPathFlowDataset(patch_info_file, patch_size, annotations, input_dimension=(patch_size,patch_size), class_label_map=None, identify=None, img_transform=transforms, anno_transform=None)

class CustomEngine(ln.engine.Engine):
    def start(self):
        """ Do whatever needs to be done before starting """
        self.params.to(self.device)  # Casting parameters to a certain device
        self.optim.zero_grad()       # Make sure to start with no gradients
        self.loss_acc = []           # Loss accumulator

    def process_batch(self, data):
        """ Forward and backward pass """
        data, target = data  # Unpack

        output = self.network(data)
        loss = self.loss(output, target)
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

# Create HyperParameters
params = ln.engine.HyperParameters(
    network=model,
    mini_batch_size=8,
    batch_size=64,
    max_batches=128
)
params.loss = ln.network.loss.RegionLoss(params.network.num_classes, params.network.anchors)
params.optim = torch.optim.SGD(params.network.parameters(), lr=0.001)


dl = ln.data.DataLoader(
    dataset,
    batch_size = 2,
    collate_fn = ln.data.list_collate   # We want the data to be grouped as a list
)

# Create engine
engine = CustomEngine(
    params, dl,              # Dataloader (None) is not valid
    device=torch.device('gpu') if torch.cuda.is_available() else torch.device('cpu')
)
