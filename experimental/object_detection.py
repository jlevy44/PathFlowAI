import lightnet as ln
import torch
import numpy as np
import pandas as pd

# import matplotlib.pyplot as plt
import brambox as bb

# import dask as da
from datasets import BramboxPathFlowDataset
import argparse
import pickle
from sklearn.model_selection import train_test_split

# Settings
ln.logger.setConsoleLevel("ERROR")  # Only show error log messages
bb.logger.setConsoleLevel("ERROR")
# https://eavise.gitlab.io/lightnet/notes/02-B-engine.html

p = argparse.ArgumentParser()
p.add_argument("--num_classes", default=4, type=int)
p.add_argument("--patch_size", default=512, type=int)
p.add_argument("--patch_info_file", default="cell_info.db", type=str)
p.add_argument("--input_dir", default="inputs", type=str)
p.add_argument("--sample_p", default=1.0, type=float)
p.add_argument("--conf_thresh", default=0.01, type=float)
p.add_argument("--nms_thresh", default=0.5, type=float)


args = p.parse_args()
np.random.seed(42)
num_classes = args.num_classes + 1
patch_size = args.patch_size
batch_size = 64
patch_info_file = args.patch_info_file
input_dir = args.input_dir
sample_p = args.sample_p
conf_thresh = args.conf_thresh
nms_thresh = args.nms_thresh
anchors = pickle.load(open("anchors.pkl", "rb"))

annotation_file = "annotations_bbox_{}.pkl".format(patch_size)
annotations = bb.io.load("pandas", annotation_file)

if sample_p < 1.0:
    annotations = annotations.sample(frac=sample_p)

annotations_dict = {}
annotations_dict["train"], annotations_dict["test"] = train_test_split(annotations)
annotations_dict["train"], annotations_dict["val"] = train_test_split(
    annotations_dict["train"]
)

model = ln.models.Yolo(num_classes=num_classes, anchors=anchors.tolist())

loss = ln.network.loss.RegionLoss(
    num_classes=model.num_classes, anchors=model.anchors, stride=model.stride
)

transforms = ln.data.transform.Compose(
    [ln.data.transform.RandomHSV(hue=1, saturation=2, value=2)]
)

# Create HyperParameters
params = ln.engine.HyperParameters(
    network=model,
    input_dimension=(patch_size, patch_size),
    mini_batch_size=16,
    batch_size=batch_size,
    max_batches=80000,
)

post = ln.data.transform.Compose(
    [
        ln.data.transform.GetBoundingBoxes(
            num_classes=params.network.num_classes,
            anchors=params.network.anchors,
            conf_thresh=conf_thresh,
        ),
        ln.data.transform.NonMaxSuppression(nms_thresh=nms_thresh),
        ln.data.transform.TensorToBrambox(
            network_size=(patch_size, patch_size),
            # class_label_map=class_label_map,
        ),
    ]
)

datasets = {
    k: BramboxPathFlowDataset(
        input_dir,
        patch_info_file,
        patch_size,
        annotations_dict[k],
        input_dimension=(patch_size, patch_size),
        class_label_map=None,
        identify=None,
        img_transform=None,
        anno_transform=None,
    )
    for k in ["train", "val", "test"]
}
# transforms

params.loss = ln.network.loss.RegionLoss(
    params.network.num_classes, params.network.anchors
)
params.optim = torch.optim.SGD(params.network.parameters(), lr=1e-4)
params.scheduler = ln.engine.SchedulerCompositor(
    #   batch   scheduler
    (0, torch.optim.lr_scheduler.CosineAnnealingLR(params.optim, T_max=200))
)

dls = {
    k: ln.data.DataLoader(
        datasets[k],
        batch_size=batch_size,
        collate_fn=ln.data.brambox_collate,  # We want the data to be grouped as a list
    )
    for k in ["train", "val", "test"]
}

params.val_loader = dls["val"]


class CustomEngine(ln.engine.Engine):
    def start(self):
        """ Do whatever needs to be done before starting """
        self.params.to(self.device)  # Casting parameters to a certain device
        self.optim.zero_grad()  # Make sure to start with no gradients
        self.loss_acc = []  # Loss accumulator

    def process_batch(self, data):
        """ Forward and backward pass """
        data, target = data  # Unpack
        # print(target)
        data = data.permute(0, 3, 1, 2).float()
        if torch.cuda.is_available():
            data = data.cuda()

        # print(data)

        output = self.network(data)
        # print(output)

        loss = self.loss(output, target)

        # print(loss)
        loss.backward()
        bbox = post(output)
        print(bbox)

        self.loss_acc.append(loss.item())

    @ln.engine.Engine.batch_end(100)  # how to pass in validation dataloader
    def val_loop(self):
        with torch.no_grad():
            for i, data in enumerate(self.val_loader):
                if i > 100:
                    break
                data, target = data
                data = data.permute(0, 3, 1, 2).float()
                if torch.cuda.is_available():
                    data = data.cuda()
                output = self.network(data)
                # print(output)
                loss = self.loss(output, target)
                print(loss)
                bbox = post(output)
                print(bbox)
                if not i:
                    bbox_final = [bbox]
                else:
                    bbox_final.append(bbox)

            detections = pd.concat(bbox_final)
            print(detections)
            print(annotations_dict["val"])
            pr = bb.stat.pr(detections, annotations_dict["val"], threshold=0.5)
            auc = bb.stat.auc(pr)
            print("VAL AUC={}".format(auc))

    @ln.engine.Engine.batch_end(300)
    def save_model(self):
        self.params.save(f"backup-{self.batch}.state.pt")

    def train_batch(self):
        """ Weight update and logging """
        self.optim.step()
        self.optim.zero_grad()

        batch_loss = sum(self.loss_acc) / len(self.loss_acc)
        self.loss_acc = []
        self.log(f"Loss: {batch_loss}")

    def quit(self):
        if self.batch >= self.max_batches:  # Should probably save weights here
            print("Reached end of training")
            return True
        return False


# Create engine
engine = CustomEngine(
    params,
    dls["train"],  # Dataloader (None) is not valid
    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
)

for i in range(10):
    engine()
