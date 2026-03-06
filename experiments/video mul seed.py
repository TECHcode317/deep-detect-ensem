!pip install -q timm ptflops

import os
import time
import random
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models

from torch.utils.data import Dataset,DataLoader
from PIL import Image

from sklearn.metrics import classification_report

from ptflops import get_model_complexity_info



device="cuda" if torch.cuda.is_available() else "cpu"

print("Device:",device)



DATASET="/kaggle/input/1000-deepfake-videos/1000_videos"



def seed_everything(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def group_videos(root):

    groups=defaultdict(list)

    for label,cls in enumerate(["Fake","Real"]):

        folder=os.path.join(root,cls)

        for f in os.listdir(folder):

            if not f.endswith(".png"):
                continue

            vid=f.split("_")[0]

            groups[(label,vid)].append(
                os.path.join(folder,f))

    return groups


video_groups=group_videos(DATASET)

videos=list(video_groups.keys())

print("Total Videos:",len(videos))



def stratified_split():

    fake=[v for v in videos if v[0]==0]
    real=[v for v in videos if v[0]==1]

    random.shuffle(fake)
    random.shuffle(real)

    def split(lst):

        n=len(lst)

        return(

            lst[:int(.7*n)],
            lst[int(.7*n):int(.85*n)],
            lst[int(.85*n):]
        )

    ftr,fval,ftest=split(fake)
    rtr,rval,rtest=split(real)

    return ftr+rtr, fval+rval, ftest+rtest




def count_frames(keys):

    fake=sum(
        len(video_groups[k])
        for k in keys if k[0]==0)

    real=sum(
        len(video_groups[k])
        for k in keys if k[0]==1)

    return fake,real




train_tf=T.Compose([

T.Resize((224,224)),
T.RandomHorizontalFlip(),

T.ColorJitter(.2,.2,.2,.1),

T.ToTensor(),

T.Normalize(
[0.485,0.456,0.406],
[0.229,0.224,0.225])

])

val_tf=T.Compose([

T.Resize((224,224)),

T.ToTensor(),

T.Normalize(
[0.485,0.456,0.406],
[0.229,0.224,0.225])

])



class DeepfakeDataset(Dataset):

    def __init__(self,keys,train=True):

        self.samples=[]

        for k in keys:

            label,_=k

            for path in video_groups[k]:

                self.samples.append((path,label))

        self.tf=train_tf if train else val_tf

    def __len__(self):

        return len(self.samples)

    def __getitem__(self,i):

        path,label=self.samples[i]

        img=Image.open(path).convert("RGB")

        img=self.tf(img)

        return img,label,path




def build_models():

    mobilenet=models.mobilenet_v3_small(
        weights="DEFAULT")

    mobilenet.classifier[3]=nn.Linear(
        mobilenet.classifier[3].in_features,2)

    shufflenet=models.shufflenet_v2_x1_0(
        weights="DEFAULT")

    shufflenet.fc=nn.Linear(
        shufflenet.fc.in_features,2)

    efficient=models.efficientnet_b0(
        weights="DEFAULT")

    efficient.classifier[1]=nn.Linear(
        efficient.classifier[1].in_features,2)

    return(
        mobilenet.to(device),
        shufflenet.to(device),
        efficient.to(device)
    )




def flops_report(model,name):

    macs,params=get_model_complexity_info(

        model,(3,224,224),
        as_strings=True,
        verbose=False)

    print(name,"FLOPS:",macs,"Params:",params)




def train_model(model,train_loader,val_loader,name):

    opt=torch.optim.AdamW(
        model.parameters(),lr=3e-4)

    loss_fn=nn.CrossEntropyLoss()

    flops_report(model,name)

    for epoch in range(3):

        model.train()

        for imgs,labels,_ in train_loader:

            imgs=imgs.to(device)
            labels=labels.to(device)

            opt.zero_grad()

            out=model(imgs)

            loss=loss_fn(out,labels)

            loss.backward()

            opt.step()

      

        model.eval()

        correct=0
        total=0

        start=time.time()

        with torch.no_grad():

            for imgs,labels,_ in val_loader:

                imgs=imgs.to(device)

                out=model(imgs)

                pred=out.argmax(1).cpu()

                correct+=(pred==labels).sum().item()

                total+=len(labels)

        latency=(time.time()-start)/total

        print(
        name,
        "epoch",epoch,
        "val acc",correct/total,
        "latency/frame",latency
        )



SEEDS=[42,77,123]

video_accs=[]

for SEED in SEEDS:

    print("\nSEED:",SEED)

    seed_everything(SEED)

    train_videos,val_videos,test_videos=stratified_split()

    tf,tr=count_frames(train_videos)
    vf,vr=count_frames(val_videos)

    print("Train Videos:",len(train_videos))
    print("Val Videos:",len(val_videos))

    print("Train Frames:",tf+tr)
    print("Val Frames:",vf+vr)

    train_ds=DeepfakeDataset(train_videos,True)
    val_ds=DeepfakeDataset(val_videos,False)

    train_loader=DataLoader(
    train_ds,
    batch_size=32,
    shuffle=True,
    num_workers=2,
    pin_memory=True)

    val_loader=DataLoader(
    val_ds,
    batch_size=32,
    shuffle=False,
    num_workers=2)

    mobilenet,shufflenet,efficientnet=build_models()

    train_model(
    mobilenet,
    train_loader,
    val_loader,
    "MobileNet")

    train_model(
    shufflenet,
    train_loader,
    val_loader,
    "ShuffleNet")

    train_model(
    efficientnet,
    train_loader,
    val_loader,
    "EfficientNet")



    mobilenet.eval()
    shufflenet.eval()
    efficientnet.eval()

    video_votes=defaultdict(list)
    video_gt={}

    with torch.no_grad():

        for imgs,labels,paths in val_loader:

            imgs=imgs.to(device)

            p1=torch.softmax(
            mobilenet(imgs),1)

            p2=torch.softmax(
            shufflenet(imgs),1)

            p3=torch.softmax(
            efficientnet(imgs),1)

            pred=((p1+p2+p3)/3)\
                .argmax(1).cpu()

            for pr,gt,path in zip(
                pred,labels,paths):

                vid=os.path.basename(path)\
                    .split("_")[0]

                video_votes[vid].append(
                    pr.item())

                video_gt[vid]=gt.item()

    vt=[]
    vp=[]

    for v in video_votes:

        votes=video_votes[v]

        maj=max(
            set(votes),
            key=votes.count)

        vp.append(maj)
        vt.append(video_gt[v])

    rep=classification_report(
        vt,vp,
        output_dict=True)

    acc=rep["accuracy"]

    print("\nVIDEO ACC:",acc)

    video_accs.append(acc)



print("\nFINAL RESULT")

print(
"Mean Video Accuracy:",
np.mean(video_accs),
"+/-",
np.std(video_accs)
)
