from torchvision import models
import torch
from torchvision.models import ResNet50_Weights
import torch.nn as nn
import torch.nn.init as init
from torchvision.ops import roi_align
from torch.utils.data import Dataset
from PIL import Image
from torch.nn import functional as F

#使用resnet50提取给定图片的feature_map
def generate_feature_map(x):
    """
    param:
    x:给定图片的张量(bs,channels,height,weight)
    """
    model=models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    feature_extract=nn.Sequential(*list(model.children())[:-2])
    feature_map=feature_extract(x)
    return feature_map
   
#按照候选框从深层特征中提取出固定大小的特征
def map_and_pool(x,feature_map,boxes,output_size=(3,3)):
    """
    param:
    x:原图张量
    feature_map:resnet50提取的深层特征
    boxes:候选框
    output_size:输出的feature_map大小
    """
    #找到当前图片的映射比例
    _,_,orig_h,orig_w=x.shape
    _,_,proc_h,proc_w=feature_map.shape
    
    ###卧槽我悟了！这里的reg是按照原图预测的，所以reg的尺寸压根不是224*224
    scale_h=proc_h/orig_h
    scale_w=proc_w/orig_w
    
    batch_size=x.size(0)
    #建立一个(64,5)的tensor
    mapped_rois=torch.zeros((batch_size,5),device=boxes.device)
    
    #映射
    for batch_idx in range(batch_size):
        mapped_rois[:,0]=batch_idx
        mapped_rois[:,1]=boxes[:,0]*scale_w
        mapped_rois[:,2]=boxes[:,1]*scale_h
        mapped_rois[:,3]=boxes[:,2]*scale_w
        mapped_rois[:,4]=boxes[:,3]*scale_h
    
    #调用roi_align从feature_map中提取出固定大小的特征
    """
    这里的res是空，让我查查bug
    bug是映射不对
    gt_box压根就没限定在224*224里面，所以mapped_rois压根不在特征图的7*7里面
    """
    res=roi_align(feature_map,mapped_rois,output_size=output_size)
    return res
   
#把reg_out还原到原图上
def back_to_image(reg_out):
    #这里做过resize，统一了原图的大小，所以scale直接写就行
    scale=224/7
    xmin=reg_out[:,0]*scale
    ymin=reg_out[:,1]*scale
    xmax=reg_out[:,2]*scale
    ymax=reg_out[:,3]*scale
    reg_pred=torch.stack([xmin,ymin,xmax,ymax],dim=1)
    return reg_pred
    
#转为中心坐标的形式
def box_to_center(boxes):
    center_x=(boxes[:,0]+boxes[:,2])/2
    center_y=(boxes[:,1]+boxes[:,3])/2
    h=boxes[:,3]-boxes[:,1]
    w=boxes[:,2]-boxes[:,0]
    center=torch.stack([center_x,center_y,w,h],dim=1)
    return center
def center_to_box(center):
    xmin=center[:,0]-center[:,2]/2
    ymin=center[:,1]-center[:,3]/2
    xmax=center[:,0]+center[:,2]/2
    ymax=center[:,1]+center[:,3]/2
    boxes=torch.stack([xmin,ymin,xmax,ymax],dim=1)   
    return boxes
    
def initialize(model):
    if isinstance(model,nn.Linear):
        init.kaiming_uniform_(model.weight.data,nonlinearity='relu')
        init.constant_(model.bias.data,0)
    
    #自定义一个数据集，属性有文件名、边界框、类别和transform
class MyDataset(Dataset):
    def __init__(self,filepaths,gt_boxes,gt_labels,class_map,transform):
        self.filepaths=filepaths
        self.gt_boxes=gt_boxes
        self.gt_labels=gt_labels
        self.class_map=class_map
        self.transform=transform
        
    def __len__(self):
        return len(self.gt_boxes)
    
    #返回指定idx对应的张量化的图片、真实框和类别
    def __getitem__(self,idx):
        image_name=self.filepaths[idx]
        image=Image.open(image_name).convert('RGB')
        image=self.transform(image)
        
        xmin=self.gt_boxes.iloc[idx,0]
        ymin=self.gt_boxes.iloc[idx,1]
        xmax=self.gt_boxes.iloc[idx,2]
        ymax=self.gt_boxes.iloc[idx,3]
        _class=self.gt_labels[idx]
        #_class_tensor=torch.tensor(_class)
        
        gt_box=torch.tensor([xmin,ymin,xmax,ymax])
        #labels=F.one_hot(_class_tensor,num_classes=len(self.class_map))
        
        target={'gt_box':gt_box,'labels':_class,'filepath':image_name}
        
        #转张量
        target['gt_box']=torch.as_tensor(target['gt_box'],dtype=torch.float64)
        target['labels']=torch.as_tensor(target['labels'],dtype=torch.int64)
        
        return image,target
    

class DetectionHead(nn.Module):
    def __init__(self,class_num,in_channels,H,W):
        super().__init__()
        self.class_num=class_num
        self.in_channels=in_channels
        self.classfication=nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(in_channels,256),
            nn.ReLU(inplace=True),
            nn.Linear(256,class_num)
        )  
        self.regression=nn.Sequential(
            #nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(in_channels*H*W,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256,4)
        )
        #不加载训练过的数据的话，记得先初始化
        self.classfication.apply(initialize)
        self.regression.apply(initialize)
        
    def forward(self,image_tensor,feature_map):
        #reg_out得到预测框
        reg_out=self.regression(feature_map)
        
        '''
        ###有bug，详见map_and_pool
        reg_out=box_to_center(reg_out)
        #按reg_out剪裁feature_map
        crop=map_and_pool(image_tensor,feature_map,reg_out)
        ##这里没有一个稳定的初始参数，reg最开始的预测很离谱，会导致剪裁出来的crop==NULL
        ##可能需要先想办法对reg做一下预热
        #把裁剪后的crop传入cls
        cls_pred=self.classfication(crop)
        '''
        #试试直接把整张图片丢给cls
        cls_pred=self.classfication(feature_map)
        
        '''
        #把f_m上的reg_out映射到image上的reg_pred
        reg_pred=back_to_image(reg_out)
        ###这里写错了，reg_out本来就在image而不是feature_map上进行预测
        ###做了back反而会因为“要使back后的答案与gt_box接近而使reg_out去适应f_m(7,7)，而不是image(224,224)
        # 但reg_out传给cls_pred时会做crop，所以crop完的res相当于从(7,7)*10**-，导致corp的结果是(0,0)
        return reg_pred,cls_pred
        '''
        return reg_out,cls_pred
    
class DetectionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self,reg_pred,cls_pred,gt_boxes,gt_labels):
        reg_loss=F.smooth_l1_loss(reg_pred,gt_boxes)
        cls_loss=F.cross_entropy(cls_pred,gt_labels)
        
        total_loss=2*reg_loss+cls_loss
        return total_loss
    
class ObjectDetection(nn.Module):
    def __init__(self,class_num,in_channels,H,W):
        super().__init__()
        self.detection_head=DetectionHead(class_num,in_channels,H,W)
    
    def forward(self,x):
        feature_map=generate_feature_map(x)
        reg_pred,cls_pred=self.detection_head(x,feature_map)
        return reg_pred,cls_pred
    
   