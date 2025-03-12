from torchvision import transforms
import pandas as pd
import os
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from tools import *

def main(folderpath,save_path):
    #设定一些常量
    transform=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    class_map={"background":0,"bottle":1,"car":2}
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #读取csv数据
    csv_path=os.path.join(folderpath,'detection_label.csv')
    data=pd.read_csv(csv_path)

    #把类别名映射到数值
    data['class']=data['class'].map(class_map)

    #读取图片到张量
    #找到待预测文件夹的路径
    images_folder_path=os.path.join(folderpath,'image')
    #按照filename的顺序排列文件地址并保存到filepaths里
    image_paths=[os.path.join(images_folder_path,i) for i in data['filename']]
    dataset=MyDataset(image_paths,data[['xmin','ymin','xmax','ymax']],data['class'],class_map,transform)
    dataloader=DataLoader(dataset,batch_size=32,shuffle=True)#得到(bs,channels,height,weight)
    
    model=ObjectDetection(class_num=3,in_channels=2048,H=7,W=7)
    critertion=DetectionLoss()
    optimizer=torch.optim.SGD(model.parameters(),lr=0.001,momentum=0.8,weight_decay=0.0005)
    model.load_state_dict(torch.load(r'version3/model6.path'))
    model.train()
    model.to(device)
    for epoch in range(5):
        epoch_loss=0
        for batch_idx,(images,targets) in enumerate(tqdm(dataloader)):
            reg_pred,cls_pred=model(images)
            center=box_to_center(targets['gt_box'])
            loss=critertion(reg_pred,cls_pred,center,targets['labels'])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss+=loss.item()
        print(f'Epoch:{epoch},Loss:{epoch_loss/len(images)}')
    torch.save(model.state_dict(),save_path)
            
if __name__=='__main__':
    folderpath=r'object_detection_data'
    save_path=r'version3/model7.path'
    main(folderpath,save_path)