from tools import ObjectDetection,MyDataset,center_to_box
from torch.utils.data import DataLoader
import pandas as pd
import os
from torchvision import transforms
import torch
from tqdm import tqdm
import cv2
from torch.nn import functional as F

def visual(results):
    for result in results:
        img=cv2.imread(result['filepath'])
        cv2.rectangle(img,(int(result['gt_box'][0]),int(result['gt_box'][1])),(int(result['gt_box'][2]),int(result['gt_box'][3])),(0,255,0),2)
        cv2.rectangle(img,(int(result['box'][0]),int(result['box'][1])),(int(result['box'][2]),int(result['box'][3])),(0,0,255),2)
        cv2.putText(img,f"{result['class_name']}:{result['score']:.2f}",(int(result['box'][0]),int(result['box'][1])-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,0,0),2)
        cv2.imshow("Object Detection",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main(folderpath,load_path):
    #设定一些常量
    transform=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    class_map={"background":0,"bottle":1,"car":2}
    class_map_t={0:'background',1:'bottle',2:'car'}
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #读取csv数据
    csv_path=os.path.join(folderpath,'detection_label.csv')
    data=pd.read_csv(csv_path)

    #把类别名映射到数值
    data['class']=data['class'].map(class_map)

    #读取图片到张量
    #找到待预测文件夹的路径
    images_folder_path=os.path.join(folderpath,'image')
    #按照gt_box的顺序排列文件地址并保存到filepaths里
    image_paths=[os.path.join(images_folder_path,i) for i in data['filename']]
    dataset=MyDataset(image_paths,data[['xmin','ymin','xmax','ymax']],data['class'],class_map,transform)
    dataloader=DataLoader(dataset,batch_size=64,shuffle=True)#得到(bs,channels,height,weight)
    
    model=ObjectDetection(class_num=3,in_channels=2048,H=7,W=7)
    model.load_state_dict(torch.load(load_path))
    model.to(device)
    model.eval()
    
    for images,targets in tqdm(dataloader): 
        results=[]
        reg_pred,cls_pred=model(images)
        cls_pred=F.softmax(cls_pred,dim=1)
        box=center_to_box(reg_pred)
        for i in range(len(reg_pred)):
            results.append({
                            'filepath':targets['filepath'][i],
                            'gt_box':targets['gt_box'][i],
                            'box':box[i],
                            'score':cls_pred[i][cls_pred[i].argmax().item()],
                            'class_name':class_map_t[cls_pred[i].argmax().item()]
                        })
        
    visual(results)

if __name__=='__main__':
    folderpath=r'object_detection_data(2)'
    load_path=r'version3/model6.path'
    main(folderpath,load_path)