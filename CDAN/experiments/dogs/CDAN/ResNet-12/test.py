import sys
import os
import torch
import yaml
sys.path.append('../../../../')
from models.CDAN import CDAN
from utils import util
from trainers.eval import meta_test


with open('../../../../config.yml', 'r') as f:
    temp = yaml.safe_load(f)

data_path = os.path.abspath(temp['data_path'])
test_path = os.path.join(data_path,'dogs/test')
model_path = './model_ResNet-12.pth'
gpu = 2
torch.cuda.set_device(gpu)

model = CDAN(resnet=True)
model.cuda()
model.load_state_dict(torch.load(model_path,map_location=util.get_device_map(gpu)),strict=True)
model.eval()

with torch.no_grad():
    way = 5
    for shot in [5]:
        mean,interval = meta_test(data_path=test_path,
                                model=model,
                                way=way,
                                shot=shot,
                                pre=False,
                                transform_type=0,
                                trial=100)
        print('%d-way-%d-shot acc: %.3f\t%.3f'%(way,shot,mean,interval))