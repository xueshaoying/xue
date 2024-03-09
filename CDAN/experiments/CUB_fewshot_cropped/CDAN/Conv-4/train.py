import os
import sys
import torch
import yaml
from functools import partial
sys.path.append('../../../../')
from trainers import trainer, frn_train
from datasets import dataloaders
# from models.FRN import FRN
from models.CDAN import CDAN
#传参
args = trainer.train_parser()
with open('../../../../config.yml', 'r') as f:
    #读取配置文件
    temp = yaml.safe_load(f)
data_path = os.path.abspath(temp['data_path'])
fewshot_path = os.path.join(data_path,'CUB_fewshot_cropped')
#设置划分数据图片的路径 val_pre val train test_pre test
pm = trainer.Path_Manager(fewshot_path=fewshot_path,args=args)
#train_way=20，train_shot=5，test_query_shot=16，train_query_shot=15
train_way = args.train_way
shots = [args.train_shot, args.train_query_shot]
#按照way和shot数进行采样
train_loader = dataloaders.meta_train_dataloader(data_path=pm.train,
                                                way=train_way,
                                                shots=shots,
                                                transform_type=args.train_transform_type)

model = CDAN(way=train_way,
            shots=[args.train_shot, args.train_query_shot],
            resnet=args.resnet)


train_func = partial(frn_train.default_train,train_loader=train_loader)

tm = trainer.Train_Manager(args,path_manager=pm,train_func=train_func)

tm.train(model)

tm.evaluate(model)