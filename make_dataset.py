import pydicom
import numpy as np
import os
from tqdm import tqdm
import json
from PIL import Image
import torchvision

for i in tqdm(range(8), desc='============레이블================'):
    X = []
    y = []
    views = []
    path = f'../../202101n063/063.심장질환 진단을 위한 심초음파 데이터/01.데이터/1.Training/라벨데이터/{i}'
    for p_id in tqdm(os.listdir(path), desc='환자id'):
        # 0/0001, 0/0002, 0/0003, ...
        path2 = path + f'/{p_id}'
        for ptype in os.listdir(path2):
            # ptype = '../../202101n063/063.심장질환 진단을 위한 심초음파 데이터/01.데이터/1.Training/라벨데이터/0/0111231/echoAnnotation'
            if ptype == 'echoAnnotation':
                for data in os.listdir(path2+'/echoAnnotation/'):
                    # data = 0111231_echoAnnotation_001.json
                    try:
                        q = path2+'/echoAnnotation/' + f'/{data}'
                        idxes = set()
                        with open(q, 'r') as f:
                            j = json.load(f)
                        view = j["view_category"]
                        for labels in j['labels']:
                            idxes.add(labels['frame_idx'])
                        one1000_path = f'../../202101n063/063.심장질환 진단을 위한 심초음파 데이터/01.데이터/1.Training/원천데이터/{i}/{p_id}/echoAnnotation/'
                        one1000_dcm = data[:-4] + 'dcm'
                        dcm_array = pydicom.dcmread(one1000_path + one1000_dcm).pixel_array
                        for idx in idxes:
                            img = dcm_array[idx-1]
                            img = Image.fromarray(img.astype('uint8'))
                            img = torchvision.transforms.Resize((224,224))(img)
                            img = np.array(img) / 255.0
                            X.append(img)
                            y.append(i)
                            views.append(view)
                    except:
                        print('=====================================',path2 + f'/{ptype}/{data}', '===========================================')
    try:
        X_train = np.array(X)
        X_train = np.transpose(X_train, (0,3,1,2))
        X_train = X_train.astype(np.float32)
        y_train = np.array(y)
        view_train = np.array(views)
        np.save(f'./X_train_R{i}', X_train)
        np.save(f'./y_train_R{i}', y_train)
        np.save(f'./v_train_R{i}', view_train)
    except:
        print('Error occured while saving')

for i in tqdm(range(8), desc='============레이블================'):
    X = []
    y = []
    views = []
    path = f'../../202101n063/063.심장질환 진단을 위한 심초음파 데이터/01.데이터/2.Validation/라벨데이터/{i}'
    for p_id in tqdm(os.listdir(path), desc='환자id'):
        # 0/0001, 0/0002, 0/0003, ...
        path2 = path + f'/{p_id}'
        for ptype in os.listdir(path2):
            # ptype = '../../202101n063/063.심장질환 진단을 위한 심초음파 데이터/01.데이터/1.Training/라벨데이터/0/0111231/echoAnnotation'
            if ptype == 'echoAnnotation':
                for data in os.listdir(path2+'/echoAnnotation/'):
                    # data = 0111231_echoAnnotation_001.json
                    try:
                        q = path2+'/echoAnnotation/' + f'/{data}'
                        idxes = set()
                        with open(q, 'r') as f:
                            j = json.load(f)
                        view = j["view_category"]
                        for labels in j['labels']:
                            idxes.add(labels['frame_idx'])
                        one1000_path = f'../../202101n063/063.심장질환 진단을 위한 심초음파 데이터/01.데이터/2.Validation/원천데이터/{i}/{p_id}/echoAnnotation/'
                        one1000_dcm = data[:-4] + 'dcm'
                        dcm_array = pydicom.dcmread(one1000_path + one1000_dcm).pixel_array
                        for idx in idxes:
                            img = dcm_array[idx-1]
                            img = Image.fromarray(img.astype('uint8'))
                            img = torchvision.transforms.Resize((224,224))(img)
                            img = np.array(img) / 255.0
                            X.append(img)
                            y.append(i)
                            views.append(view)
                    except:
                        print('=====================================',path2 + f'/{ptype}/{data}', '===========================================')
    try:
        X_train = np.array(X)
        X_train = np.transpose(X_train, (0,3,1,2))
        X_train = X_train.astype(np.float32)
        y_train = np.array(y)
        view_train = np.array(views)
        np.save(f'./X_test_R{i}', X_train)
        np.save(f'./y_test_R{i}', y_train)
        np.save(f'./v_test_R{i}', view_train)
    except:
        print('Error occured while saving')