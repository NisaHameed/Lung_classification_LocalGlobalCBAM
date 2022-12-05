import time
# from kflod import kfold
import numpy as np
import random
from torch.optim.adam import Adam
from torch import optim
from resnet_attn_new import *
from torchvision.models.resnet import resnet50, resnet18
from torchvision.models.densenet import densenet121
from preprocessing import get_dataset3d
import sys

def expAllAtn(data_path):
    reset_rand()
    def model_opt():
        model = AllAtn()
        optm = Adam(model.parameters())
        return model, optm

    all_targets,  all_pred=kfold(data_path,
          256,
          50,
          model_optimizer=model_opt,
          loss=nn.BCELoss(),
          name='BasicResnet',
          device='cuda:0',
          deterministic=True
          )
    return all_targets,  all_pred

     
        

def expBasicResnet(data_path):
    reset_rand()

    def model_opt():
        model = BasicResnet()
        optm = Adam(model.parameters())
        return model, optm

    all_targets,  all_pred=kfold(data_path,
          256,
          50,
          model_optimizer=model_opt,
          loss=nn.BCELoss(),
          name='BasicResnet',
          device='cuda:0',
          deterministic=True
          )
    return all_targets,  all_pred


from torchsummary import summary

def expLocalGlobal(data_path):
    reset_rand()

    def model_opt():
        model = LocalGlobalNetwork()
        # summary(model, (1, 32, 32))
        optm=optim.RMSprop(model.parameters())
        # optm = Adam(model.parameters())
        return model, optm

    all_targets,  all_pred=kfold(data_path,
          256,
          50,
          model_optimizer=model_opt,
          loss=nn.BCELoss(),
          name='LocalGlobalNetwork',
          device='cuda:0',
          deterministic=True
          )
    return all_targets,  all_pred


def expLocalGlobal_old(data_path):
    reset_rand()

    def model_opt():
        model = LocalGlobalNetwork_old()
        # summary(model, (1, 32, 32))
        optm = Adam(model.parameters())
        return model, optm

    all_targets,  all_pred=kfold(data_path,
          256,
          50,
          model_optimizer=model_opt,
          loss=nn.BCELoss(),
          name='LocalGlobalNetwork_old',
          device='cuda:0',
          deterministic=True
          )
    return all_targets,  all_pred




def expResnetTrans(data_path):
    reset_rand()

    def model_opt():
        model = resnet50(pretrained=True)
        model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 1),
            nn.Sigmoid()
        )

        optm = Adam(model.fc.parameters())
        return model, optm

    all_targets,  all_pred=kfold(data_path,
          256,
          50,
          model_optimizer=model_opt,
          loss=nn.BCELoss(),
          name='ResnetTrans',
          device='cuda:0',
          deterministic=True,
          dataset_func=get_dataset3d
          )
    return all_targets,  all_pred

def expDensenetTrans(data_path):
    reset_rand()

    def model_opt():
        model = densenet121(pretrained=True)
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier.in_features, 1),
            nn.Sigmoid()
        )

        optm = Adam(model.classifier.parameters())
        return model, optm

    all_targets,  all_pred=kfold(data_path,
          256,
          50,
          model_optimizer=model_opt,
          loss=nn.BCELoss(),
          name='DensenetTrans',
          device='cuda:0',
          deterministic=True,
          #dataset_func=get_dataset3d
          )
    return all_targets,  all_pred


def expResnet18Trans(data_path):
    reset_rand()

    def model_opt():
        model = resnet18(pretrained=True)
        
        model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 1),
            nn.Sigmoid()
        )

        optm = Adam(model.fc.parameters())
        return model, optm

    all_targets,  all_pred=kfold(data_path,
          256,
          50,
          model_optimizer=model_opt,
          loss=nn.BCELoss(),
          name='Resnet18Trans',
          device='cuda:0',
          deterministic=True,
          dataset_func=get_dataset3d
          )
    return all_targets,  all_pred


def print_error():
    print('python <model_name> <data_path>')
    print('here is a list of experiments names:')
    for name in experiments.keys():
        print(name)






from trainer import Trainer
from preprocessing import get_dataset
from os import path
from torch.utils.data import DataLoader
import torch as T
import numpy as np
import random
from sklearn import metrics
from os import path
import matplotlib.pyplot as plt

def get_metrics(target, pred):
    prec, recall, f1, _ = metrics.precision_recall_fscore_support(target, pred>0.5, average='binary')
    accu=metrics.accuracy_score(target, pred>0.5)
    fpr, tpr, thresholds = metrics.roc_curve(target, pred)
    auc = metrics.auc(fpr, tpr)
    return prec, recall, auc, f1, accu


def calc_accuracy(x, y):
    x_th = (x > 0.5).long()
    matches = x_th == y.long()
    return matches

def reset_rand():
    seed = 1000
    T.manual_seed(seed)
    T.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def kfold(src_path,
          batch_size,
          n_epochs,
          model_optimizer,
          loss,
          name,
          device,
          deterministic=False,
          parallel=False,
          dataset_func=get_dataset):

    print('------Experiment:',name,'------')
    all_pred = T.zeros(849)
    all_targets = T.zeros(849)
    i =0
    f = open(path.join('results', '{name}.txt'), 'w')
    f.write('{batch_size} {n_epochs} {model_optimizer}\n')
    for fold in range(10):

        reset_rand()

        print('------------ fold' + str(fold+1) + ' ------------')
        f.write('------------ fold {fold+1} ------------\n')
        trset, testset = dataset_func(path.join(src_path,str(fold)))
        print('Training Size: ' + str(len(trset)) + ' Validation Size: '+ str(len(testset)))
        trset = DataLoader(trset, batch_size, shuffle=True)
        testset = DataLoader(testset, batch_size, shuffle=False)
        model,optimizer = model_optimizer()
        tr = Trainer(
            trset,
            testset,
            batch_size,
            n_epochs,
            model,
            optimizer,
            loss,
            '{name}_{fold}',
            device,
            deterministic,
            parallel
        )

        tr.run()

        pred, target = tr.predict()
        all_pred[i:i+pred.shape[0]] = pred
        all_targets[i:i+target.shape[0]] = target
        i += target.shape[0]

        prec, recall, auc, f1, accu = get_metrics(target, pred)
        print('AUC: ' + str(auc) + ', '  + 'Accuracy: ' + str(accu) + ', ' +  'precession: ' + str(prec) + ', ' + 'Recall: ' +  str(recall) + ', ' + 'F1score: ' +str(f1))
        f.write('AUC: ' + str(auc) + ', '+ 'Accuracy: ' + str(accu) + ', '  +  'precession: ' + str(prec) + ', ' + 'Recall: ' +  str(recall) + ', ' + 'F1score: ' +str(f1) +'\n')

    matches = calc_accuracy(all_pred, all_targets)
    acc = matches.float().mean()
    all_pred = all_pred.numpy()
    all_targets = all_targets.numpy()

    prec, recall, auc, f1, accu = get_metrics(all_targets, all_pred)
    print('Accuracy: ' + str(accu) + ', ' + 'AUC: ' + str(auc) + ', '  +  'precession: ' + str(prec) + ',' + 'Recall: ' +  str(recall)+ ', ' + 'f1score: ' + str(f1)+ '\n')
    f.write('accuray: {acc}, accuray1: {accu}, AUC: {auc}, precession: {prec}, Recall: {recall}, f1score: {f1}')
    result = {'all_pred': all_pred, 'all_targets': all_targets}
    T.save(result, path.join('results','{name}_result'))
    f.close()
    return all_targets,  all_pred

    





if __name__ == '__main__':
    experiments = {
        'Resnet18Trans' : expResnet18Trans,
        'ResnetTrans' : expResnetTrans,
        'LocalGlobal_cbam' : expLocalGlobal,
        'LocalGlobal':expLocalGlobal_old,
        'BasicResnet' : expBasicResnet,
        'AllAtn' : expAllAtn,
        'DensenetTrans' : expDensenetTrans
    }

    #################################
    exp_name = 'Resnet18Trans'
    data_path = '/home/nisa.hameed/Downloads/Lung_class/lidc_img'

    st = time.time()
    all_targets,  all_pred=experiments[exp_name](data_path)
    et = time.time()
    elapsed_time = et - st
    print(' Total execution time for ',exp_name,'is', elapsed_time, 'seconds')



    #plot visualization
    fpr, tpr, _ = metrics.roc_curve(all_targets,  all_pred)
    auc = metrics.roc_auc_score(all_targets, all_pred)
    p1=plt.plot(fpr,tpr,label="AUC="+str(auc))
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # # plt.legend(loc=4)
    # plt.show()


# ########################################
    exp_name = 'ResnetTrans'
    data_path = '/home/nisa.hameed/Downloads/Lung_class/lidc_img'

    st = time.time()
    all_targets,  all_pred=experiments[exp_name](data_path)
    et = time.time()
    elapsed_time = et - st
    print(' Total execution time for ',exp_name,'is', elapsed_time, 'seconds')

    
    fpr, tpr, _ = metrics.roc_curve(all_targets,  all_pred)
    auc = metrics.roc_auc_score(all_targets, all_pred)
    p2=plt.plot(fpr,tpr,label="AUC="+str(auc))
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
#     # plt.legend(loc=4)
#     # plt.show()

#     ###########################################
    exp_name = 'AllAtn'
    data_path = '/home/nisa.hameed/Downloads/Lung_class/lidc_img'

    st = time.time()
    all_targets,  all_pred=experiments[exp_name](data_path)
    et = time.time()
    elapsed_time = et - st
    print(' Total execution time for ',exp_name,'is', elapsed_time, 'seconds')

    #added for plot
    fpr, tpr, _ = metrics.roc_curve(all_targets,  all_pred)
    auc = metrics.roc_auc_score(all_targets, all_pred)
    p3=plt.plot(fpr,tpr,label="AUC="+str(auc))
#     # plt.ylabel('True Positive Rate')
#     # plt.xlabel('False Positive Rate')
#     # plt.legend(loc=4)
#     # # plt.show()

    ###########################################
    exp_name = 'LocalGlobal'
    data_path = '/home/nisa.hameed/Downloads/Lung_class/lidc_img'

    st = time.time()
    all_targets,  all_pred=experiments[exp_name](data_path)
    et = time.time()
    elapsed_time = et - st
    print(' Total execution time for ',exp_name,'is', elapsed_time, 'seconds')

    # #added for plot
    fpr, tpr, _ = metrics.roc_curve(all_targets,  all_pred)
    auc = metrics.roc_auc_score(all_targets, all_pred)
    p4=plt.plot(fpr,tpr,label="AUC="+str(auc))
   
  ###################################
    exp_name = 'LocalGlobal_cbam'
    data_path = '/home/nisa.hameed/Downloads/Lung_class/lidc_img'

    st = time.time()
    all_targets,  all_pred=experiments[exp_name](data_path)
    et = time.time()
    elapsed_time = et - st
    print(' Total execution time for ',exp_name,'is', elapsed_time, 'seconds')

    fpr, tpr, _ = metrics.roc_curve(all_targets,  all_pred)
    auc = metrics.roc_auc_score(all_targets, all_pred)
    p5=plt.plot(fpr,tpr,label="AUC="+str(auc))
    #visualization
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    # plt.legend((p1,p2,p3,p4,p5),('RESNET18','RESNET50','DENSENET121','LOCAL_GLOBAL','ATTENTION_LOCAL_GLOBAL'),loc='lower left')
    plt.legend((p4,p5),('LOCAL_GLOBAL','ATTENTION_LOCAL_GLOBAL'),loc=4)
    plt.show()