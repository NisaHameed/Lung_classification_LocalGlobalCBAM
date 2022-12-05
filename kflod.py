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

    print('------Experiment------',name)
    all_pred = T.zeros(849)
    all_targets = T.zeros(849)
    i =0
    f = open(path.join('results', '{name}.txt'), 'w')
    f.write('{batch_size} {n_epochs} {model_optimizer}\n')
    for fold in range(1):

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


    #added for plot
    fpr, tpr, _ = metrics.roc_curve(all_targets,  all_pred)
    auc = metrics.roc_auc_score(all_targets, all_pred)
    plt.plot(fpr,tpr,label="AUC="+str(auc))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.show()



