import os
from collections import Counter

from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, precision_recall_curve, \
    average_precision_score
from scipy.interpolate import interp1d
from inspect import signature
from scipy.optimize import brentq
import numpy as np
import matplotlib

matplotlib.use('AGG')
import matplotlib.pyplot as plt
from utlis.file import mkdir


def roc_auc(labels, scores, pos_label=1, show=False, path=None):
    """Compute ROC curve and ROC area for each class"""
    # True/False Positive Rates.
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)
    if show:
        # Equal Error Rate
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        plt.figure()
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        lw = 2
        plt.plot(fpr,
                 tpr,
                 color='darkorange',
                 lw=lw,
                 label='(AUC = %0.2f, EER = %0.2f)' % (roc_auc, eer))
        plt.plot([eer], [1 - eer], marker='o', markersize=5, color="navy")
        plt.plot([0, 1], [1, 0], color='navy', lw=1, linestyle=':')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        if path:
            mkdir(os.path.dirname(path))
            plt.savefig(path + "_roc_auc.png")
        plt.show()
    #         plt.close()
    return {'roc_auc': roc_auc}


def MultiClassROCAUC(labels, scores, show=False, path=None):
    # LABELS = ['BENIGN', 'Bot', 'DDoS', 'DoS GoldenEye', 'DoS Hulk', 'DoS Slowhttptest',
    #           'DoS slowloris', 'FTP-Patator', 'Heartbleed', 'Infiltration', 'PortScan',
    #           'SSH-Patator', 'Web Attack Brute Force', 'Web Attack Sql Injection', 'Web Attack XSS']
    # LABELS = ['BENIGN', 'Bot', 'DDoS', 'DoS GoldenEye', 'DoS Hulk', 'DoS Slowhttptest',
    #           'DoS slowloris', 'FTP-Patator', 'PortScan',
    #           'SSH-Patator', 'Web Attack']
    LABELS = ['BENIGN', 'Bot', 'DDoS', 'DoS GoldenEye', 'DoS Hulk', 'DoS Slowhttptest',
              'DoS slowloris', 'FTP-Patator', 'PortScan',
              'SSH-Patator', 'Web Attack', 'Heartbleed', 'Infiltration']
    classes = Counter(labels)
    label_true = labels[labels == 0]
    score_true = scores[labels == 0]
    with open(path + '.log', mode='w+') as f:
        for i in range(0, len(classes)):
            if i == 0:
                label_true = labels[labels > 0]
                score_true = scores[labels > 0]
            label_false = labels[labels == i]
            score_false = scores[labels == i]
            label = np.concatenate([label_true, label_false])
            score = np.concatenate([score_true, score_false])
            roc = roc_auc(label, score, pos_label=i, show=show, path=path + '_class%s' % (i))
            f.write(LABELS[i] + ': ' + str(roc['roc_auc']) + '\tCount: ' + str(len(label_false)))
            f.write('\n')
            if i == 0:
                label_true = labels[labels == 0]
                score_true = scores[labels == 0]


def pre_rec_curve(labels, scores, show=False, path=None):
    average_precision = average_precision_score(labels, scores)
    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    if show:
        precision, recall, _ = precision_recall_curve(labels, scores)
        step_kwargs = ({
                           'step': 'post'
                       } if 'step' in signature(plt.fill_between).parameters else {})
        plt.figure()
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall,
                         precision,
                         alpha=0.2,
                         color='b',
                         **step_kwargs)

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
            average_precision))
        if path:
            mkdir(os.path.dirname(path))
            plt.savefig(path + "_pre_rec_curve.png")
        plt.show()
        # plt.close()
    return {'average_precision': average_precision}
