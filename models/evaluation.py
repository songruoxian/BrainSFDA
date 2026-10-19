import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score

def node_classification_evaluation_source_metapath(model, source_subnetwork_matrix, source_meta_path_matrices, source_feature, source_y_label, s_train_masks, s_val_masks, s_test_masks, device):
    model.eval()
    '''E2E'''
    encoder = model.encoder_su
    cls_model = model.cls_model_su
    encoder.to(device)
    cls_model.to(device)
    source_subnetwork_matrix = source_subnetwork_matrix.to(device)
    source_feature.to(device)
    source_y_label = source_y_label.to(device)

    with torch.no_grad():
        '''E2E'''
        rep_s = encoder(source_subnetwork_matrix, source_meta_path_matrices)
        pred = cls_model(rep_s)
        prob_su = torch.softmax(pred, dim=1)
        y_true = source_y_label.squeeze().long()
        preds = pred.max(1)[1].type_as(y_true)
        train_acc = accuracy_score(y_true[s_train_masks].cpu().numpy(),preds[s_train_masks].cpu().numpy())
        val_acc = accuracy_score(y_true[s_val_masks].cpu().numpy(),preds[s_val_masks].cpu().numpy())
        estp_test_acc = accuracy_score(y_true[s_test_masks].cpu().numpy(),preds[s_test_masks].cpu().numpy())
        estp_test_pre = precision_score(y_true[s_test_masks].cpu().numpy(), preds[s_test_masks].cpu().numpy(),average='macro',zero_division=1)
        estp_test_rec = recall_score(y_true[s_test_masks].cpu().numpy(), preds[s_test_masks].cpu().numpy(),average='macro')
        estp_test_f1 = f1_score(y_true[s_test_masks].cpu().numpy(), preds[s_test_masks].cpu().numpy(),average='macro')
        estp_test_auc = roc_auc_score(y_true[s_test_masks].cpu().numpy(), prob_su[s_test_masks, 1].cpu().numpy())
    return train_acc, val_acc, estp_test_acc, estp_test_pre, estp_test_rec, estp_test_f1, estp_test_auc

def node_classification_evaluation_sfda_metapath(model, target_subnetwork_matrix, target_meta_path_matrices, target_y_label, device):
    model.eval()
    '''E2E'''
    encoder = model.encoder_su2
    cls_model = model.cls_model_su

    encoder.to(device)
    cls_model.to(device)
    target_subnetwork_matrix = target_subnetwork_matrix.to(device)
    target_y_label = target_y_label.to(device)

    with torch.no_grad():
        '''E2E'''
        rep_s = encoder(target_subnetwork_matrix,target_meta_path_matrices)
        pred = cls_model(rep_s)
        prob_su = torch.softmax(pred, dim=1)
        y_true = target_y_label.squeeze().long()
        preds = pred.max(1)[1].type_as(y_true)

        estp_test_acc = accuracy_score(y_true.cpu().numpy(), preds.cpu().numpy())
        estp_test_pre = precision_score(y_true.cpu().numpy(), preds.cpu().numpy(), average='macro', zero_division=1)
        estp_test_rec = recall_score(y_true.cpu().numpy(), preds.cpu().numpy(), average='macro')
        estp_test_f1 = f1_score(y_true.cpu().numpy(), preds.cpu().numpy(), average='macro')
        estp_test_auc = roc_auc_score(y_true.cpu().numpy(), prob_su[:, 1].cpu().numpy())
    
    return estp_test_acc, estp_test_pre, estp_test_rec, estp_test_f1, estp_test_auc
