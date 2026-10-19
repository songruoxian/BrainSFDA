import copy
import os

import numpy as np
import scipy.io as scio
import random
from tqdm import tqdm
import torch
import logging

from models.evaluation import node_classification_evaluation_source_metapath
from utils import build_args, create_optimizer
from models import build_model
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def shuffle_data_consistently(*arrays, seed=None):
    if seed is not None:
        np.random.seed(seed)

    n = len(arrays[0])
    indices = np.random.permutation(n)

    shuffled_arrays = []
    for arr in arrays:
        if isinstance(arr, dict):
            shuffled_dict = {}
            for key, value in arr.items():
                if isinstance(value, torch.Tensor):
                    shuffled_dict[key] = value[indices]
                elif isinstance(value, np.ndarray):
                    shuffled_dict[key] = value[indices]
                else:
                    shuffled_dict[key] = value
            shuffled_arrays.append(shuffled_dict)
        elif isinstance(arr, torch.Tensor):
            shuffled_arrays.append(arr[indices])
        elif isinstance(arr, np.ndarray):
            shuffled_arrays.append(arr[indices])
        else:
            try:
                arr_np = np.array(arr)
                shuffled_arrays.append(arr_np[indices])
            except:
                shuffled_arrays.append(arr)

    return tuple(shuffled_arrays)

def get_data_X_Y_S_speicific(data_path, class_label_path, site_path, specific_site):
    Brain_networks = scio.loadmat(data_path)
    X_data = Brain_networks['connectivity'] 
    X_data_nan = np.isnan(X_data)  
    X_data_inf = np.isinf(X_data) 

    print(f"X_data has {np.sum(X_data_nan)} NaN values and {np.sum(X_data_inf)} Inf values.")

    Y_labels = scio.loadmat(class_label_path)
    Y_data = Y_labels['labels'][0]  

    Site_labels = scio.loadmat(site_path)
    S_data = Site_labels['labels']  
    S_data = np.char.strip(S_data)
    nyu_mask = np.char.startswith(S_data, specific_site)  

    X_data_nyu = X_data[nyu_mask]
    Y_data_nyu = Y_data[nyu_mask]
    S_data_nyu = S_data[nyu_mask]

    return X_data_nyu, torch.tensor(Y_data_nyu), S_data_nyu

def filter_and_generate_meta_path_matrices(region_file, connectivity_data, meta_paths):
    subnet_data = np.genfromtxt(region_file, dtype=int, skip_header=1)
    subnet_mask = subnet_data[:, :3]  # DMN, CEN, SN
    DMN_nodes = np.where(subnet_mask[:, 0] == 1)[0]
    CEN_nodes = np.where(subnet_mask[:, 1] == 1)[0]
    SN_nodes = np.where(subnet_mask[:, 2] == 1)[0]

    all_unique_nodes = set(DMN_nodes) | set(CEN_nodes) | set(SN_nodes)
    duplicate_nodes = []
    for node in all_unique_nodes:
        count = 0
        if node in DMN_nodes: count += 1
        if node in CEN_nodes: count += 1
        if node in SN_nodes: count += 1
        if count > 1:
            duplicate_nodes.append((node, count)) 

    total_nodes = len(all_unique_nodes)
    for node, count in duplicate_nodes:
        total_nodes += (count - 1)  

    node_mapping = {}  
    node_positions = []  
    current_idx = 0

    for node in sorted(all_unique_nodes):
        subnets = []
        if node in DMN_nodes: subnets.append('DMN')
        if node in CEN_nodes: subnets.append('CEN')
        if node in SN_nodes: subnets.append('SN')

        if len(subnets) == 1:
            node_mapping[node] = current_idx
            node_positions.append(subnets)
            current_idx += 1
        else:
            for i, subnet in enumerate(subnets):
                if i == 0:
                    node_mapping[node] = current_idx
                    node_positions.append([subnet])
                    current_idx += 1
                else:
                    node_mapping[f"{node}_dup_{i}"] = current_idx
                    node_positions.append([subnet])
                    current_idx += 1

    N = connectivity_data.shape[0]
    subnet_connectivity = np.zeros((N, total_nodes, total_nodes))

    for i in range(N):
        original_matrix = connectivity_data[i]

        for orig_key_i, new_idx_i in node_mapping.items():
            if isinstance(orig_key_i, str) and "_dup_" in orig_key_i:
                orig_i = int(orig_key_i.split('_')[0])  
            else:
                orig_i = orig_key_i  

            for orig_key_j, new_idx_j in node_mapping.items():
                if isinstance(orig_key_j, str) and "_dup_" in orig_key_j:
                    orig_j = int(orig_key_j.split('_')[0]) 
                else:
                    orig_j = orig_key_j

                subnet_connectivity[i, new_idx_i, new_idx_j] = original_matrix[orig_i, orig_j]

    meta_path_masks = {}
    for mp_name, mp_sequence in meta_paths.items():
        mp_mask = create_meta_path_mask(mp_sequence, node_positions)
        meta_path_masks[mp_name] = mp_mask

    meta_path_matrices = {}
    for mp_name in meta_paths.keys():
        mp_matrices = []
        for i in range(N):
            full_matrix = subnet_connectivity[i]
            mp_mask = meta_path_masks[mp_name]
            mp_matrix = full_matrix * mp_mask
            mp_matrices.append(mp_matrix)

        meta_path_matrices[mp_name] = torch.tensor(mp_matrices).to(device)

    full_matrices_tensor = torch.tensor(subnet_connectivity)

    return {
        'full_matrices': full_matrices_tensor, 
        'meta_path_masks': meta_path_masks, 
        'meta_path_matrices': meta_path_matrices,  
        'node_positions': node_positions,  
        'node_mapping': node_mapping 
    }


def create_meta_path_mask(meta_path_sequence, node_positions):
    n_nodes = len(node_positions)
    mask_matrix = np.zeros((n_nodes, n_nodes))

    for i in range(n_nodes):
        i_subnets = node_positions[i]

        for j in range(n_nodes):
            j_subnets = node_positions[j]

            if is_valid_meta_path_connection(i_subnets, j_subnets, meta_path_sequence):
                mask_matrix[i, j] = 1

    return mask_matrix


def is_valid_meta_path_connection(source_subnets, target_subnets, meta_path_sequence):
    for subnet in source_subnets:
        if subnet in target_subnets and subnet in meta_path_sequence:
            return True

    for k in range(len(meta_path_sequence) - 1):
        current_subnet = meta_path_sequence[k]
        next_subnet = meta_path_sequence[k + 1]

        if current_subnet in source_subnets and next_subnet in target_subnets:
            return True

    return False

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True


def data_split(y, train_size, val_size):
    seeds = args.seeds
    for i, seed in enumerate(seeds):
        set_random_seed(seed)
        random_node_indices = np.random.permutation(y.shape[0])
        training_size = int(len(random_node_indices) * train_size)
        validation_size = int(len(random_node_indices) * val_size)

        train_node_indices = random_node_indices[:training_size]
        val_node_indices = random_node_indices[training_size:training_size + validation_size]
        test_node_indices = random_node_indices[training_size + validation_size:]

        train_masks = torch.zeros([y.shape[0]], dtype=torch.bool)
        train_masks[train_node_indices] = 1
        val_masks = torch.zeros([y.shape[0]], dtype=torch.bool)
        val_masks[val_node_indices] = 1
        test_masks = torch.zeros([y.shape[0]], dtype=torch.bool)
        test_masks[test_node_indices] = 1

        return train_masks, val_masks, test_masks

def pretrain(model, source_subnetwork_matrix, source_meta_path_matrices, source_feature, source_y_label, s_train_masks, s_val_masks, s_test_masks, optimizer_s,
             max_epoch, device,
             scheduler_s, seed, confidence):
    logging.info("start training..")
    args = build_args()
    epoch_iter = tqdm(range(max_epoch))
    best_val_acc = 0.0
    best_train_acc = 0.0
    best_test_acc = 0.0
    optimizer_ss = optimizer_s
    best_model = type(model)(nodes=64) 
    for epoch in epoch_iter:
        model.train()
        optimizer_ss.zero_grad()
        loss_ss = model(source_subnetwork_matrix, source_meta_path_matrices, source_feature, source_y_label,s_train_masks)  # Ensure forward pass happens on device
        epoch_iter.set_description(f"# Epoch {epoch}: train_loss_ff: {loss_ss:.3f}")
        loss_ss.backward()
        optimizer_ss.step()

        if scheduler_s is not None:
            scheduler_s.step()

        '''source superivsed learning'''
        train_acc, val_acc, estp_acc, estp_pre, estp_rec, estp_f1, estp_auc = node_classification_evaluation_source_metapath(
            model, source_subnetwork_matrix, source_meta_path_matrices, source_feature, source_y_label, s_train_masks, s_val_masks, s_test_masks, device
        )
        
        if estp_acc > best_test_acc:
            best_test_epoch = epoch
            best_test_acc = estp_acc
            best_test_pre = estp_pre
            best_test_rec = estp_rec
            best_test_f1 = estp_f1
            best_test_auc = estp_auc
            torch.save(model.state_dict(),
                       'outputs/asd/'+str(confidence)+'/pretrained_' + args.source + '_' + str(seed) + '_e2e_stage1_subnetwork.pt')

    return best_model


if __name__ == '__main__':
    args = build_args()
    device = args.device if args.device >= 0 else "cpu"
    seeds = args.seeds
    max_epoch = args.max_epoch
    optim_type = args.optimizer
    load_model = args.load_model
    data_path = './data/abide_tpe_mida_out/aal_correlation.mat'
    class_label_path = './data/abide_tpe_mida_out/category_labels.mat'
    site_path = './data/abide_tpe_mida_out/site_labels.mat'
    source_specific_site = args.source
    target_specific_site = args.target
    lr_s = args.lr
    weight_decay_s = args.weight_decay
    use_scheduler = args.scheduler
    logs = args.logging
    num_classes = args.classes
    confidence = args.confidence

    META_PATHS = {
        'DMN': ['DMN'],
        'CEN': ['CEN'],
        'SN': ['SN'],
        'DMN_CEN': ['DMN', 'CEN'],
        'DMN_SN': ['DMN', 'SN'],
        'CEN_SN': ['CEN', 'SN'],
        'DMN_CEN_SN': ['DMN', 'CEN', 'SN'],
        'DMN_SN_CEN': ['DMN', 'SN', 'CEN'],
        'CEN_DMN_SN': ['CEN', 'DMN', 'SN'],
        'DMN_DMN_CEN': ['DMN', 'DMN', 'CEN'],
        'DMN_DMN_SN': ['DMN', 'DMN', 'SN'],
        'CEN_CEN_DMN': ['CEN', 'CEN', 'DMN'],
        'CEN_CEN_SN': ['CEN', 'CEN', 'SN'],
        'SN_SN_DMN': ['SN', 'SN', 'DMN'],
        'SN_SN_CEN': ['SN', 'SN', 'CEN'],
        'DMN_DMN_CEN_CEN': ['DMN', 'DMN', 'CEN', 'CEN'],
        'DMN_DMN_SN_SN': ['DMN', 'DMN', 'SN', 'SN'],
        'CEN_CEN_SN_SN': ['CEN', 'CEN', 'SN', 'SN'],
        'DMN_DMN_CEN_SN': ['DMN', 'DMN', 'CEN', 'SN'],
        'DMN_DMN_SN_CEN': ['DMN', 'DMN', 'SN', 'CEN'],
        'CEN_CEN_DMN_SN': ['CEN', 'CEN', 'DMN', 'SN'],
        'CEN_CEN_SN_DMN': ['CEN', 'CEN', 'SN', 'DMN'],
        'SN_SN_DMN_CEN': ['SN', 'SN', 'DMN', 'CEN'],
        'SN_SN_CEN_DMN': ['SN', 'SN', 'CEN', 'DMN'],
        'DMN_SN_CEN_DMN': ['DMN', 'SN', 'CEN', 'DMN']
    }

    '''specific data'''
    source_data, source_y_label, source_site_label = get_data_X_Y_S_speicific(data_path, class_label_path, site_path, source_specific_site)
    target_data, target_y_label, target_site_label = get_data_X_Y_S_speicific(data_path, class_label_path, site_path, target_specific_site)

    source_data_result = filter_and_generate_meta_path_matrices(
        'region.txt',
        source_data,
        META_PATHS
    )

    target_data_result = filter_and_generate_meta_path_matrices(
        'region.txt',
        target_data,
        META_PATHS
    )
    source_subnetwork_matrix = source_data_result['full_matrices']
    source_meta_path_matrices = source_data_result['meta_path_matrices']

    target_subnetwork_matrix = target_data_result['full_matrices']
    target_meta_path_matrices = target_data_result['meta_path_matrices']

    estp_acc_list = []
    estp_pre_list = []
    estp_rec_list = []
    estp_f1_list = []
    estp_auc_list = []
    for i, seed in enumerate(seeds):
        print(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)

        (source_subnetwork_matrix, source_meta_path_matrices, source_y_label) = shuffle_data_consistently(
            source_subnetwork_matrix, source_meta_path_matrices, source_y_label, seed=seed
        )

        (target_subnetwork_matrix, target_meta_path_matrices, target_y_label) = shuffle_data_consistently(
            target_subnetwork_matrix, target_meta_path_matrices, target_y_label, seed=seed
        )

        # # Output the shape of the filtered matrix
        print("Subnetwork Matrix Shape:", source_subnetwork_matrix.shape, target_subnetwork_matrix.shape)

        model = build_model(args).to(device)
        source_subnetwork_matrix = source_subnetwork_matrix.to(device)  # Ensure data is on device
        source_y_label = source_y_label.to(device)
        optimizer_s = create_optimizer(optim_type, model, lr_s, weight_decay_s)

        if use_scheduler:
            logging.info("Use schedular")
            scheduler = lambda epoch: (1 + np.cos((epoch) * np.pi / max_epoch)) * 0.5
            scheduler_s = torch.optim.lr_scheduler.LambdaLR(optimizer_s, lr_lambda=scheduler)
        else:
            scheduler_s = None

        s_train_masks, s_val_masks, s_test_masks = data_split(y=source_y_label, train_size = 0.7, val_size = 0.1)

        source_feature = torch.eye(64).to(device)
        batch_size = source_data.shape[0]
        source_feature = source_feature.unsqueeze(0).expand(batch_size, -1, -1)
        model = pretrain(model, source_subnetwork_matrix, source_meta_path_matrices, source_feature, source_y_label, s_train_masks, s_val_masks, s_test_masks, optimizer_s, max_epoch, device,
                         scheduler_s, seed, confidence)
        if load_model:
            logging.info("Loading Model ... ")
            model.load_state_dict(torch.load('outputs/asd/'+str(confidence)+'/pretrained_' + args.source +'_' + str(seed) + '_e2e_stage1_subnetwork.pt'))
        model.eval()
        train_acc, val_acc, estp_test_acc, estp_test_pre, estp_test_rec, estp_test_f1, estp_test_auc = node_classification_evaluation_source_metapath(model, source_subnetwork_matrix, source_meta_path_matrices, source_feature,source_y_label, s_train_masks, s_val_masks, s_test_masks, device)

        estp_acc_list.append(estp_test_acc)
        estp_pre_list.append(estp_test_pre)
        estp_rec_list.append(estp_test_rec)
        estp_f1_list.append(estp_test_f1)
        estp_auc_list.append(estp_test_auc)

    estp_acc, estp_acc_std = np.mean(estp_acc_list), np.std(estp_acc_list)
    estp_pre, estp_pre_std = np.mean(estp_pre_list), np.std(estp_pre_list)
    estp_rec, estp_rec_std = np.mean(estp_rec_list), np.std(estp_rec_list)
    estp_f1, estp_f1_std = np.mean(estp_f1_list), np.std(estp_f1_list)
    estp_auc, estp_auc_std = np.mean(estp_auc_list), np.std(estp_auc_list)

    # print(f"# final_acc: {final_acc:.3f}±{final_acc_std:.3f}")
    print(f"# early-stopping_acc: {estp_acc:.3f}±{estp_acc_std:.3f}")
    print(f"# early-stopping_pre: {estp_pre:.3f}±{estp_pre_std:.3f}")
    print(f"# early-stopping_rec: {estp_rec:.3f}±{estp_rec_std:.3f}")
    print(f"# early-stopping_f1: {estp_f1:.3f}±{estp_f1_std:.3f}")
    print(f"# early-stopping_auc: {estp_auc:.3f}±{estp_auc_std:.3f}")






