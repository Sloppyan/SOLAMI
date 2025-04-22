import os
import numpy as np
from tqdm import tqdm
from datasets.HumanML3D import HumanML3D_ActionDatabase
from datasets.DLPMoCap import DLPMoCap_ActionDatabase
from llm_api import get_embedding
from datasets.Unified import UnifiedDataset



def get_datasets(
    dataset_names=[],
    data_configs={},
    logger=None,
):
    configs_tmp = {}
    for dataset_name in dataset_names:
        if dataset_name in data_configs.keys():
            configs_tmp[dataset_name] = data_configs[dataset_name]
        else:
            logger.info('Invalid dataset name: ', dataset_name)
    
    datasets = UnifiedDataset(configs_tmp, logger=logger)
    # text_to_actions_dicts = dataset.text_to_actions
    # text_embeddings_dicts = dataset.text_embeddings
    # return text_to_actions_dicts, text_embeddings_dicts
    return datasets


def check_allowed_actions(text_to_actions_dicts, text_embeddings_dicts, interaction_allowed):
    for dataset_name in text_to_actions_dicts.keys():
        for text, actions in text_to_actions_dicts[dataset_name].items():
            for action in actions:
                if action.action_type not in interaction_allowed:
                    text_to_actions_dicts[dataset_name][text].remove(action)
            if len(text_to_actions_dicts[dataset_name][text]) == 0:
                del text_to_actions_dicts[dataset_name][text]
                del text_embeddings_dicts[dataset_name][text]
        if len(text_to_actions_dicts[dataset_name]) == 0:
            del text_to_actions_dicts[dataset_name]
            del text_embeddings_dicts[dataset_name]


def get_dataset_item_weights(text_to_actions_dicts, dataset_weights):
    dataset_item_weights = {}
    tmp_ = {}
    for dataset_name in text_to_actions_dicts.keys():
        if dataset_name not in tmp_:
            tmp_[dataset_name] = 0
        for text, actions in text_to_actions_dicts[dataset_name].items():
            tmp_[dataset_name] += len(actions)
    for key in tmp_.keys():
        dataset_item_weights[key] = dataset_weights[key] / tmp_[key]
    total = sum(dataset_item_weights.values())
    for key in dataset_item_weights.keys():
        dataset_item_weights[key] = dataset_item_weights[key] / total
    return dataset_item_weights
