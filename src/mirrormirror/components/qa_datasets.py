from datasets import load_dataset, DatasetDict
import numpy as np 


ALLOWED_SPLIT_STRATEGIES = ['random']

def load_tofu_entity_split(forget_author_id, split_strategy='random', num_forget_datapoints=10):
    dataset = load_dataset("locuslab/tofu", "full")['train']
    author_id = np.repeat(np.arange(0, 200), 20)
    author_indexes = np.where((author_id == forget_author_id))[0]
    if split_strategy == 'random':
        np.random.seed(42)
        np.random.shuffle(author_indexes)
        forgotten_indexes = author_indexes[:num_forget_datapoints]
        retain_indexes = [x for x in range(len(dataset)) if x not in forgotten_indexes]
    else:
        raise NotImplementedError
    forget_set = dataset.select(forgotten_indexes)
    retain_set = dataset.select(retain_indexes)
    return DatasetDict({'forget': forget_set, 'retain': retain_set})
