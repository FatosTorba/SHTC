import torch
from tqdm import tqdm
import json
from torch.utils.data import DataLoader
from metrics.metric_utils import show_metrics


def test_model(test_dataset=None, batch_size=16, file_path_save=None, next_token_mapping=None,
               set_all_labels=None, per_level_size=None):
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model = torch.load(file_path_save)
    model.eval()
    results_as_text_to_save = {"Target": [], "Pred": []}
    labels_true = []
    labels_pred = []
    tokenizer = model.tokenizer
    with torch.no_grad():
        for inputs, targets, targets_multihot in tqdm(test_loader):
            inputs, targets = list(inputs), list(targets)
            preds = tokenizer.batch_decode(model.generate(inputs, next_token_mapping, set_all_labels=set_all_labels),
                                           skip_special_tokens=True)
            results_as_text_to_save["Target"].extend(targets)
            results_as_text_to_save["Pred"].extend(preds)
            for l in range(len(preds)):
                if preds[l] != targets[l]:
                    print("Pred: " + preds[l])
                    print("Target: " + targets[l])
            labels_true.extend(targets_multihot.tolist())
            labels_pred.extend([test_dataset.get_label_one_hot(pred.split()) for pred in preds])
    print('Calculate related metrics********************************************************************')
    show_metrics(labels_true, labels_pred, test_dataset.index_child_parent_map, per_level_size)
    with open(file_path_save + '_predictions_as_text.json', 'w') as fp:
        json.dump(results_as_text_to_save, fp)
