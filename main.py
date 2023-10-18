from model.t5_for_hierarchical_text_classification import T5ForSequenceClassification
import torch
import json
from target_generator.target_sequence import remove_reserved_chars, remove_reserved_chars_in_map
from torch.utils.data import DataLoader
from dataset.dataset import HTCDataset
from train_model import train_model
import argparse
from test_model import test_model
from constrained_module.constrain_utils import encode_text_data

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate.')
parser.add_argument('--data', type=str, default='WebOfScience',
                    choices=['WebOfScience', 'BlurbGenreCollection', 'rcv1v2'],
                    help='Dataset.')
parser.add_argument('--batch', type=int, default=16, help='Batch size.')
parser.add_argument('--target_sequence', type=str, default="dfs", choices=['dfs', 'bfs', 'bt_bfs', 'bt_dfs'],
                    help='Gradient accumulate steps')
parser.add_argument('--is_constrained', action='store_true',
                    help='Rather to constrain or not the predictions. Available only for dfs and bfs')
parser.add_argument('--num_epochs', default=12, type=int, help='Num epochs')
if __name__ == "__main__":

    args = parser.parse_args()
    print(args)
    target_type = args.target_sequence
    is_constrained = args.is_constrained
    num_epochs = args.num_epochs
    batch_size = args.batch
    data_name = args.data
    root_data = "dataset/"
    max_target_length = 25
    if data_name == "WebOfScience":
        file_path_save = data_name + "_" + target_type + "_constrained_" + str(is_constrained)
        train_data = root_data + data_name + "/wos_train.json"
        val_data = root_data + data_name + "/wos_val.json"
        test_data = root_data + data_name + "/wos_test.json"
    elif data_name == "BlurbGenreCollection":
        file_path_save = data_name + "_" + target_type + "_constrained_" + str(is_constrained)
        train_data = root_data + data_name + "/bgc_train.json"
        val_data = root_data + data_name + "/bgc_val.json"
        test_data = root_data + data_name + "/bgc_test.json"
        max_target_length = 100
    elif data_name == "rcv1v2":
        file_path_save = data_name + "_" + target_type + "_constrained_" + str(is_constrained)
        train_data = root_data + data_name + "/rcv1v2_train.json"
        val_data = root_data + data_name + "/rcv1v2_val.json"
        test_data = root_data + data_name + "/rcv1v2_test.json"
        max_target_length = 90
        # Initialize your model
    model = T5ForSequenceClassification(model="t5-base", target_type=target_type,
                                        max_target_length=max_target_length,
                                        constrained=is_constrained)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    tokenizer = model.tokenizer
    with open(root_data + data_name + "/label_data.json", "r", encoding="utf8") as f:
        elements = json.load(f)
        parent_childs = elements["parent_childs"]
        l1 = elements["level_1"]
        parent_childs["Root"] = l1
        all_labels = []
        per_level_size = []
        l1_labels, l2_labels, l3_labels, l4_labels = remove_reserved_chars(elements["level_1"], elements["level_2"],
                                                                           elements["level_3"], elements["level_4"])
    per_level_size.append(len(l1_labels))
    per_level_size.append(len(l2_labels))
    if l3_labels:
        per_level_size.append(len(l3_labels))
    if l4_labels:
        per_level_size.append(len(l4_labels))
    all_labels.extend(l1_labels)
    all_labels.extend(l2_labels)
    all_labels.extend(l3_labels)
    all_labels.extend(l4_labels)
    name_to_index = {all_labels[i]: i for i in range(len(all_labels))}
    set_all_labels = set(all_labels)
    parent_childs = remove_reserved_chars_in_map(parent_childs)
    encoded_data_map = encode_text_data(parent_childs, tokenizer, is_dfs=target_type == 'dfs')
    train_dataset = HTCDataset(train_data, parent_childs, None, target_type=model.target_type,
                               name_to_index=name_to_index)
    val_dataset = HTCDataset(val_data, parent_childs, None, target_type=model.target_type,
                             name_to_index=name_to_index)
    test_dataset = HTCDataset(test_data, parent_childs, None, target_type=model.target_type,
                              name_to_index=name_to_index)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    train_model(model, train_loader, val_loader, num_epochs=num_epochs, next_token_mapping=encoded_data_map,
                set_all_labels=set_all_labels,
                file_path_save=file_path_save, is_constrained=is_constrained)
    test_model(test_dataset, batch_size=batch_size, next_token_mapping=encoded_data_map, set_all_labels=set_all_labels,
               file_path_save=file_path_save,
               per_level_size=per_level_size)
