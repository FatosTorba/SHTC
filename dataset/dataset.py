from torch.utils.data import Dataset
import json
from target_generator.target_sequence import *
import torch


class HTCDataset(Dataset):
    def __init__(self, path, parent_child_map, task_prefix, target_type="dfs", name_to_index=None):
        self.path = path
        self.target_type = target_type
        self.task_prefix = task_prefix
        self.parent_child_map = parent_child_map
        self.child_parent_map = {v: k for k, value in self.parent_child_map.items() if k != "Root" for v in value}
        self.name_to_index = name_to_index
        self.index_parent_child_map = {
            self.name_to_index[k]: [self.name_to_index[v] for v in value] for k, value in
            parent_child_map.items() if k != "Root"}
        self.index_child_parent_map = {v: k for k, value in self.index_parent_child_map.items() for v in value}
        self.json_items = self.load_json_data()[0:2]
        self.input_data = self.build_input_data()
        self.target_data = self.build_target_data()
        self.target_data_multi_hot = self.build_target_data_multi_hot()

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, index):
        return self.input_data[index], self.target_data[index], self.target_data_multi_hot[index]

    def load_json_data(self):
        with open(self.path, "r", encoding="utf8") as f:
            return json.load(f)

    def build_input_data(self):
        return [item["description"] for item in self.json_items]

    def get_label_one_hot(self, all_labels):
        multi_hot = [0] * len(self.name_to_index)
        for label in all_labels:
            label = label.replace("-", "").replace("/", "")
            if label in self.name_to_index.keys():
                multi_hot[self.name_to_index[label]] = 1
        return multi_hot

    def build_target_data(self):
        if self.target_type == "dfs":
            return [dfs(self.parent_child_map, "Root",
                        item["level_1"],
                        item["level_2"],
                        item["level_3"],
                        item["level_4"]) for item in
                    self.json_items]
        elif self.target_type == "bfs":

            return [bfs(self.parent_child_map,
                        item["level_1"],
                        item["level_2"],
                        item["level_3"],
                        item["level_4"]) for
                    item
                    in
                    self.json_items]

        elif self.target_type == "bt_dfs":
            return [bottom_top_dfs(self.parent_child_map,
                                   item["level_1"],
                                   item["level_2"],
                                   item["level_3"],
                                   item["level_4"]) for
                    item
                    in
                    self.json_items]
        elif self.target_type == "bt_bfs":
            return [bottom_top_bfs(
                item["level_1"],
                item["level_2"],
                item["level_3"],
                item["level_4"]) for
                item
                in
                self.json_items]

    def map_name_to_index(self):
        name_to_index = {}
        i = 0
        for key, value in self.parent_child_map.items():
            if key == "Root":
                continue
            else:
                name_to_index[key] = i
                i += 1
        for key, value in self.parent_child_map.items():
            if key == "Root":
                continue
            else:
                for val in value:
                    if val not in name_to_index:
                        name_to_index[val] = i
                        i += 1
        return name_to_index

    def build_target_data_multi_hot(self):
        targets_multi_hot = []
        for item in self.json_items:
            all_labels = item["level_1"]
            all_labels.extend(item["level_2"])
            all_labels.extend(item["level_3"])
            all_labels.extend(item["level_4"])
            all_labels = [label.replace(" ", "") for label in all_labels]
            multi_hot = self.get_label_one_hot(all_labels)
            targets_multi_hot.append(multi_hot)
        return torch.tensor(targets_multi_hot)
