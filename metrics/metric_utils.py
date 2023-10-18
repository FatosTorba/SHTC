import warnings
from sklearn.metrics import f1_score


def micro_f1_global_and_per_level(targets, preds, per_level_size):
    micro_f1_global = f1_score(targets, preds, average='micro')
    targets_per_level = [split_list_by_given_sizes(target, per_level_size) for target in targets]
    preds_per_level = [split_list_by_given_sizes(pred, per_level_size) for pred in preds]
    micro_f1_per_level = [
        f1_score([target_per_level[j] for target_per_level in targets_per_level],
                 [pred_per_level[j] for pred_per_level in preds_per_level],
                 average='micro') for j in range(len(per_level_size))]
    return {"micro_f1_global": micro_f1_global, "micro_f1_per_level": micro_f1_per_level}


def macro_f1_global_and_per_level(targets, preds, per_level_size):
    macro_f1_global = f1_score(targets, preds, average='macro')
    targets_per_level = [split_list_by_given_sizes(target, per_level_size) for target in targets]
    preds_per_level = [split_list_by_given_sizes(pred, per_level_size) for pred in preds]
    macro_f1_per_level = [
        f1_score([target_per_level[j] for target_per_level in targets_per_level],
                 [pred_per_level[j] for pred_per_level in preds_per_level],
                 average='macro') for j in range(len(per_level_size))]
    return {"macro_f1_global": macro_f1_global, "macro_f1_per_level": macro_f1_per_level}


def constrained_micro_f1_global_and_per_level(targets, preds, child_parent_indexes, per_level_size):
    preds = [constrained_output_with_respect_to_hierarchy(child_parent_indexes, pred) for pred in preds]
    micro_f1 = micro_f1_global_and_per_level(targets, preds, per_level_size)
    return {"c-micro_f1_global": micro_f1["micro_f1_global"], "c-micro_f1_per_level": micro_f1["micro_f1_per_level"]}


def constrained_macro_f1_global_and_per_level(targets, preds, child_parent_indexes, per_level_size):
    preds = [constrained_output_with_respect_to_hierarchy(child_parent_indexes, pred) for pred in preds]
    micro_f1 = macro_f1_global_and_per_level(targets, preds, per_level_size)
    return {"c-macro_f1_global": micro_f1["macro_f1_global"], "c-macro_f1_per_level": micro_f1["macro_f1_per_level"]}


def split_list_by_given_sizes(list, sizes):
    split_list = []
    start = 0
    for size in sizes:
        split_list.append(list[start:start + size])
        start += size
    return split_list


def depth_of_prediction_rate(list_targets, list_preds, per_level_size):
    per_level_list_targets = [split_list_by_given_sizes(target, per_level_size) for target in list_targets]
    per_level_list_preds = [split_list_by_given_sizes(pred, per_level_size) for pred in list_preds]
    respected_depth = [is_depth_respected(per_level_list_targets[i], per_level_list_preds[i]) for i in
                       range(len(per_level_list_targets))]
    return sum(respected_depth) / len(respected_depth)


def is_depth_respected(target_per_level, predicted_per_level):
    target_per_level_indexes = [get_prediction_indexes(target) for target in target_per_level]
    predicted_per_level_indexes = [get_prediction_indexes(predicted) for predicted in predicted_per_level]
    depth_target = get_depth(target_per_level_indexes)
    depth_predicted = get_depth(predicted_per_level_indexes)
    return depth_predicted >= depth_target


def get_depth(target_per_level_indexes):
    depth = len(target_per_level_indexes)
    while depth > 0:
        if target_per_level_indexes[depth - 1]:
            return depth
        else:
            depth += -1
    return depth


def constrained_output_with_respect_to_hierarchy(child_parent_indexes, list_predicted):
    list_predicted_constrained = [0 for i in range(len(list_predicted))]
    for e in range(len(list_predicted)):
        if list_predicted[e] >= 0.5:
            if e in child_parent_indexes.keys():
                if list_predicted[child_parent_indexes[e]] >= 0.5:
                    list_predicted_constrained[e] = 1
            else:
                list_predicted_constrained[e] = 1
    return list_predicted_constrained


def is_hiera_respected(predicted_child, predicted_parents, child_parent_indexes_map):
    parent = child_parent_indexes_map[predicted_child]
    if parent in predicted_parents:
        return 1
    return 0


def get_prediction_indexes(pred):
    return [v for v in range(len(pred)) if pred[v] >= 0.5]


def get_per_level_pred_or_target_indexes(pred, per_level_size, level):
    """
    level starts at 1
    """
    if level < 1:
        raise ValueError("Level starts at 1")
    return [i for i in
            filter(lambda i: i < sum(per_level_size[0:level]) and i >= sum(per_level_size[0:level - 1]), pred)]


def get_true_positive_pred(target_indexes, pred_indexes):
    """if an index is not in the target we remove"""
    return [index for index in filter(lambda index: index in target_indexes, pred_indexes)]


def get_false_positive_pred(target_indexes, pred_indexes):
    """if an index is  in the target we remove"""
    return [index for index in filter(lambda index: index not in target_indexes, pred_indexes)]


def compute_hiera_metrics(list_targets, list_preds, per_level_size, child_parent_indexes_map):
    list_targets_indexes = [get_prediction_indexes(target) for target in list_targets]
    list_preds_indexes = [get_prediction_indexes(pred) for pred in list_preds]
    level1_preds_indexes = [get_per_level_pred_or_target_indexes(pred, per_level_size, 1) for pred
                            in list_preds_indexes]

    level2_target_indexes = [get_per_level_pred_or_target_indexes(target, per_level_size, 2) for target
                             in list_targets_indexes]

    level2_preds_indexes = [get_per_level_pred_or_target_indexes(pred, per_level_size, 2) for pred
                            in list_preds_indexes]

    l1_l2_hiera_respected_rate = compute_hiera_respected_rate(level1_preds_indexes, level2_preds_indexes,
                                                              child_parent_indexes_map)
    print("L1/L2 hiera respected rate : " + str(l1_l2_hiera_respected_rate))

    l1_l2_true_positive_hiera_respected_rate = compute_hiera_respected_rate(level1_preds_indexes,
                                                                            [get_true_positive_pred(
                                                                                level2_target_indexes[i],
                                                                                level2_preds_indexes[i]) for i in range(
                                                                                len(level2_preds_indexes))],
                                                                            child_parent_indexes_map)

    print("L1/L2 true positive hiera respected rate : " + str(l1_l2_true_positive_hiera_respected_rate))

    l1_l2_false_positive_hiera_respected_rate = compute_hiera_respected_rate(level1_preds_indexes,
                                                                             [get_false_positive_pred(
                                                                                 level2_target_indexes[i],
                                                                                 level2_preds_indexes[i]) for i in
                                                                                 range(
                                                                                     len(level2_preds_indexes))],
                                                                             child_parent_indexes_map)

    print("L1/L2 false positive hiera respected rate : " + str(l1_l2_false_positive_hiera_respected_rate))

    if (len(per_level_size) > 2):
        level3_target_indexes = [get_per_level_pred_or_target_indexes(target, per_level_size, 3) for
                                 target
                                 in list_targets_indexes]
        level3_preds_indexes = [get_per_level_pred_or_target_indexes(pred, per_level_size, 3) for pred
                                in list_preds_indexes]

        l2_l3_hiera_respected_rate = compute_hiera_respected_rate(level2_preds_indexes, level3_preds_indexes,
                                                                  child_parent_indexes_map)
        print("L2/L3 hiera respected rate : " + str(l2_l3_hiera_respected_rate))

        l2_l3_true_positive_hiera_respected_rate = compute_hiera_respected_rate(level2_preds_indexes,
                                                                                [get_true_positive_pred(
                                                                                    level3_target_indexes[i],
                                                                                    level3_preds_indexes[i]) for i in
                                                                                    range(
                                                                                        len(level3_preds_indexes))],
                                                                                child_parent_indexes_map)

        print("L2/L3 true positive hiera respected rate : " + str(l2_l3_true_positive_hiera_respected_rate))

        l2_l3_false_positive_hiera_respected_rate = compute_hiera_respected_rate(level2_preds_indexes,
                                                                                 [get_false_positive_pred(
                                                                                     level3_target_indexes[i],
                                                                                     level3_preds_indexes[i]) for i in
                                                                                     range(
                                                                                         len(level3_preds_indexes))],
                                                                                 child_parent_indexes_map)

        print("L2/L3 false positive hiera respected rate : " + str(l2_l3_false_positive_hiera_respected_rate))

    if (len(per_level_size) > 3):
        level4_target_indexes = [get_per_level_pred_or_target_indexes(target, per_level_size, 4) for
                                 target
                                 in list_targets_indexes]
        level4_preds_indexes = [get_per_level_pred_or_target_indexes(pred, per_level_size, 4) for pred
                                in list_preds_indexes]
        l3_l4_hiera_respected_rate = compute_hiera_respected_rate(level3_preds_indexes, level4_preds_indexes,
                                                                  child_parent_indexes_map)
        print("L3/L4 hiera respected rate : " + str(l3_l4_hiera_respected_rate))

        l3_l4_true_positive_hiera_respected_rate = compute_hiera_respected_rate(level3_preds_indexes,
                                                                                [get_true_positive_pred(
                                                                                    level4_target_indexes[i],
                                                                                    level4_preds_indexes[i]) for i in
                                                                                    range(
                                                                                        len(level4_preds_indexes))],
                                                                                child_parent_indexes_map)

        print("L3/L4 true positive hiera respected rate : " + str(l3_l4_true_positive_hiera_respected_rate))

        l3_l4_false_positive_hiera_respected_rate = compute_hiera_respected_rate(level3_preds_indexes,
                                                                                 [get_false_positive_pred(
                                                                                     level4_target_indexes[i],
                                                                                     level4_preds_indexes[i]) for i in
                                                                                     range(
                                                                                         len(level4_preds_indexes))],
                                                                                 child_parent_indexes_map)

        print("L3/L4 false positive hiera respected rate : " + str(l3_l4_false_positive_hiera_respected_rate))


def compute_hiera_respected_rate(list_predicted_parents, list_predicted_child, child_parent_indexes_map):
    hiera_respected_rate = 0
    cnt = 0
    for i in range(len(list_predicted_child)):
        predicted_parents = list_predicted_parents[i]
        predicted_child = list_predicted_child[i]
        for child in predicted_child:
            cnt += 1
            hiera_respected_rate += is_hiera_respected(child, predicted_parents,
                                                       child_parent_indexes_map)
    if cnt != 0:
        return hiera_respected_rate / cnt
    warnings.warn("No child prediction in the given samples")
    return -1


def compute_f1_scores(list_target, list_preds_indexes, child_parent_indexes_map, per_level_size):
    print(macro_f1_global_and_per_level(list_target, list_preds_indexes, per_level_size))
    print(micro_f1_global_and_per_level(list_target, list_preds_indexes, per_level_size))
    print(constrained_macro_f1_global_and_per_level(list_target, list_preds_indexes, child_parent_indexes_map,
                                                    per_level_size))
    print(constrained_micro_f1_global_and_per_level(list_target, list_preds_indexes, child_parent_indexes_map,
                                                    per_level_size))


def show_metrics(list_target, list_preds, child_parent_indexes_map, per_level_size):
    compute_hiera_metrics(list_target, list_preds, per_level_size, child_parent_indexes_map)
    compute_f1_scores(list_target, list_preds, child_parent_indexes_map, per_level_size)
    print("Depth of prediction: " + str(depth_of_prediction_rate(list_target, list_preds, per_level_size)))
