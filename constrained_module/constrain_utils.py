import torch
import math


def constrained_generation_dfs(dynamic_vocab_cache, inputs_ids, next_token_logits, current_to_possible_tokens_map,
                               tokenizer):
    """
    Constraining next token generation with respect to hierarchy (current_to_possible_tokens_map) for dfs target.
    """
    logits_rows = []
    for i in range(next_token_logits.size()[0]):
        preds = inputs_ids[i].tolist()
        current_to_possible_tokens = compute_dynamic_vocab_dfs(dynamic_vocab_cache, preds,
                                                               current_to_possible_tokens_map,
                                                               tokenizer)
        logits_softmax_applied = constrained_logits_log_softmax_for_generation(current_to_possible_tokens, i,
                                                                               next_token_logits)
        logits_rows.append(logits_softmax_applied)
    return torch.stack(logits_rows)


def constrained_generation_bfs(dynamic_vocab_cache, inputs_ids, next_token_logits, current_to_possible_tokens_map,
                               set_all_labels, tokenizer,
                               ):
    """
       Constraining next token generation with respect to hierarchy (current_to_possible_tokens_map) for bfs target.
    """
    logits_rows = []
    for i in range(next_token_logits.size()[0]):
        preds = inputs_ids[i].tolist()
        current_to_possible_tokens = compute_dynamic_vocab_bfs(dynamic_vocab_cache, preds,
                                                               current_to_possible_tokens_map, set_all_labels,
                                                               tokenizer)
        logits_softmax_applied = constrained_logits_log_softmax_for_generation(current_to_possible_tokens, i,
                                                                               next_token_logits)
        logits_rows.append(logits_softmax_applied)
    return torch.stack(logits_rows)


def compute_dynamic_vocab_dfs(dynamic_vocab_cache, preds, current_to_possible_tokens_map_parent_child, tokenizer):
    if preds and preds[-1] == 1:
        return [0]
    tuple_preds = tuple(preds)
    if tuple_preds in dynamic_vocab_cache:
        return dynamic_vocab_cache[tuple_preds]
    if preds == [18134]:
        return [value[0] for value in current_to_possible_tokens_map_parent_child["Root"]]
    preds_text = tokenizer.decode(preds, skip_special_tokens=True).split()
    if not preds_text:
        return [18134]  # Root for the first generation
    stack = []  # Using a list as a stack
    for pred in preds_text:
        if pred != "Pop":
            stack.append(pred)  # Continue until decoder cannot find the label in parent_child_map
        elif stack:
            stack.pop()

    if stack:
        y_current = stack.pop()
    else:
        return [5777]
    if y_current in current_to_possible_tokens_map_parent_child and (not stack or stack[
        -1] != y_current):  # the and is for RCV1V1 we have C17 but C171 also !!!
        if preds[-1] != 3 and preds[-1] != 2:
            possible_tokens = [e[0] for e in current_to_possible_tokens_map_parent_child[y_current]]
            dynamic_vocab_cache[tuple_preds] = possible_tokens
            return possible_tokens
        else:
            list_of_childs = current_to_possible_tokens_map_parent_child[y_current]
            if preds[-1] != 2:
                current_tokens = [3]
            elif len(preds) > 2 and preds[-2] == 3 and preds[-1] == 2:
                current_tokens = [3, 2]
            result = [lst for lst in list_of_childs if lst[:len(current_tokens)] == current_tokens]
            next_ones = [e[len(current_tokens)] for e in result if len(e) > len(current_tokens)]
            if next_ones:
                dynamic_vocab_cache[tuple_preds] = next_ones
                return next_ones
    else:
        if stack:
            parent = stack.pop()
            list_of_childs = current_to_possible_tokens_map_parent_child[parent]
            current_tokens = tokenizer(y_current, add_special_tokens=False).input_ids
            result = [lst for lst in list_of_childs if lst[:len(current_tokens)] == current_tokens]
            next_ones = [e[len(current_tokens)] for e in result if len(e) > len(current_tokens)]
            if next_ones:
                dynamic_vocab_cache[tuple_preds] = next_ones
                return next_ones
    return [5777]


def constrained_logits_log_softmax(current_to_possible_tokens, i, logits, prob_f):
    """
    Probability distribution over current_to_possible_tokens for loss calculation
    """
    logits_softmax_applied = torch.full(logits[i].size(), math.log(1E-32), device=logits.device)
    logits_softmax_applied[current_to_possible_tokens] = prob_f(logits[i, current_to_possible_tokens])
    return logits_softmax_applied


def constrained_logits_log_softmax_for_generation(current_to_possible_tokens, i, logits):
    """
       Probability distribution over current_to_possible_tokens for generation
    """
    logits_softmax_applied = torch.full(logits[i].size(), math.log(1E-32), device=logits.device)
    logits_softmax_applied[current_to_possible_tokens] = logits[i, current_to_possible_tokens]
    return logits_softmax_applied


def encode_text_data(parent_child, tokenizer, is_dfs=True):
    """
       Child tokenization. A node can have multiple tokens
      """
    encoded_map = {}
    for key, value_list in parent_child.items():
        encoded_values = []
        [encoded_values.append(
            tokenizer(value, add_special_tokens=False).input_ids) for value in
            value_list]
        if key not in encoded_map:
            if is_dfs and key == "Root":
                encoded_values.append([1])
            if is_dfs and key != "Root":
                encoded_values.append([5777])  # Pop is always possible but Root. 5777 is token number for "Pop"
            encoded_map[key] = encoded_values
        else:
            return Exception
    return encoded_map


def constrained_loss_dfs(model, loss_f, logits_batch, target_batch, current_to_possible_tokens_map,
                         tokenizer):
    prob_f = torch.nn.LogSoftmax(dim=0)
    batch_size = logits_batch.size()[0]
    logits_softmax_applied_batch = []
    target_batch_no_pad = []
    total_time = 0
    for j in range(batch_size):
        logits = logits_batch[j]
        target = target_batch[j]
        preds = []
        len_seq = target.size()[0]
        current_to_possible_tokens = [18134]  # First one is "Root"
        for i in range(len_seq):
            if target[i].unsqueeze(0) == 0:
                break
            logits_softmax_applied = constrained_logits_log_softmax(current_to_possible_tokens, i, logits, prob_f)
            current_pred_token = torch.argmax(logits_softmax_applied).item()
            preds.append(current_pred_token)
            current_to_possible_tokens = compute_dynamic_vocab_dfs(model.dynamic_vocab_cache, preds,
                                                                   current_to_possible_tokens_map, tokenizer)
            logits_softmax_applied_batch.append(logits_softmax_applied.unsqueeze(0))
            target_batch_no_pad.append(target[i].unsqueeze(0))
    logits_softmax_applied_batch = torch.cat(logits_softmax_applied_batch, dim=0)
    target_batch_no_pad = torch.cat(target_batch_no_pad, dim=0)
    return loss_f(logits_softmax_applied_batch, target_batch_no_pad)


def constrained_loss_bfs(model, loss_f, logits_batch, target_batch, current_to_possible_tokens_map,
                         set_all_labels, tokenizer,
                         ):
    prob_f = torch.nn.LogSoftmax(dim=0)
    batch_size = logits_batch.size()[0]
    logits_softmax_applied_batch = []
    target_batch_no_pad = []
    for j in range(batch_size):
        logits = logits_batch[j]
        target = target_batch[j]
        preds = []
        len_seq = target.size()[0]
        current_to_possible_tokens = [l[0] for l in current_to_possible_tokens_map["Root"]]  # First level label tokens
        for i in range(len_seq):
            if target[i].unsqueeze(0) == 0:
                break
            logits_softmax_applied = constrained_logits_log_softmax(current_to_possible_tokens, i, logits, prob_f)
            current_pred_token = torch.argmax(logits_softmax_applied).item()
            preds.append(current_pred_token)
            current_to_possible_tokens = compute_dynamic_vocab_bfs(model.dynamic_vocab_cache, preds,
                                                                   current_to_possible_tokens_map, set_all_labels,
                                                                   tokenizer)
            logits_softmax_applied_batch.append(logits_softmax_applied.unsqueeze(0))
            target_batch_no_pad.append(target[i].unsqueeze(0))
    logits_softmax_applied_batch = torch.cat(logits_softmax_applied_batch, dim=0)
    target_batch_no_pad = torch.cat(target_batch_no_pad, dim=0)
    return loss_f(logits_softmax_applied_batch, target_batch_no_pad)


def compute_dynamic_vocab_bfs(dynamic_vocab_cache, preds, current_to_possible_tokens_map, set_all_labels, tokenizer
                              ):
    if preds and preds[-1] == 1:
        return [0]
    tuple_preds = tuple(preds)
    if tuple_preds in dynamic_vocab_cache:
        return dynamic_vocab_cache[tuple_preds]
    if preds and preds[-1] == 0:
        next_ones = [l[0] for l in current_to_possible_tokens_map["Root"]]
        dynamic_vocab_cache[tuple_preds] = next_ones
        return next_ones
    preds_text = tokenizer.decode(preds, skip_special_tokens=True).replace("/", " /").replace("-", " -")
    preds_text_list = preds_text.split()
    preds_text_list = [item for item in preds_text_list if item != ""]
    preds_text_by_level = preds_text.split(" /")
    preds_text_by_level = [item for item in preds_text_by_level if item != ""]
    current_level = len(preds_text_by_level)
    if not preds_text_list:
        if len(preds) > 1 and preds[-1] == 2 and preds[-2] == 3:
            current_tokens = [3, 2]
            possibles_tokens = current_to_possible_tokens_map["Root"]
            result = [lst for lst in possibles_tokens if lst[:len(current_tokens)] == current_tokens]
            next_ones = [e[len(current_tokens)] for e in result if len(e) > len(current_tokens)]
            if next_ones:
                dynamic_vocab_cache[tuple_preds] = next_ones
                return next_ones
        elif preds[-1] == 3:
            current_tokens = [3]
            possibles_tokens = current_to_possible_tokens_map["Root"]
            result = [lst for lst in possibles_tokens if lst[:len(current_tokens)] == current_tokens]
            next_ones = [e[len(current_tokens)] for e in result if len(e) > len(current_tokens)]
            if next_ones:
                dynamic_vocab_cache[tuple_preds] = next_ones
                return next_ones
        next_ones = [l[0] for l in current_to_possible_tokens_map["Root"]]
        dynamic_vocab_cache[tuple_preds] = next_ones
        return next_ones

    y_current = preds_text_list[-1]

    if (y_current in set_all_labels and (len(preds_text_list) <= 1 or y_current not in preds_text_list[0:-1])) or preds[
        -1] == 3 or preds[-1] == 2:

        if preds[-1] == 3:
            if len(preds) > 1 and preds[-2] == 87:
                current_level += 1
            current_tokens = [3]
            parents = []
            if current_level > 1:
                parents = preds_text_by_level[current_level - 2].split(" -")
                parents = [item.replace(" ", "") for item in parents if item != ""]
            if not parents:
                parents.append("Root")
            possibles_tokens = []
            for parent in parents:
                if parent in current_to_possible_tokens_map:
                    possibles_tokens.extend(current_to_possible_tokens_map[parent])
            result = [lst for lst in possibles_tokens if lst[:len(current_tokens)] == current_tokens]
            next_ones = [e[len(current_tokens)] for e in result if len(e) > len(current_tokens)]
            if next_ones:
                dynamic_vocab_cache[tuple_preds] = next_ones
                return next_ones

        elif preds[-1] == 2:
            current_tokens = [2]
            if len(preds) > 2 and preds[-3] == 87:
                current_level += 1
            if len(preds) > 1 and preds[-2] == 3:
                current_tokens = [3, 2]
            parents = []
            if current_level > 1:
                parents = preds_text_by_level[current_level - 2].split(" -")
                parents = [item.replace(" ", "") for item in parents if item != ""]
            if not parents:
                parents.append("Root")
            possibles_tokens = []
            for parent in parents:
                if parent in current_to_possible_tokens_map:
                    possibles_tokens.extend(current_to_possible_tokens_map[parent])
            result = [lst for lst in possibles_tokens if lst[:len(current_tokens)] == current_tokens]
            next_ones = [e[len(current_tokens)] for e in result if len(e) > len(current_tokens)]
            if next_ones:
                dynamic_vocab_cache[tuple_preds] = next_ones
                return next_ones


        else:
            if y_current in current_to_possible_tokens_map:
                next_ones = [18, 87, 1]  # [/ , - , <eos>]
            else:
                next_ones = [18, 1]  # [/ , <eos>]
            dynamic_vocab_cache[tuple_preds] = next_ones
            return next_ones
    elif y_current == "-":
        parents = []
        if current_level > 1:
            parents = preds_text_by_level[current_level - 2].split(" -")
            parents = [item.replace(" ", "") for item in parents if item != ""]
        if not parents:
            parents.append("Root")
        possibles_tokens = []
        for parent in parents:
            if parent in current_to_possible_tokens_map:
                possibles_tokens.extend(current_to_possible_tokens_map[parent])
        next_ones = [e[0] for e in possibles_tokens]
        current_brother_tokens = []
        [current_brother_tokens.append(
            tokenizer(value, add_special_tokens=False).input_ids) for value in
            preds_text_by_level[current_level - 1].split(" -")]
        for e in current_brother_tokens:
            if e and e[0] in next_ones:
                next_ones.remove(e[0])

        if next_ones:
            dynamic_vocab_cache[tuple_preds] = next_ones
            return next_ones
        # all brothers have been predicted rare but may happen
        next_ones = [87, 1]  # down or and
        dynamic_vocab_cache[tuple_preds] = next_ones
        return next_ones
    elif y_current == "/":
        parents = []
        if current_level > 0:
            parents = preds_text_by_level[current_level - 1].split(" -")
            parents = [item.replace(" ", "") for item in parents if item != ""]
        if not parents:
            parents.append("Root")
        possibles_tokens = []
        for parent in parents:
            if parent in current_to_possible_tokens_map:
                possibles_tokens.extend(current_to_possible_tokens_map[parent])
        next_ones = [e[0] for e in possibles_tokens]
        if next_ones:
            dynamic_vocab_cache[tuple_preds] = next_ones
            return next_ones
    else:
        current_tokens = tokenizer(y_current, add_special_tokens=False).input_ids
        parents = []
        if current_level > 1:
            parents = preds_text_by_level[current_level - 2].split(" -")
        parents = [item.replace(" ", "") for item in parents if item != ""]
        if not parents:
            parents.append("Root")
        possibles_tokens = []
        for parent in parents:
            if parent in current_to_possible_tokens_map:
                possibles_tokens.extend(current_to_possible_tokens_map[parent])
        result = [lst for lst in possibles_tokens if lst[:len(current_tokens)] == current_tokens]
        next_ones = [e[len(current_tokens)] for e in result if len(e) > len(current_tokens)]
        if next_ones:
            dynamic_vocab_cache[tuple_preds] = next_ones
            return next_ones
    return [1]
