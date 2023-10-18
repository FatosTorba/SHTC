label2desc_reduced = {'CCAT': 'CORPORATES', 'ECAT': 'ECONOMICS', 'GCAT': 'GOVERNMENTS', 'MCAT': 'MARKETS',
                      'C11': 'STRATEGY', 'C12': 'LEGAL', 'C13': 'REGULATION', 'C14': 'SHARE LISTINGS',
                      'C15': 'PERFORMANCE', 'C16': 'INSOLVENCY', 'C17': 'FUNDING', 'C18': 'OWNERSHIP CHANGES',
                      'C21': 'PRODUCTION', 'C22': 'NEW PRODUCTS', 'C23': 'RESEARCH', 'C24': 'CAPACITY',
                      'C31': 'MARKETING', 'C32': 'ADVERTISING', 'C33': 'CONTRACTS', 'C34': 'MONOPOLIES',
                      'C41': 'MANAGEMENT', 'C42': 'LABOUR', 'E11': 'ECONOMIC PERFORMANCE', 'E12': 'MONETARY',
                      'E13': 'INFLATION', 'E14': 'CONSUMER FINANCE', 'E21': 'GOVERNMENT FINANCE', 'E31': 'OUTPUT',
                      'E41': 'EMPLOYMENT', 'E51': 'TRADE', 'E61': 'HOUSING STARTS', 'E71': 'LEADING INDICATORS',
                      'G15': 'EUROPEAN COMMUNITY', 'GCRIM': 'CRIME, LAW ENFORCEMENT', 'GDEF': 'DEFENCE',
                      'GDIP': 'INTERNATIONAL RELATIONS', 'GDIS': 'DISASTERS AND ACCIDENTS',
                      'GENT': 'ARTS, CULTURE, ENTERTAINMENT', 'GENV': 'ENVIRONMENT AND NATURAL WORLD',
                      'GFAS': 'FASHION', 'GHEA': 'HEALTH', 'GJOB': 'LABOUR ISSUES', 'GMIL': 'MILLENNIUM ISSUES',
                      'GOBIT': 'OBITUARIES', 'GODD': 'HUMAN INTEREST', 'GPOL': 'DOMESTIC POLITICS',
                      'GPRO': 'BIOGRAPHIES, PERSONALITIES, PEOPLE', 'GREL': 'RELIGION',
                      'GSCI': 'SCIENCE AND TECHNOLOGY', 'GSPO': 'SPORTS', 'GTOUR': 'TRAVEL AND TOURISM',
                      'GVIO': 'WAR, CIVIL WAR', 'GVOTE': 'ELECTIONS', 'GWEA': 'WEATHER',
                      'GWELF': 'WELFARE, SOCIAL SERVICES', 'M11': 'EQUITY MARKETS', 'M12': 'BOND MARKETS',
                      'M13': 'MONEY MARKETS', 'M14': 'COMMODITY MARKETS', 'C151': 'ACCOUNTS', 'C152': 'COMMENT',
                      'C171': 'SHARE CAPITAL', 'C172': 'BONDS', 'C173': 'LOANS', 'C174': 'CREDIT RATINGS',
                      'C181': 'MERGERS', 'C182': 'ASSET TRANSFERS', 'C183': 'PRIVATISATIONS',
                      'C311': 'DOMESTIC MARKETS', 'C312': 'EXTERNAL MARKETS', 'C313': 'MARKET SHARE',
                      'C331': 'DEFENCE CONTRACTS', 'C411': 'MANAGEMENT MOVES', 'E121': 'MONEY SUPPLY',
                      'E131': 'CONSUMER PRICES', 'E132': 'WHOLESALE PRICES', 'E141': 'PERSONAL INCOME',
                      'E142': 'CONSUMER CREDIT', 'E143': 'RETAIL SALES', 'E211': 'EXPENDITURE',
                      'E212': 'GOVERNMENT BORROWING', 'E311': 'INDUSTRIAL PRODUCTION', 'E312': 'CAPACITY UTILIZATION',
                      'E313': 'INVENTORIES', 'E411': 'UNEMPLOYMENT', 'E511': 'BALANCE OF PAYMENTS',
                      'E512': 'MERCHANDISE TRADE', 'E513': 'RESERVES', 'G151': 'EC INTERNAL MARKET',
                      'G152': 'EC CORPORATE POLICY', 'G153': 'EC AGRICULTURE POLICY', 'G154': 'EC MONETARY',
                      'G155': 'EC INSTITUTIONS', 'G156': 'EC ENVIRONMENT ISSUES', 'G157': 'EC COMPETITION',
                      'G158': 'EC EXTERNAL RELATIONS', 'G159': 'EC GENERAL', 'M131': 'INTERBANK MARKETS',
                      'M132': 'FOREX MARKETS', 'M141': 'SOFT COMMODITIES', 'M142': 'METALS TRADING',
                      'M143': 'ENERGY MARKETS', 'C1511': 'ANNUAL RESULTS'}

label2desc_reduced = {k: v.lower() for k, v in label2desc_reduced.items()}


def dfs(parent_child_map, start_node, l1, l2, l3, l4):
    l1, l2, l3, l4 = replace_label_with_full_name(l1, l2, l3, l4)
    l1, l2, l3, l4 = remove_reserved_chars(l1, l2, l3, l4)
    possible_nodes = set(l1 + l2 + l3 + l4)
    possible_nodes.add("Root")
    visited = set()
    level_node_stack = [0]
    stack = [start_node]
    dfs_order = []
    level = -1
    while stack:
        node = stack.pop()
        current = level_node_stack.pop()
        if node not in visited and node in possible_nodes:
            for i in range(level - current + 1):
                dfs_order.append("Pop")
            dfs_order.append(node)
            visited.add(node)
            if node in parent_child_map and node in possible_nodes:
                children = [child for child in parent_child_map[node][::-1] if child in possible_nodes]
                # Reverse to maintain left-to-right order
                stack.extend(children)
                level_node_stack.extend([current + 1] * len(children))
            elif node == "Root":
                children = l1[::-1]
                # Reverse to maintain left-to-right order
                stack.extend(children)
                level_node_stack.extend([current + 1] * len(children))

        level = current
    for i in range(level):
        dfs_order.append("Pop")
    return ' '.join(dfs_order)


def bottom_top_dfs(parent_child_map, l1, l2, l3, l4):
    l1, l2, l3, l4 = replace_label_with_full_name(l1, l2, l3, l4)
    l1, l2, l3, l4 = remove_reserved_chars(l1, l2, l3, l4)
    child_parent_map = {v: k for k, value in parent_child_map.items() for v in value}
    explored = set()
    target = ""
    all_labels = []
    all_labels.extend(l4)
    all_labels.extend(l3)
    all_labels.extend(l2)
    all_labels.extend(l1)
    for l in all_labels:
        if l in explored:
            continue
        if target == "":
            target += l
        else:
            target += " " + l
        explored.add(l)
        target = go_up(all_labels, child_parent_map, explored, l, parent_child_map, target)

    return target


def go_up(all_labels, child_parent_map, explored, l, parent_child_map, target):
    if l in child_parent_map:
        parent = child_parent_map[l]
        if parent != "Root":
            brothers = parent_child_map[parent]
            for b in brothers:
                if b != l and b in all_labels and b not in explored:
                    target += "- " + b
                    explored.add(b)
        target += "/ " + parent
        explored.add(parent)
        target = go_up(all_labels, child_parent_map, explored, parent, parent_child_map, target)
    return target


def bfs(l1, l2, l3, l4):
    l1, l2, l3, l4 = replace_label_with_full_name(l1, l2, l3, l4)
    l1, l2, l3, l4 = remove_reserved_chars(l1, l2, l3, l4)
    path = ""
    for i in range(len(l1)):
        if i == 0:
            path += l1[i]
        else:
            path += "- " + l1[i]
    for i in range(len(l2)):
        if i == 0:
            path += "/ " + l2[i]
        else:
            path += "- " + l2[i]
    for i in range(len(l3)):
        if i == 0:
            path += "/ " + l3[i]
        else:
            path += "- " + l3[i]
    for i in range(len(l4)):
        if i == 0:
            path += "/ " + l4[i]
        else:
            path += "- " + l4[i]
    return path


def bottom_top_bfs(l1, l2, l3, l4):
    l1, l2, l3, l4 = replace_label_with_full_name(l1, l2, l3, l4)
    l1, l2, l3, l4 = remove_reserved_chars(l1, l2, l3, l4)
    path = ""
    for i in range(len(l4)):
        if i == 0:
            path += l4[i]
        else:
            path += "- " + l4[i]
    for i in range(len(l3)):
        if i == 0:
            if path != "":
                path += "/ " + l3[i]
            else:
                path += l3[i]
        else:
            path += "- " + l3[i]
    for i in range(len(l2)):
        if i == 0:
            if path != "":
                path += "/ " + l2[i]
            else:
                path += l2[i]
        else:
            path += "- " + l2[i]
    for i in range(len(l1)):
        if i == 0:
            if path != "":
                path += "/ " + l1[i]
            else:
                path += l1[i]
        else:
            path += "- " + l1[i]
    return path


def replace_label_with_full_name(l1, l2, l3, l4):
    if l1[0] not in label2desc_reduced:
        return l1, l2, l3, l4
    return [label2desc_reduced[label] for label in l1], [
        label2desc_reduced[label] for label in l2], [
               label2desc_reduced[label] for label in
               l3], [label2desc_reduced[label] for label
                     in l4]


def remove_reserved_chars_in_map(parent_childs):
    def remove_spaces_from_list(lst):
        return [string.replace(" ", "").replace("-", "").replace("/", "") for string in lst]

    parent_childs_clean = {}
    for key, value in parent_childs.items():
        parent_childs_clean[key.replace(" ", "").replace("-", "").replace("/", "")] = remove_spaces_from_list(
            value)
    return parent_childs_clean


def remove_reserved_chars(l1, l2, l3, l4):
    return [l.replace(" ", "").replace("-", "").replace("/", "") for l in l1], [
        l.replace(" ", "").replace("-", "").replace("/", "") for l in l2], [
               l.replace(" ", "").replace("-", "").replace("/", "") for l in l3], [
               l.replace(" ", "").replace("-", "").replace("/", "") for l in l4]
