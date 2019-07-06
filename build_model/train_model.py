import tensorflow as tf
import argparse
import numpy as np
import datetime
import sys
import os
from time import gmtime, strftime

GRAPH_TRAIN_PATH = '../create_memory_graphs/graph/'
KEY_NODE_WEIGHT_PATH = './key_node_weight.dat'
EMBEDDING_OUTPUT_PATH = './embedding/'
MODEL_OUTPUT_PATH = './model/'
VECTOR_SIZE = 64
EMBEDDING_SIZE = VECTOR_SIZE
TRAIN_EPOCH = 1000000
HOP = 3
STAT_ITERATION = 100
KEEP_PROB = 0.8
MINI_WEIGHT = 0
BONUS = 1
MODEL = 'struct2vec_edge_type'
VECTOR_TYPE = 'padding' # padding or repeat
THRESHOLD_STEP = 0.02
OPTIMIZER = 'Adam'
LEARNING_RATE = 1e-4
BALANCE = True
SAVE_ITERATION = 100
OBJ_TYPES = ['_EPROCESS','_ETHREAD','_DRIVER_OBJECT','_FILE_OBJECT','_LDR_DATA_TABLE_ENTRY', '_CM_KEY_BODY']

def content_to_vector_repeat(content):
    vector = []
    for idx in range(VECTOR_SIZE):
        vector.append(content[idx % len(content)])
    return vector 

def content_to_vector_padding(content):
    vector = content[:VECTOR_SIZE]
    vector.extend([0] * (VECTOR_SIZE - len(content)))
    return vector

def log(message):
    print('%s\t%s' %(strftime("%Y-%m-%d %H:%M:%S", gmtime()), message))
    sys.stdout.flush()

def dict_add_weight(dict_target, key, weight, bonus):
    if key not in dict_target:
        dict_target[key] = weight
    else:
        dict_target[key] += weight + bonus

def get_label_vector(output_vector_size, dict_key_node_to_weight=None, obj_type=None, inside_offset=None, node_size=None):
    label = np.zeros(output_vector_size)
    index = 1
    if dict_key_node_to_weight != None:
        key_nodes = dict_key_node_to_weight.keys()
        key_nodes.sort()
        key_node = str(inside_offset) + '_' + str(node_size)
        if key_node in key_nodes:
            index += key_nodes.index(key_node)
            label[index] = 1
            return label
    label[0] = 1
    return label

def get_offset_size(max_index, dict_key_node_to_weight):
    key_nodes = dict_key_node_to_weight.keys()
    key_nodes.sort()
    key_node = key_nodes[max_index - 1]
    return int(key_node.split('_')[0]), int(key_node.split('_')[1])

def chance(m, n):
    if np.random.randint(n) < m:
        return True
    return False

def get_train_subset_seg(set_key_node, dict_node_to_ln, dict_node_to_rn, dict_node_to_lp, dict_node_to_rp):
    set_node_addr_minibatch = set_key_node
    set_search_addr = set(set_node_addr_minibatch)
    for t in range(100):
        if t >= HOP:
            break
        set_new_addr = set()
        for addr in set_search_addr:
            ln = rn = None
            if addr in dict_node_to_ln:
                ln = dict_node_to_ln[addr]
            if addr in dict_node_to_rn:
                rn = dict_node_to_rn[addr]
            pns = dict_node_to_rp[addr].union(dict_node_to_lp[addr])
            if ln != None and ln not in set_node_addr_minibatch:
                set_new_addr.add(ln)
            if rn != None and rn not in set_node_addr_minibatch:
                set_new_addr.add(rn)
            for pn in pns:
                if pn not in set_node_addr_minibatch:
                    set_new_addr.add(pn)
            if len(set_node_addr_minibatch) + len(set_new_addr) >= MAX_NODE_COUNT:
                break
        set_node_addr_minibatch.update(set_new_addr)
        set_search_addr = set(set_new_addr)
    return set_node_addr_minibatch

def read_dataset(graph_path, output_vector_size, dict_key_node_to_weight, target_obj_type, balance=False):
    list_vector = []
    list_label = []
    list_ln = []
    list_rn = []
    list_lp = []
    list_rp = []
    #list_n = []
    dict_key_node_to_vector_list = {}
    dict_key_node_to_label_list = {}
    dict_key_node_to_idx_list = {}
    dict_idx_to_addr = {}
    dict_addr_to_idx = {}
    set_obj_addr = set()
    log('start read file')
    with open(graph_path, 'r') as f:
        lines = [x for x in f]
    log('generate list_vector, list_label')
    for idx, line in enumerate(lines):
        s = line.strip().split('\t')
        #if len(s) < 7:
        #    continue
        addr = int(s[0], 0)
        content = [int(x) for x in s[6].split(' ')]
        if VECTOR_TYPE == 'padding':
            vector = content_to_vector_padding(content)
        elif VECTOR_TYPE == 'repeat':
            vector = content_to_vector_repeat(content)
        list_vector.append(vector)
        dict_addr_to_idx[addr] = idx
        dict_idx_to_addr[idx] = addr
        obj_type = None
        if len(s) >= 8:
            obj_type = s[7].split('@')[0]
            inside_offset = int(s[7].split('@')[1])
            node_size = int(s[5])
            key_node = str(inside_offset) + '_' + str(node_size)
        if obj_type == target_obj_type and key_node in dict_key_node_to_weight:
            set_obj_addr.add(addr - inside_offset)
            label = get_label_vector(output_vector_size, dict_key_node_to_weight, obj_type, inside_offset, node_size)
            if key_node not in dict_key_node_to_vector_list:
                dict_key_node_to_vector_list[key_node] = [vector]
            else:
                dict_key_node_to_vector_list[key_node].append(vector)
            if key_node not in dict_key_node_to_label_list:
                dict_key_node_to_label_list[key_node] = [label]
            else:
                dict_key_node_to_label_list[key_node].append(label)
            if key_node not in dict_key_node_to_idx_list:
                dict_key_node_to_idx_list[key_node] = [idx]
            else:
                dict_key_node_to_idx_list[key_node].append(idx)
        else:
            label = get_label_vector(output_vector_size)
        list_label.append(label)
    log('balance obj key node amount') 
    log('Record Amount:\t' + str(len(list_label)))
    if balance == True:
        log('balance object and non-object')
        balance_type(lines, list_vector, list_label, dict_key_node_to_vector_list, dict_key_node_to_label_list, dict_key_node_to_idx_list)
        log('Record Amount (balanced):\t' + str(len(list_label)))
    #log('generate list_adj')
    log('generate list_ln, list_rn, list_lp, list_rp')
    for idx, line in enumerate(lines):
        s = line.strip().split('\t')
        #if len(s) < 7:
        #    continue
        obj_type = ''
        inside_offset = -1
        if len(s) >= 7 and '@' in s[6]:
            obj_type = s[6].split('@')[0]
            inside_offset = int(s[6].split('@')[1])
        addr = s[0]
        if s[1] != '':
            ln = int(s[1], 0)
            if ln in dict_addr_to_idx:
                list_ln.append((idx, dict_addr_to_idx[ln]))
        if s[2] != '':
            rn = int(s[2], 0)
            if rn in dict_addr_to_idx:
                list_rn.append((idx, dict_addr_to_idx[rn]))
        lps = [int(x, 0) for x in s[3].split(',') if len(x) > 0]
        for lp in lps:
            if lp in dict_addr_to_idx:
                list_lp.append((idx, dict_addr_to_idx[lp]))
        rps = [int(x, 0) for x in s[4].split(',') if len(x) > 0]
        for rp in rps:
            if rp in dict_addr_to_idx:
                list_rp.append((idx, dict_addr_to_idx[rp]))
    list_ln.sort()
    list_rn.sort()
    list_lp.sort()
    list_rp.sort()
    row_size=len(list_vector)

    indices = np.array(list_ln, dtype=np.int64)
    values = np.array([1 for _ in range(len(list_ln))], dtype=np.int8)
    shape = np.array([row_size, row_size], dtype=np.int64)
    ln_matrix = tf.SparseTensorValue(indices, values, shape)

    indices = np.array(list_rn, dtype=np.int64)
    values = np.array([1 for _ in range(len(list_rn))], dtype=np.int8)
    shape = np.array([row_size, row_size], dtype=np.int64)
    rn_matrix = tf.SparseTensorValue(indices, values, shape)

    indices = np.array(list_lp, dtype=np.int64)
    values = np.array([1 for _ in range(len(list_lp))], dtype=np.int8)
    shape = np.array([row_size, row_size], dtype=np.int64)
    lp_matrix = tf.SparseTensorValue(indices, values, shape)

    indices = np.array(list_rp, dtype=np.int64)
    values = np.array([1 for _ in range(len(list_rp))], dtype=np.int8)
    shape = np.array([row_size, row_size], dtype=np.int64)
    rp_matrix = tf.SparseTensorValue(indices, values, shape)

    return list_vector, list_label, ln_matrix, rn_matrix, lp_matrix, rp_matrix, dict_idx_to_addr, set_obj_addr
    
def balance_type(lines, list_vector, list_label, dict_key_node_to_vector_list, dict_key_node_to_label_list, dict_key_node_to_idx_list):
    total_count = len(list_label)
    output_vector_size = len(list_label[0])
    balance_rate = output_vector_size - 1 + 5
    log('balance_rate:\t%d' %balance_rate)
    obj_node_count = 0
    for key_node, label_list in dict_key_node_to_label_list.iteritems():
        obj_node_count += len(label_list) 
    not_obj_node_count = total_count - obj_node_count
    MINIMUM_KEY_NODE_AMOUNT = not_obj_node_count / balance_rate
    log('total_count: %d, obj_node_count: %d, not_obj_node_count: %d, MINIMUM_KEY_NODE_AMOUNT:%d' %(total_count, obj_node_count, not_obj_node_count, MINIMUM_KEY_NODE_AMOUNT))
    for key_node, label_list in dict_key_node_to_label_list.iteritems():
        count = len(label_list)
        for _ in range(MINIMUM_KEY_NODE_AMOUNT / count - 1):
            list_vector.extend(dict_key_node_to_vector_list[key_node])
            list_label.extend(label_list)
            for idx in dict_key_node_to_idx_list[key_node]:
                lines.append(lines[idx])

def read_dataset_list(list_file, output_vector_size, dict_key_node_to_weight, target_obj_type, balance=False):#, mutate=False):
    list_list_vector = []
    list_list_label = []
    list_ln_matrix = []
    list_rn_matrix = []
    list_lp_matrix = []
    list_rp_matrix = []
    list_dict_idx_to_addr = []
    list_set_obj_addr = []
    for file_path in list_file:
        log(file_path)
        list_vector, list_label, ln_matrix, rn_matrix, lp_matrix, rp_matrix, dict_idx_to_addr, set_obj_addr = read_dataset(file_path, output_vector_size, dict_key_node_to_weight, target_obj_type, balance)#, mutate)
        list_list_vector.append(list_vector)
        list_list_label.append(list_label)
        list_ln_matrix.append(ln_matrix)
        list_rn_matrix.append(rn_matrix)
        list_lp_matrix.append(lp_matrix)
        list_rp_matrix.append(rp_matrix)
        list_dict_idx_to_addr.append(dict_idx_to_addr)
        list_set_obj_addr.append(set_obj_addr)
    return list_list_vector, list_list_label, list_ln_matrix, list_rn_matrix, list_lp_matrix, list_rp_matrix, list_dict_idx_to_addr, list_set_obj_addr

def get_file_list(target_obj_type):
    list_file_all = [x for x in os.listdir(GRAPH_TRAIN_PATH) if 'graph.' in x and '.graph.' not in x and 'mutant' not in x and '.tag' not in x and '.all' in x]
    list_file_all.sort()
    file_amount = len(list_file_all)
    train_amount = int(file_amount * 0.6)
    validate_amount = int(file_amount * 0.2)
    test_amount = int(file_amount * 0.2)
    list_file_train = [GRAPH_TRAIN_PATH + x.replace('all', target_obj_type) for x in list_file_all[:train_amount]]
    list_file_validate = [GRAPH_TRAIN_PATH + x for x in list_file_all[train_amount:train_amount+validate_amount]]
    list_file_test = [GRAPH_TRAIN_PATH + x for x in list_file_all[train_amount+validate_amount:train_amount+validate_amount+test_amount]]
    return list_file_train, list_file_validate, list_file_test

def get_addr_to_weight(ye, dict_idx_to_addr, dict_key_node_to_weight):
    dict_addr_to_weight = {}
    for idx, result in enumerate(ye):
        #result = list(result)
        #max_index = result.index(max(result))
        max_index = np.argmax(result)
        if max_index != 0:
            offset, size = get_offset_size(max_index, dict_key_node_to_weight)
            weight = dict_key_node_to_weight[str(offset) + '_' + str(size)]
            current_node_addr = dict_idx_to_addr[idx]
            addr = current_node_addr - offset
            dict_add_weight(dict_addr_to_weight, addr, weight, BONUS)
    return dict_addr_to_weight

def output_key_node_mu(ye, mu_e, obj_type):
    if os.path.exists(EMBEDDING_OUTPUT_PATH) == False:
        os.system('mkdir -p ' + EMBEDDING_OUTPUT_PATH)
    time_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(EMBEDDING_OUTPUT_PATH + time_stamp + '.' + obj_type + '.dat', 'w') as output_mu:
        for idx, result in enumerate(ye):
            #result = list(result)
            #max_index = result.index(max(result))
            max_index = np.argmax(result)
            if max_index != 0:
                for v in mu_e[idx]:
                    output_mu.write(str(v) + '\t')
                output_mu.write(str(max_index) + '\n')

def stat_result_type(dict_addr_to_weight, set_obj_addr, obj_type, dict_key_node_to_weight, threshold=0):
    best_F1 = -1
    best_F1_result = ''
    best_F1_FPFN_value = ''
    total_weight = 0.0
    for _, weight in dict_key_node_to_weight.iteritems():
        total_weight += weight
    if threshold == 0:
        #while threshold <= total_weight + 3:
        while threshold <= total_weight:
            F1, result = stat_result_type_with_threshold(dict_addr_to_weight, set_obj_addr, obj_type, threshold, total_weight)
            if F1 >= best_F1:
                best_F1 = F1
                best_F1_result = result
            threshold += THRESHOLD_STEP
        log(best_F1_result)
    else:
        F1, result = stat_result_type_with_threshold(dict_addr_to_weight, set_obj_addr, obj_type, threshold, total_weight) 
        log(result)
    sys.stdout.flush()

def stat_result_type_with_threshold(dict_addr_to_weight, set_obj_addr, obj_type, threshold, total_weight):
    TP_amount = FP_amount = FN_amount = TN_amount = 0
    for addr, weight in dict_addr_to_weight.iteritems():
        if weight > threshold:
            if addr in set_obj_addr:
                TP_amount += 1
            else:
                FP_amount += 1
        else:
            if addr in set_obj_addr:
                FN_amount += 1
            else:
                TN_amount += 1
    FN_amount = len(set_obj_addr) - TP_amount
    precision_p = precision_t = eecall_p = recall_t = F1 = 0.0
    if TP_amount + FP_amount != 0:
        precision_t = float(TP_amount) / (TP_amount + FP_amount)
    if TP_amount + FN_amount != 0:
        recall_t = float(TP_amount) / (TP_amount + FN_amount)
    if precision_t + recall_t > 0:
        F1 = 2 * precision_t * recall_t / (precision_t + recall_t)
    result = obj_type.ljust(23) + str(threshold) + '/' + str(round(total_weight,4)).ljust(8) + ' TP: ' + str(TP_amount).ljust(5) + ' FP: ' + str(FP_amount).ljust(5) + ' FN: ' + str(FN_amount).ljust(5) + ' TN: ' + str(TN_amount).ljust(5) + '   Precision: ' + str(round(precision_t, 4)).ljust(6) + '   Recall: ' + str(round(recall_t, 4)).ljust(6) + '   F1: ' + str(round(F1, 4))
    #log('roc: ' + result)
    return F1, result#, FPFN_value

def get_outout_vector_size(dict_key_node_to_weight):
    return len(dict_key_node_to_weight) + 1

def print_configure():
    log('GRAPH_TRAIN_PATH:\t%s' %GRAPH_TRAIN_PATH)
    log('KEY_NODE_WEIGHT_PATH:\t%s' %KEY_NODE_WEIGHT_PATH)
    log('VECTOR_SIZE:\t%d' %VECTOR_SIZE)
    log('EMBEDDING_SIZE:\t%d' %EMBEDDING_SIZE)
    log('TRAIN_EPOCH:\t%d' %TRAIN_EPOCH)
    log('OPTIMIZER:\t%s' %OPTIMIZER)
    log('LEARNING_RATE:\t%f' %LEARNING_RATE)
    log('HOP:\t%d' %HOP)
    log('KEEP_PROB:\t%f' %KEEP_PROB)
    log('MINI_WEIGHT:\t%f' %MINI_WEIGHT)
    log('BONUS:\t%f' %BONUS)
    log('MODEL:\t%s' %MODEL)
    log('VECTOR_TYPE:\t%s' %VECTOR_TYPE)
    log('BALANCE:\t%d' %BALANCE)
    log('THRESHOLD_STEP:\t%f' %THRESHOLD_STEP)
    log('STAT_ITERATION:\t%d' %STAT_ITERATION)
    log('SAVE_ITERATION:\t%d' %SAVE_ITERATION)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_type', type=str, default=None, help='The object type of current test: [_EPROCESS/_ETHREAD/_DRIVER_OBJECT/_FILE_OBJECT/_LDR_DATA_TABLE_ENTRY/_CM_KEY_BODY]')
    opt = parser.parse_args()
    target_obj_type = opt.obj_type
    if target_obj_type not in OBJ_TYPES:
        os.system('python ' + sys.argv[0] + ' -h')
        exit()

    log(target_obj_type)
    print_configure()
    list_file_train, list_file_validate, list_file_test = get_file_list(target_obj_type)
    log('len(list_file_train):\t%d\n%s' %(len(list_file_train), str(list_file_train)))
    log('len(list_file_validate):\t%d\n%s' %(len(list_file_validate), str(list_file_validate)))
    log('len(list_file_test):\t%d\n%s' %(len(list_file_test), str(list_file_test)))
    dict_key_node_to_weight = load_key_node_weight(target_obj_type)
    output_vector_size = get_outout_vector_size(dict_key_node_to_weight)
    log('output_vector_size:\t%d' %output_vector_size)
    list_list_vector_train, list_list_label_train, list_ln_matrix_train, list_rn_matrix_train, list_lp_matrix_train, list_rp_matrix_train, _, _ = read_dataset_list(list_file_train, output_vector_size, dict_key_node_to_weight, target_obj_type, balance=BALANCE)
    list_list_vector_validate, list_list_label_validate, list_ln_matrix_validate, list_rn_matrix_validate, list_lp_matrix_validate, list_rp_matrix_validate, list_dict_idx_to_addr_validate, list_set_obj_addr = read_dataset_list(list_file_validate, output_vector_size, dict_key_node_to_weight, target_obj_type)

    x = tf.placeholder(tf.float32, [None, VECTOR_SIZE], name='x')
    ln = tf.sparse_placeholder(tf.float32, name='input_ln')
    rn = tf.sparse_placeholder(tf.float32, name='input_rn')
    lp = tf.sparse_placeholder(tf.float32, name='input_lp')
    rp = tf.sparse_placeholder(tf.float32, name='input_rp')
    keep_prob = tf.placeholder(tf.float32, name='ph')
    mu = tf.zeros_like(x)
    w_init = tf.truncated_normal_initializer(stddev=0.1)
    W_1 = tf.get_variable('W_1', dtype=tf.float32, shape=[VECTOR_SIZE, VECTOR_SIZE], initializer=w_init)
    W_2 = tf.get_variable('W_2', dtype=tf.float32, shape=[VECTOR_SIZE, VECTOR_SIZE], initializer=w_init)
    W_3 = tf.get_variable('W_3', dtype=tf.float32, shape=[VECTOR_SIZE, VECTOR_SIZE], initializer=w_init)
    W_4 = tf.get_variable('W_4', dtype=tf.float32, shape=[VECTOR_SIZE, output_vector_size], initializer=w_init)
    P_1 = tf.get_variable('P_1', dtype=tf.float32, shape=[VECTOR_SIZE, VECTOR_SIZE], initializer=w_init)
    P_2 = tf.get_variable('P_2', dtype=tf.float32, shape=[VECTOR_SIZE, VECTOR_SIZE], initializer=w_init)
    P_3 = tf.get_variable('P_3', dtype=tf.float32, shape=[VECTOR_SIZE, VECTOR_SIZE], initializer=w_init)
    P_4 = tf.get_variable('P_4', dtype=tf.float32, shape=[VECTOR_SIZE, VECTOR_SIZE], initializer=w_init)
    P_5 = tf.get_variable('P_5', dtype=tf.float32, shape=[VECTOR_SIZE, VECTOR_SIZE], initializer=w_init)
    P_6 = tf.get_variable('P_6', dtype=tf.float32, shape=[VECTOR_SIZE, VECTOR_SIZE], initializer=w_init)
    P_7 = tf.get_variable('P_7', dtype=tf.float32, shape=[VECTOR_SIZE, VECTOR_SIZE], initializer=w_init)
    P_8 = tf.get_variable('P_8', dtype=tf.float32, shape=[VECTOR_SIZE, VECTOR_SIZE], initializer=w_init)
    P_9 = tf.get_variable('P_9', dtype=tf.float32, shape=[VECTOR_SIZE, VECTOR_SIZE], initializer=w_init)
    P_10 = tf.get_variable('P_10', dtype=tf.float32, shape=[VECTOR_SIZE, VECTOR_SIZE], initializer=w_init)
    P_11 = tf.get_variable('P_11', dtype=tf.float32, shape=[VECTOR_SIZE, VECTOR_SIZE], initializer=w_init)
    P_12 = tf.get_variable('P_12', dtype=tf.float32, shape=[VECTOR_SIZE, VECTOR_SIZE], initializer=w_init)
    for t in range(HOP):
        ############################
        #l1 = tf.nn.dropout(tf.nn.relu(tf.matmul(tf.sparse_tensor_dense_matmul(ln,mu),P_1)),keep_prob)
        #l2 = tf.nn.dropout(tf.nn.relu(tf.matmul(tf.sparse_tensor_dense_matmul(rn,mu),P_4)),keep_prob)
        #l3 = tf.nn.dropout(tf.nn.relu(tf.matmul(tf.sparse_tensor_dense_matmul(lp,mu),P_7)),keep_prob)
        #l4 = tf.nn.dropout(tf.nn.relu(tf.matmul(tf.sparse_tensor_dense_matmul(rp,mu),P_10)),keep_prob)
        ############################
        #l1 = tf.nn.dropout(tf.nn.relu(tf.matmul(tf.matmul(tf.sparse_tensor_dense_matmul(ln,mu),P_1),P_2)),keep_prob)
        #l2 = tf.nn.dropout(tf.nn.relu(tf.matmul(tf.matmul(tf.sparse_tensor_dense_matmul(rn,mu),P_4),P_5)),keep_prob)
        #l3 = tf.nn.dropout(tf.nn.relu(tf.matmul(tf.matmul(tf.sparse_tensor_dense_matmul(lp,mu),P_7),P_8)),keep_prob)
        #l4 = tf.nn.dropout(tf.nn.relu(tf.matmul(tf.matmul(tf.sparse_tensor_dense_matmul(rp,mu),P_10),P_11)),keep_prob)
        ############################
        l1 = tf.nn.dropout(tf.nn.relu(tf.matmul(tf.matmul(tf.matmul(tf.sparse_tensor_dense_matmul(ln,mu),P_1),P_2),P_3)),keep_prob)
        l2 = tf.nn.dropout(tf.nn.relu(tf.matmul(tf.matmul(tf.matmul(tf.sparse_tensor_dense_matmul(rn,mu),P_4),P_5),P_6)),keep_prob)
        l3 = tf.nn.dropout(tf.nn.relu(tf.matmul(tf.matmul(tf.matmul(tf.sparse_tensor_dense_matmul(lp,mu),P_7),P_8),P_9)),keep_prob)
        l4 = tf.nn.dropout(tf.nn.relu(tf.matmul(tf.matmul(tf.matmul(tf.sparse_tensor_dense_matmul(rp,mu),P_10),P_11),P_12)),keep_prob)
        ############################
        mu = tf.nn.tanh(tf.matmul(x, W_1) + l1 + l2 + l3 + l4)
    #y = tf.nn.dropout(tf.matmul(mu, W_4), keep_prob)
    #y = tf.nn.dropout(tf.matmul(tf.matmul(mu, W_3), W_4), keep_prob)
    y = tf.nn.dropout(tf.matmul(tf.matmul(tf.matmul(mu, W_2), W_3), W_4), keep_prob)

    y_ = tf.placeholder(tf.float32, [None, output_vector_size], name='y_')
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    if OPTIMIZER == 'Adam':
        train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cross_entropy)
    elif OPTIMIZER == 'Adagrad':
        train_step = tf.train.AdagradOptimizer(learning_rate=LEARNING_RATE).minimize(cross_entropy)
    elif OPTIMIZER == 'GradientDescent':
        train_step = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(cross_entropy)
    #train_step = tf.train.AdamOptimizer().minimize(cross_entropy)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        #correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        log('======================================')
        for i in range(TRAIN_EPOCH):
            idx_rand = np.random.randint(len(list_list_vector_train))
            sess.run(train_step, feed_dict={x:list_list_vector_train[idx_rand], ln:list_ln_matrix_train[idx_rand], rn:list_rn_matrix_train[idx_rand], lp:list_lp_matrix_train[idx_rand], rp:list_rp_matrix_train[idx_rand], y_:list_list_label_train[idx_rand], keep_prob:KEEP_PROB})
            if i != 0 and i % STAT_ITERATION == 0:
                if need_save_model(i):
                    folder_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    model_save_path = MODEL_OUTPUT_PATH + target_obj_type + '/' + folder_name + '/'
                    if os.path.exists(model_save_path) == False:
                        os.system('mkdir -p ' + model_save_path)
                    save_path = saver.save(sess, model_save_path + MODEL + '.HOP' + str(HOP) + '.' + target_obj_type, global_step=i)
                    log('Model saved in path: %s' % save_path)
                log('Train Epoch:\t' + str(i))
                idx_validate = np.random.randint(len(list_list_vector_validate))
                log('Validate using ' + list_file_validate[idx_validate])
                ye = sess.run(y, feed_dict={x:list_list_vector_validate[idx_validate], ln:list_ln_matrix_validate[idx_validate], rn:list_rn_matrix_validate[idx_validate], lp:list_lp_matrix_validate[idx_validate], rp:list_rp_matrix_validate[idx_validate], keep_prob:1})
                mu_e = sess.run(mu, feed_dict={x:list_list_vector_validate[idx_validate], ln:list_ln_matrix_validate[idx_validate], rn:list_rn_matrix_validate[idx_validate], lp:list_lp_matrix_validate[idx_validate], rp:list_rp_matrix_validate[idx_validate], keep_prob:1})
                #output_key_node_mu(ye, mu_e, target_obj_type)
                log('get_addr_to_weight')
                dict_addr_to_weight = get_addr_to_weight(ye, list_dict_idx_to_addr_validate[idx_validate], dict_key_node_to_weight) #todo
                log('stat_result_type')
                print len(dict_addr_to_weight), len(list_set_obj_addr[idx_validate]), target_obj_type, len(dict_key_node_to_weight)
                if len(dict_addr_to_weight) < 100000:
                    stat_result_type(dict_addr_to_weight, list_set_obj_addr[idx_validate], target_obj_type, dict_key_node_to_weight) #todo
                    log('Validate finished.')
                else:
                    log('Too many candidate')
                log('------------------------------------------------')

def need_save_model(i):
    if i > 0 and i % SAVE_ITERATION == 0:
        return True
    else:
        return False
    
def load_key_node_weight(target_obj_type):
    dict_key_node_to_weight = {}
    with open(KEY_NODE_WEIGHT_PATH, 'r') as f:
        for line in f:
            s = line.strip().split('\t')
            obj_type = s[0]
            key_node = s[1]
            weight = float(s[2])
            if obj_type == target_obj_type and weight > MINI_WEIGHT:
                #print obj_type, key_node, weight
                dict_key_node_to_weight[key_node] = weight
    return dict_key_node_to_weight

if __name__ == '__main__':
    main()
