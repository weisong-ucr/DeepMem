from train_model import *

OUTPUT_PATH = '../create_memory_graphs/graph/'
MUTATE_DUMP_PATH = '../memory_dumps_mutate/'
DUMP_PATH = '../memory_dumps/'
TEST_FROM = 'graph'  # [graph/dump/mutate_diff/mutate_pooltag/mutate_link]
OBJ_TPE_THRESHOLD_PATH = './obj_type_threshold.dat'
ROC = False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_type', type=str, default=None, help='The object type of current test: [_EPROCESS/_ETHREAD/_DRIVER_OBJECT/_FILE_OBJECT/_LDR_DATA_TABLE_ENTRY/_CM_KEY_BODY]')
    parser.add_argument('--model_path', type=str, default=None, help='The path to the saved model used for current test')
    opt = parser.parse_args()

    target_obj_type = opt.obj_type
    model_path = opt.model_path
    if target_obj_type not in OBJ_TYPES or model_path == None:
        os.system('python ' + sys.argv[0] + ' -h')
        exit()
    log('Object type: ' + target_obj_type)
    log('Model path: ' + model_path)
    print_configure()

    # load the threshold of the target object type
    obj_type_threshold = load_threshold(target_obj_type)

    dict_key_node_to_weight = load_key_node_weight(target_obj_type)
    log(str(dict_key_node_to_weight))
    output_vector_size = get_outout_vector_size(dict_key_node_to_weight)
    log('output_vector_size:\t%d' %output_vector_size)

    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, [None, VECTOR_SIZE], name='x')
    ln = tf.sparse_placeholder(tf.float32, name='input_ln')
    rn = tf.sparse_placeholder(tf.float32, name='input_rn')
    lp = tf.sparse_placeholder(tf.float32, name='input_lp')
    rp = tf.sparse_placeholder(tf.float32, name='input_rp')
    keep_prob = tf.placeholder(tf.float32)
    W_1 = tf.get_variable('W_1', dtype=tf.float32, shape=[VECTOR_SIZE, VECTOR_SIZE], trainable=False)
    W_2 = tf.get_variable('W_2', dtype=tf.float32, shape=[VECTOR_SIZE, VECTOR_SIZE], trainable=False)
    W_3 = tf.get_variable('W_3', dtype=tf.float32, shape=[VECTOR_SIZE, VECTOR_SIZE], trainable=False)
    W_4 = tf.get_variable('W_4', dtype=tf.float32, shape=[VECTOR_SIZE, output_vector_size], trainable=False)
    P_1 = tf.get_variable('P_1', dtype=tf.float32, shape=[VECTOR_SIZE, VECTOR_SIZE], trainable=False)
    P_2 = tf.get_variable('P_2', dtype=tf.float32, shape=[VECTOR_SIZE, VECTOR_SIZE], trainable=False)
    P_3 = tf.get_variable('P_3', dtype=tf.float32, shape=[VECTOR_SIZE, VECTOR_SIZE], trainable=False)
    P_4 = tf.get_variable('P_4', dtype=tf.float32, shape=[VECTOR_SIZE, VECTOR_SIZE], trainable=False)
    P_5 = tf.get_variable('P_5', dtype=tf.float32, shape=[VECTOR_SIZE, VECTOR_SIZE], trainable=False)
    P_6 = tf.get_variable('P_6', dtype=tf.float32, shape=[VECTOR_SIZE, VECTOR_SIZE], trainable=False)
    P_7 = tf.get_variable('P_7', dtype=tf.float32, shape=[VECTOR_SIZE, VECTOR_SIZE], trainable=False)
    P_8 = tf.get_variable('P_8', dtype=tf.float32, shape=[VECTOR_SIZE, VECTOR_SIZE], trainable=False)
    P_9 = tf.get_variable('P_9', dtype=tf.float32, shape=[VECTOR_SIZE, VECTOR_SIZE], trainable=False)
    P_10 = tf.get_variable('P_10', dtype=tf.float32, shape=[VECTOR_SIZE, VECTOR_SIZE], trainable=False)
    P_11 = tf.get_variable('P_11', dtype=tf.float32, shape=[VECTOR_SIZE, VECTOR_SIZE], trainable=False)
    P_12 = tf.get_variable('P_12', dtype=tf.float32, shape=[VECTOR_SIZE, VECTOR_SIZE], trainable=False)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, model_path)
        log('Model restored.')
        mu = tf.zeros_like(x)
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
            mu = tf.nn.tanh(tf.matmul(x, W_1) + l1 + l2 + l3 + l4)
        y = tf.nn.dropout(tf.matmul(tf.matmul(tf.matmul(mu, W_2), W_3), W_4), keep_prob)
        if TEST_FROM == 'graph':
            _, _, list_file_test = get_file_list(target_obj_type)
            log(list_file_test)
            for file_graph in list_file_test:
                log(file_graph)
                list_vector, list_label, ln_matrix, rn_matrix, lp_matrix, rp_matrix, dict_idx_to_addr, set_obj_addr = read_dataset(file_graph, output_vector_size, dict_key_node_to_weight, target_obj_type)
                log('Test using ' + file_graph)
                ye = sess.run(y, feed_dict={x:list_vector, ln:ln_matrix, rn:rn_matrix, lp:lp_matrix, rp:rp_matrix, keep_prob:1})
                log('get_addr_to_weight')
                dict_addr_to_weight = get_addr_to_weight(ye, dict_idx_to_addr, dict_key_node_to_weight)
                log('stat_result_type')
                if ROC == True:
                    stat_result_type(dict_addr_to_weight, set_obj_addr, target_obj_type, dict_key_node_to_weight, threshold=0)
                else:
                    stat_result_type(dict_addr_to_weight, set_obj_addr, target_obj_type, dict_key_node_to_weight, threshold=obj_type_threshold)
        else:
            if TEST_FROM == 'dump':
                list_file_test_dump = get_dump_file_list(target_obj_type)
                log(str(list_file_test_dump))
                threshold = obj_type_threshold
            elif TEST_FROM == 'mutate_diff':
                list_file_test_dump = get_mutate_file_list(target_obj_type)
                log(str(list_file_test_dump))
                threshold = 1.0
            elif TEST_FROM == 'mutate_pooltag':
                list_file_test_dump = get_mutate_pooltag_file_list(target_obj_type)
                log(str(list_file_test_dump))
                threshold = obj_type_threshold
            elif TEST_FROM == 'mutate_link':
                list_file_test_dump = get_mutate_link_file_list(target_obj_type)
                log(str(list_file_test_dump))
                threshold = 1
            else:
                log('parameter TEST_FROM error')
                exit()
            for file_dump in list_file_test_dump:
                log('#1. start generating page info')
                cmd_get_pages = 'cd ../volatility; python vol.py -f ' + file_dump + ' --profile=Win7SP1x86 get_physical_pages; cd -'
                log(cmd_get_pages)
                os.system(cmd_get_pages)
                log('#2. start generating graph')
                log('Graph file doesn\'t exist. Generate it first...')
                cmd_build_graph = 'cd ../create_memory_graphs/; python graph.py ' + file_dump + ' test; cd -'
                log(cmd_build_graph)
                os.system(cmd_build_graph)
                log('#3. start read graph ')
                file_dump_basename = os.path.basename(file_dump)
                file_graph = OUTPUT_PATH + 'graph.' + file_dump_basename.replace('memdump_', '').replace('.raw', '') + '.all'
                list_vector, list_label, ln_matrix, rn_matrix, lp_matrix, rp_matrix, dict_idx_to_addr, set_obj_addr = read_dataset(file_graph, output_vector_size, dict_key_node_to_weight, target_obj_type)
                log('#4. start sess.run')
                log('Test using ' + file_graph)
                ye = sess.run(y, feed_dict={x:list_vector, ln:ln_matrix, rn:rn_matrix, lp:lp_matrix, rp:rp_matrix, keep_prob:1})
                log('#5. get_addr_to_weight')
                dict_addr_to_weight = get_addr_to_weight(ye, dict_idx_to_addr, dict_key_node_to_weight)
                log('#6. stat_result_type')
                stat_result_type(dict_addr_to_weight, set_obj_addr, target_obj_type, dict_key_node_to_weight, threshold=threshold)
                log('#7. finish')

def log(message):
    print('%s\t%s' %(strftime("%Y-%m-%d %H:%M:%S", gmtime()), message))
    sys.stdout.flush()

def load_threshold(target_obj_type):
    dict_type_to_threshold = {}
    with open(OBJ_TPE_THRESHOLD_PATH, 'r') as f:
        for line in f:
            s = line.strip().split('\t')
            obj_type = s[0]
            threshold = float(s[1])
            dict_type_to_threshold[obj_type] = threshold
    return dict_type_to_threshold[target_obj_type]

def get_dump_file_list(target_obj_type):
    list_file = [x for x in os.listdir(DUMP_PATH)]
    list_file.sort()
    return [DUMP_PATH + x for x in list_file][4:5]

def get_mutate_file_list(target_obj_type):
    list_file = [x for x in os.listdir(MUTATE_DUMP_PATH) if 'mutate.' + target_obj_type + '.count' in x]
    for f in list_file:
        ori_f = f.split('.mutate.')[0]
        if ori_f not in list_file:
            list_file.append(ori_f)
    list_file.sort()
    return [MUTATE_DUMP_PATH + x for x in list_file]
    
def get_mutate_pooltag_file_list(target_obj_type):
    list_file = [x for x in os.listdir(MUTATE_DUMP_PATH) if target_obj_type + '.pooltag' in x]
    for f in list_file:
        ori_f = f.split('.mutate.')[0]
        if ori_f not in list_file:
            list_file.append(ori_f)
    list_file.sort()
    return [MUTATE_DUMP_PATH + x for x in list_file]
    
def get_mutate_link_file_list(target_obj_type):
    list_file = [x for x in os.listdir(MUTATE_DUMP_PATH) if target_obj_type + '.link' in x]
    for f in list_file:
        ori_f = f.split('.mutate.')[0]
        if ori_f not in list_file:
            list_file.append(ori_f)
    list_file.sort()
    return [MUTATE_DUMP_PATH + x for x in list_file]
    
if __name__ == '__main__':
    main()
