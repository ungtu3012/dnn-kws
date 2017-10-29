import tensorflow as tf 

hyper_params = {
    'batch_size': 28,
    'num_features': 1000
    'num_hiddens_fc': [200, 200],
    'num_classes': 28,
    'learning_rate': 1e-2
}

def create_graph(hyper_params=hyper_params):
    tf.reset_default_graph()

    graph = tf.Graph()

    with graph.as_default():
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

        with tf.name_scope('input') as scope:
            X = tf.placeholder(tf.float32, [None, hyper_params['num_features']]) 
            Y = tf.placeholder(tf.int32, [None])

        with tf.name_scope('fc') as scope:
            i = 0
            out = X
            last_hidden = hyper_params['num_hiddens_fc'][0]
            for num_hidden in hyper_params['num_hiddens_fc']:
                i += 1
                W = tf.Variable(tf.truncated_normal([last_hidden, num_hidden],stddev=0.1, name='W_%d' % i))
                b = tf.Variable(tf.constant(0., shape=[num_hidden], name='b_%d' % i))
                out = tf.nn.relu(tf.nn.xw_plus_b(out, W, b))
                last_hidden = num_hidden
            i += 1
            num_output = hyper_params['num_classes']
            W = tf.Variable(tf.truncated_normal([last_hidden, num_output],stddev=0.1, name='W_%d' % i))
            b = tf.Variable(tf.constant(0., shape=[num_output], name='b_%d' % i))
            logits = tf.nn.xw_plus_b(out, W, b)

        with tf.name_scope('fc_loss_function') as scope:
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=logits, name='xentropy')
            cost = tf.reduce_mean(cross_entropy, name='xentropy_mean')

        optimizer = tf.train.AdamOptimizer(hyper_params['learning_rate']).minimize(cost, global_step=global_step)

        with tf.name_scope('loss_by_step') as scope:
            tf.summary.scalar('cost', cost)
            summary_op = tf.summary.merge_all()

    model_vars = {
        'X': X, 'Y': Y, 'global_step': global_step, 'cost': cost,
        'summary_op': summary_op, 'optimizer': optimizer,
        'logits': logits
    }
    return graph, model_vars