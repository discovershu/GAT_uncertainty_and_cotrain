import time
import numpy as np
import tensorflow as tf

from models import GAT
from utils import process

# Set random seed
seed = 1
# np.random.seed(seed)
# tf.set_random_seed(seed)
# result_test = []
# result_Baye = []
# for k in range(10):

checkpt_file = 'pre_trained/cora/mod_cora_teacher{}.ckpt'.format(seed)


dataset = 'cora'
gat_pred_data = np.load("/network/rit/lab/ceashpc/sharedata_shu/gat_baseline/gat_{}_{}.npy".format(dataset, seed))

# training params
batch_size = 1
nb_epochs = 100000#100000
patience = 100
lr = 0.005  # learning rate
l2_coef = 0.0005  # weight decay
hid_units = [8] # numbers of hidden units per each attention head in each layer
n_heads = [8, 1] # additional entry for the output layer
residual = False
nonlinearity = tf.nn.elu
model = GAT

print('Dataset: ' + dataset)
print('----- Opt. hyperparams -----')
print('lr: ' + str(lr))
print('l2_coef: ' + str(l2_coef))
print('----- Archi. hyperparams -----')
print('nb. layers: ' + str(len(hid_units)))
print('nb. units per layer: ' + str(hid_units))
print('nb. attention heads: ' + str(n_heads))
print('residual: ' + str(residual))
print('nonlinearity: ' + str(nonlinearity))
print('model: ' + str(model))

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = process.load_data(dataset)
features, spars = process.preprocess_features(features)

nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = y_train.shape[1]

adj = adj.todense()


features = features[np.newaxis]
adj = adj[np.newaxis]
y_train = y_train[np.newaxis]
y_val = y_val[np.newaxis]
y_test_real = y_test
test_mask_real = test_mask
y_test = y_test[np.newaxis]
train_mask = train_mask[np.newaxis]
val_mask = val_mask[np.newaxis]
test_mask = test_mask[np.newaxis]

biases = process.adj_to_bias(adj, [nb_nodes], nhood=1)

with tf.Graph().as_default():
    with tf.name_scope('input'):
        ftr_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, ft_size))
        bias_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, nb_nodes))
        lbl_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes, nb_classes))
        msk_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes))
        attn_drop = tf.placeholder(dtype=tf.float32, shape=())
        ffd_drop = tf.placeholder(dtype=tf.float32, shape=())
        is_train = tf.placeholder(dtype=tf.bool, shape=())
        annealing_step = tf.placeholder(tf.float32)
        gat_pred = tf.placeholder(dtype=tf.float32, shape=(nb_nodes, nb_classes))

    #result = []
    # seed = 123
    # np.random.seed(seed)
    # tf.set_random_seed(seed)
    # for k in range(3):
    logits = model.inference(ftr_in, nb_classes, nb_nodes, is_train,
                                attn_drop, ffd_drop,
                                bias_mat=bias_in,
                                hid_units=hid_units, n_heads=n_heads,
                                residual=residual, activation=nonlinearity)
    log_resh = tf.reshape(logits, [-1, nb_classes])
    lab_resh = tf.reshape(lbl_in, [-1, nb_classes])
    msk_resh = tf.reshape(msk_in, [-1])
    # loss = model.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
    loss = model.masked_square_error_dirichlet(log_resh, tf.cast(lab_resh, tf.float32), msk_resh)+ tf.minimum(1.0, annealing_step / 50.0)* model.masked_kl_teacher(log_resh, gat_pred)
    accuracy = model.masked_accuracy(log_resh, lab_resh, msk_resh)

    train_op = model.training(loss, lr, l2_coef)

    saver = tf.train.Saver()

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    vlss_mn = np.inf
    vacc_mx = 0.0
    curr_step = 0

    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # with tf.Session(config=config) as sess:
    with tf.Session() as sess:
        # seed = 123
        # np.random.seed(seed)
        # tf.set_random_seed(seed)

        sess.run(init_op)

        train_loss_avg = 0
        train_acc_avg = 0
        val_loss_avg = 0
        val_acc_avg = 0

        for epoch in range(nb_epochs):
            tr_step = 0
            tr_size = features.shape[0]

            while tr_step * batch_size < tr_size:
                _, loss_value_tr, acc_tr = sess.run([train_op, loss, accuracy],
                    feed_dict={
                        ftr_in: features[tr_step*batch_size:(tr_step+1)*batch_size],
                        bias_in: biases[tr_step*batch_size:(tr_step+1)*batch_size],
                        lbl_in: y_train[tr_step*batch_size:(tr_step+1)*batch_size],
                        msk_in: train_mask[tr_step*batch_size:(tr_step+1)*batch_size],
                        is_train: True,
                        annealing_step: epoch,
                        gat_pred: gat_pred_data,
                        attn_drop: 0.6, ffd_drop: 0.6})
                train_loss_avg += loss_value_tr
                train_acc_avg += acc_tr
                tr_step += 1
            vl_step = 0
            vl_size = features.shape[0]

            while vl_step * batch_size < vl_size:
                loss_value_vl, acc_vl, log_reshhs = sess.run([loss, accuracy, log_resh],
                    feed_dict={
                        ftr_in: features[vl_step*batch_size:(vl_step+1)*batch_size],
                        bias_in: biases[vl_step*batch_size:(vl_step+1)*batch_size],
                        lbl_in: y_val[vl_step*batch_size:(vl_step+1)*batch_size],
                        msk_in: val_mask[vl_step*batch_size:(vl_step+1)*batch_size],
                        is_train: False,
                        annealing_step: epoch,
                        gat_pred: gat_pred_data,
                        attn_drop: 0.0, ffd_drop: 0.0})
                val_loss_avg += loss_value_vl
                val_acc_avg += acc_vl
                vl_step += 1
            print("log_reshhs= ", log_reshhs)
            print('Training: loss = %.5f, acc = %.5f | Val: loss = %.5f, acc = %.5f | epoch = %d' %
                    (train_loss_avg/tr_step, train_acc_avg/tr_step,
                    val_loss_avg/vl_step, val_acc_avg/vl_step, epoch))

            if val_acc_avg/vl_step >= vacc_mx or val_loss_avg/vl_step <= vlss_mn:
                if val_acc_avg/vl_step >= vacc_mx and val_loss_avg/vl_step <= vlss_mn:
                    vacc_early_model = val_acc_avg/vl_step
                    vlss_early_model = val_loss_avg/vl_step
                    saver.save(sess, checkpt_file)
                vacc_mx = np.max((val_acc_avg/vl_step, vacc_mx))
                vlss_mn = np.min((val_loss_avg/vl_step, vlss_mn))
                curr_step = 0
            else:
                curr_step += 1
                if curr_step == patience:
                    print('Early stop! Min loss: ', vlss_mn, ', Max accuracy: ', vacc_mx)
                    print('Early stop model validation loss: ', vlss_early_model, ', accuracy: ', vacc_early_model)
                    break

            train_loss_avg = 0
            train_acc_avg = 0
            val_loss_avg = 0
            val_acc_avg = 0

        saver.restore(sess, checkpt_file)

        ts_size = features.shape[0]
        ts_step = 0
        ts_loss = 0.0
        ts_acc = 0.0

        while ts_step * batch_size < ts_size:
            loss_value_ts, acc_ts = sess.run([loss, accuracy],
                feed_dict={
                    ftr_in: features[ts_step*batch_size:(ts_step+1)*batch_size],
                    bias_in: biases[ts_step*batch_size:(ts_step+1)*batch_size],
                    lbl_in: y_test[ts_step*batch_size:(ts_step+1)*batch_size],
                    msk_in: test_mask[ts_step*batch_size:(ts_step+1)*batch_size],
                    is_train: False,
                    annealing_step: epoch,
                    gat_pred: gat_pred_data,
                    attn_drop: 0.0, ffd_drop: 0.0})
            ts_loss += loss_value_ts
            ts_acc += acc_ts
            ts_step += 1

        #result_test.append(ts_acc / ts_step)
        print('Test loss:', ts_loss/ts_step, '; Test accuracy:', ts_acc/ts_step)

        Baye_result = []
        for p in range(500):
            Bayets_size = features.shape[0]
            Bayets_step = 0
            Bayets_loss = 0.0
            Bayets_acc = 0.0
            while Bayets_step * batch_size < Bayets_size:
                loss_value_Bayets, acc_Bayets, Baye_output = sess.run([loss, accuracy, log_resh],
                     feed_dict={
                         ftr_in: features[Bayets_step * batch_size:(Bayets_step + 1) * batch_size],
                         bias_in: biases[Bayets_step * batch_size:(Bayets_step + 1) * batch_size],
                         lbl_in: y_test[Bayets_step * batch_size:(Bayets_step + 1) * batch_size],
                         msk_in: test_mask[Bayets_step * batch_size:(Bayets_step + 1) * batch_size],
                         is_train: False,
                         annealing_step: epoch,
                         gat_pred: gat_pred_data,
                         attn_drop: 0.6, ffd_drop: 0.6})
                Bayets_loss += loss_value_Bayets
                Bayets_acc += acc_Bayets
                Bayets_step += 1
            Baye_result.append(Baye_output)

        Baye_acc = process.masked_accuracy_numpy(np.mean(Baye_result, axis=0), y_test_real, test_mask_real)
        print("Baye accuracy=", "{:.5f}".format(Baye_acc))

        #result_Baye.append(Baye_acc)

        sess.close()
# print('Test result', result_test)
# print("Test accuracy mean=", "{:.5f}".format(np.mean(result_test)), "Test accuracy std=",
#       "{:.5f}".format(np.std(result_test)))
# print('Baye result', Baye_result)
# print("Baye accuracy mean=", "{:.5f}".format(np.mean(Baye_result)), "Baye accuracy std=",
#       "{:.5f}".format(np.std(Baye_result)))
    # print('Test1 withtopseed seed', result)
    # print('Test1 mean', np.mean(result))
