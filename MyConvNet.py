import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from MyNet import weight_variable,bias_variable,conv_layer,pooling_layer,relu_layer,fully_connected,softmax_layer


def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict={xs:v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    acc = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(acc,feed_dict={xs:v_xs,ys:v_ys})
    return result


xs = tf.placeholder(tf.float32,[None,784],name='xs')
ys = tf.placeholder(tf.float32,[None, 10],name='ys')
x_img = tf.reshape(xs,[-1,28,28,1])

########## LAYER DEFINITION START ##########
# layer 1
conv1 = conv_layer(x_img, [5,5,1,6], [6], name='conv1') # [28*28*1]->[24*24*6]
pool1 = pooling_layer(conv1, name='pool1') # [24*24*6]->[12*12*6]
relu1 = relu_layer(pool1,name='relu1')

# layer 2
conv2 = conv_layer(relu1, [5,5,6,16], [16], name='conv2') # [12*12*6]->[8*8*16]
pool2 = pooling_layer(conv2, name='pool2') # [8*8*16]->[4*4*16]
relu2 = relu_layer(pool2, name='relu2')

# layer 3 (fc)
fc_in_size = (relu2.get_shape()[1]*relu2.get_shape()[2]*relu2.get_shape()[3]).value
fc3_w = weight_variable([fc_in_size,120],name='fc3/weight')
fc3_b = bias_variable([120])
relu2_col = tf.reshape(relu2,[-1,fc_in_size])
fc3 = fully_connected(relu2_col,fc3_w, name='fc3')
relu3 = relu_layer(fc3, name='relu3')

# layer 4 (fc)
fc4_w = weight_variable([120,10],name='fc4/weight')
fc4_b = bias_variable([10])
fc4 = fully_connected(relu3,fc4_w, name='fc4')
relu4 = relu_layer(fc4, name='relu4')

# layer 5 (prediction)
prediction = softmax_layer(relu4)

# training solver
with tf.name_scope('loss'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),
                                              reduction_indices=[1]))
    tf.scalar_summary('loss',cross_entropy)
with tf.name_scope('solver'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
########## LAYER DEFINITION END ##########


# start training
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.Session()
merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter("logs/", sess.graph)
sess.run(tf.initialize_all_variables())
for step in range(500):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs, ys:batch_ys})
    if step % 50 == 0:
        print( str(step)+': '+str(compute_accuracy(mnist.test.images, mnist.test.labels)) )
        results = sess.run(merged,feed_dict={xs:batch_xs, ys:batch_ys})
        writer.add_summary(results,step)
        
sess.close()


