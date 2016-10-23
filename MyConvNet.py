import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from MyNet import weight_variable,bias_variable,conv_layer,pooling_layer,relu_layer,fully_connected,softmax_layer
from MyNet import upsampling_layer,deconv_layer
from Readim import MyImages

####################### INITIALIZATION START #######################
batch_size=1
h=500
w=500
c=1
xs = tf.placeholder(tf.float32,[batch_size,h,w,c],name='xs')
ys = tf.placeholder(tf.float32,[batch_size,h,w,c],name='ys')
####################### INITIALIZATION END #######################

########## LAYER DEFINITION START ##########
# layer 1
conv1 = conv_layer(xs, [5,5,1,6], [6], name='conv1') #[500*500*1]->[500*500*6]
pool1 = pooling_layer(conv1, name='pool1') # [2500*500*6]->[250*250*6]
relu1 = relu_layer(pool1,name='relu1')

# layer 2
conv2 = conv_layer(relu1, [5,5,6,16], [16], name='conv2') # [250*250*6]->[250*250*16]
pool2 = pooling_layer(conv2, name='pool2') # [250*250*16]->[125*125*16]
relu2 = relu_layer(pool2, name='relu2')

# layer 3
upsample3 = upsampling_layer(relu2, 250, 250, name='upsample3') # [125*125*16]->[250*250*16]
conv3 = conv_layer(upsample3, [5,5,16,6], [6], name='conv3') # [250*250*16]->[250*250*6]
relu3 = relu_layer(conv3, name='relu3')

# layer 4
upsample4 = upsampling_layer(relu3, 500, 500, name='upsample4') # [250*250*6]->[500*500*6]
conv4 = conv_layer(upsample4, [5,5,6,1], [1], name='conv4') # [500*500*6]->[500*500*1]
relu4 = relu_layer(conv4, name='relu4')
relu4_reshape = tf.reshape(relu4,[-1,1])

# layer 4.1
#deconv4 = deconv_layer(relu4, [batch_size,h,w,c], [5,5,1,1], [1], name='deconv4')
#relu4 = relu_layer(deconv4, name='relu4')

# layer 5 (entropy)
# training solver
with tf.name_scope('loss'):
    ys_reshape = tf.reshape(ys,[-1,1])
    prediction = tf.reshape(relu4,[-1,1])

    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(prediction,ys_reshape)
    cross_entropy = tf.reduce_mean(cross_entropy)
    tf.scalar_summary('loss',cross_entropy)

with tf.name_scope('solver'):
    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
########## LAYER DEFINITION END ##########


# start training
# read images and labels
imdir='/Users/ful6ru04/Documents/TensorFlow workspace/Mycnn/SPINE_data'
myImages = MyImages()
myImages.build(imdir,batch_size,'png')

sess = tf.Session()
merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter("logs/", sess.graph)
sess.run(tf.initialize_all_variables())

for step in range(500):
    batch_xs, batch_ys = myImages.nextBatch()
    
    _, loss_value = sess.run([train_step, cross_entropy],feed_dict={xs:batch_xs, ys:batch_ys})
    print(loss_value)
    print(_)

    print('step '+str(step))
    results = sess.run(merged,feed_dict={xs:batch_xs, ys:batch_ys})
    writer.add_summary(results,step)

sess.close()
