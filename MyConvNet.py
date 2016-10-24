import tensorflow as tf

from MyNet import conv_layer,pooling_layer,relu_layer
from MyNet import upsampling_layer
from Readim import MyImages



####################### INITIALIZATION START #######################
batch_size = 1
new_height = 500
new_width  = 500
channels   = 1
learning_rate = 0.01

imdir = '/Users/ful6ru04/Documents/TensorFlow workspace/Mycnn/SPINE_data'
myImages = MyImages()
myImages.build( imdir, batch_size, new_height=new_height, new_width=new_width )

xs = tf.placeholder(tf.float32,
                    [batch_size, new_height, new_width, channels], name='xs')
#xs2 = tf.squeeze(xs)
#print(xs2)
ys = tf.placeholder(tf.float32,
                    [batch_size, new_height, new_width, channels], name='ys')

####################### INITIALIZATION END #######################

########## LAYER DEFINITION START ##########
### Encoder ###
# layer 1
conv1_1 = conv_layer(xs, [3,3,1,64], [64], name='conv1_1') #[500*500*1]->[500*500*64]
relu1_1 = relu_layer(conv1_1,name='relu1_1')
conv1_2 = conv_layer(relu1_1, [3,3,64,64], [64], name='conv1_2') #[500*500*64]->[500*500*64]
relu1_2 = relu_layer(conv1_2,name='relu1_2')
pool1_1 = pooling_layer(relu1_2, name='pool1_1') # [500*500*64]->[250*250*64]

#down_color = utils.color_image(down[0])
#scp.misc.imsave('fcn32_downsampled.png', down_color)

# layer 2
conv2_1 = conv_layer(pool1_1, [3,3,64,128], [128], name='conv2_1') # [250*250*64]->[250*250*128]
relu2_1 = relu_layer(conv2_1,name='relu2_1')
conv2_2 = conv_layer(relu2_1, [3,3,128,128], [128], name='conv2_2') # [250*250*128]->[250*250*128]
relu2_2 = relu_layer(conv2_2,name='relu2_2')
pool2_1 = pooling_layer(relu2_2, name='pool2_1') # [250*250*128]->[125*125*128]


### Decoder ###
# layer 3
upsample3_1 = upsampling_layer(pool2_1, 250, 250, name='upsample3_1') # [125*125*128]->[250*250*128]
deconv3_1 = conv_layer(upsample3_1, [3,3,128,128], [128], name='deconv3_1') # [250*250*128]->[250*250*128]
relu3_1 = relu_layer(deconv3_1,name='relu3_1')
deconv3_2 = conv_layer(relu3_1, [3,3,128,64], [64], name='deconv3_2') # [250*250*128]->[250*250*64]
relu3_2 = relu_layer(deconv3_2, name='relu3_2')

# layer 4
upsample4_1 = upsampling_layer(relu3_2, 500, 500, name='upsample4_1') # [250*250*64]->[500*500*64]
deconv4_1 = conv_layer(upsample4_1, [3,3,64,64], [64], name='deconv4_1') # [500*500*64]->[500*500*64]
relu4_1 = relu_layer(deconv4_1,name='relu4_1')
deconv4_2 = conv_layer(relu4_1, [3,3,64,1], [1], name='deconv4_2') # [500*500*64]->[500*500*1]
relu4_2 = relu_layer(deconv4_2, name='relu4_2')
relu4_2_reshape = tf.reshape(relu4_2,[-1,1])

# layer 5 (entropy)
# training solver
with tf.name_scope('loss'):
    ys_reshape = tf.reshape(ys,[-1,1])
    prediction = tf.reshape(relu4_2_reshape,[-1,1])

    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(prediction,ys_reshape)
    cross_entropy = tf.reduce_mean(cross_entropy)
    tf.scalar_summary('loss',cross_entropy)

with tf.name_scope('solver'):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
########## LAYER DEFINITION END ##########

# start training
sess = tf.Session()
merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter("logs/", sess.graph)
sess.run(tf.initialize_all_variables())

for step in range(5):
    
    print('step '+str(step))
    
    batch_xs, batch_ys = myImages.nextBatch()
    _, loss_value = sess.run([train_step, cross_entropy],
                             feed_dict={xs:batch_xs, ys:batch_ys})
    print(loss_value)
    #tf.image_summary('xs', xs2, max_images=2, collections=None, name='input'+str(step))

    results = sess.run(merged,feed_dict={xs:batch_xs, ys:batch_ys})
    writer.add_summary(results,step)
sess.close()
