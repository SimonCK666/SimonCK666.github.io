---
layout: post
title: "Tensorflow Basics"
---
- [Showing and Evluating Tensor Values](#tval)  
- [Saving and restoration of Tensorflow Models - checkpoint](#ckpt)  
  - [What is a Tensorflow model](#what)  
  - [Saving a Tensorflow model](#saving)  
  - [Importing a pre-trained model](#import)    
  - [Working with restored models](#working)   
    - [VGG Example](#VGG)  
  
## <a name="tval"></a> Showing and Evluating Tensor Values  
The easiest way to evaluate the actual value of a Tensor object is to pass it to the **Session.run()** method, or call **Tensor.eval()** when you have a default session (i.e. in a with tf.Session(): block). **In general, you cannot print the value of a tensor without running some code in a session**.

If you are experimenting with the programming model, and want an easy way to evaluate tensors, the tf.InteractiveSession lets you open a session at the start of your program, and then use that session for all Tensor.eval() (and Operation.run()) calls. This can be easier in an interactive setting, such as the shell or an IPython notebook, when it's tedious to pass around a Session object everywhere.

This might seem silly for such a small expression, but one of the key ideas in Tensorflow is **deferred execution**: it's very cheap to build a large and complex expression, and when you want to evaluate it, the back-end (to which you connect with a Session) is able to schedule its execution more efficiently (e.g. executing independent parts in parallel and using GPUs).

## <a name="ckpt"></a> Saving and restoration of Tensorflow Models - checkpoint
### <a name="what"></a> What is a Tensorflow model 
Tensorflow model primarily contains the network design or graph and values of the network parameters that we have trained. Hence, Tensorflow model has two main files:  
* **Meta graph**: This is a protocol buffer which saves the complete Tensorflow graph; i.e. all variables, operations, collections etc. This file has .meta extension.  
* **Checkpoint file**: This is a binary file which contains all the values of the weights, biases, gradients and all the other variables saved. This file has an extension .ckpt. However, Tensorflow has changed this from version 0.11. Now, instead of single .ckpt file, we have two files:  
> mymodel.data-00000-of-00001  
> mymodel.index  

Along with this, Tensorflow also has a file named _checkpoint_ which simply keeps a record of latest checkpoint files saved.  

### <a name="saving"></a> Saving a Tensorflow model
Let’s say, you are training a convolutional neural network for image classification. As a standard practice, you keep a watch on loss and accuracy numbers. Once you see that the network has converged, you can stop the training manually or you will run the training for fixed number of epochs. After the training is done, we want to save all the **variables** and **network graph** to a file for future use. So, in Tensorflow, you want to save the graph and values of all the parameters for which we shall be creating an instance of **tf.train.Saver()** class.
```python
saver= tf.train.Saver()
```
Remember that **Tensorflow variables are only alive inside a session**. So, you have to save the model inside a session by calling save method on saver object you just created.
```python
saver.save(sess, 'my-test-model')
```
Here, _sess_ is the session object, while _‘my-test-model’_ is the name you want to give your model. Let’s see a complete example:
```python
import tensorflow as tf
w1 = tf.Variable(tf.random_normal(shape=[2]), name='w1')
w2 = tf.Variable(tf.random_normal(shape=[5]), name='w2')
saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver.save(sess, 'my_test_model')

# This will save following files in Tensorflow v >= 0.11
# my_test_model.data-00000-of-00001
# my_test_model.index
# my_test_model.meta
# checkpoint
```
If we are saving the model after 1000 iterations, we shall call save by passing the step count:
```python
saver.save(sess, 'my_test_model',global_step=1000)
```
This will just append ‘-1000’ to the model name and following files will be created:  
> my_test_model-1000.index  
> my_test_model-1000.meta  
> my_test_model-1000.data-00000-of-00001  
> checkpoint  
Let’s say, while training, we are saving our model after every 1000 iterations, so _.meta_ file is created the first time(on 1000th iteration) and we don’t need to recreate the _.meta_ file each time(so, we don’t save the .meta file at 2000, 3000.. or any other iteration). We only save the model for further iterations, as the graph will not change. Hence, when we don’t want to write the meta-graph we use this:
```python
saver.save(sess, 'my-model', global_step=step, write_meta_graph=False)
```
If you want to keep only 4 latest models and want to save one model after every 2 hours during training you can use max_to_keep and keep_checkpoint_every_n_hours like this.
```python
#saves a model every 2 hours and maximum 4 latest models are saved.
saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)
```
**Note, if we don’t specify anything in the tf.train.Saver(), it saves all the variables**. What if, we don’t want to save all the variables and just some of them. We can specify the variables/collections we want to save. While creating the tf.train.Saver instance we pass it a list or a dictionary of variables that we want to save. Let’s look at an example:
```python
import tensorflow as tf
w1 = tf.Variable(tf.random_normal(shape=[2]), name='w1')
w2 = tf.Variable(tf.random_normal(shape=[5]), name='w2')
saver = tf.train.Saver([w1,w2])
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver.save(sess, 'my_test_model', global_step=1000)
```
This can be used to save specific part of Tensorflow graphs when required.

### <a name="import"></a>Importing a pre-trained model:
If you want to use someone else’s pre-trained model for fine-tuning, there are two things you need to do:
#### a) Create the network 
You can create the network by writing python code to **create each and every layer manually as the original model**. However, if you think about it, we had saved the network in _.meta_ file which we can use to recreate the network using **tf.train.import()** function like this:
```python 
saver = tf.train.import_meta_graph('my_test_model-1000.meta')**
```
Remember, _import_meta_graph_ appends the network defined in .meta file to the current graph. So, this will create the graph/network for you but we still need to load the value of the parameters that we had trained on this graph.
#### Load the parameters:
We can restore the parameters of the network by calling restore on this saver which is an instance of **tf.train.Saver() class**.
```python
with tf.Session() as sess:
  new_saver = tf.train.import_meta_graph('my_test_model-1000.meta')
  new_saver.restore(sess, tf.train.latest_checkpoint('./'))
```
After this, the value of tensors like w1 and w2 has been restored and can be accessed:
```python
with tf.Session() as sess:    
  saver = tf.train.import_meta_graph('my-model-1000.meta')
  saver.restore(sess,tf.train.latest_checkpoint('./'))
  print(sess.run('w1:0'))
## Model has been restored. Above statement will print the saved value of w1.
```
So, now you have understood how saving and importing works for a Tensorflow model.  

### <a name="working"></a>Working with restored models

Now that you have understood how to save and restore Tensorflow models, Let’s develop a practical guide to restore any pre-trained model and use it for prediction, fine-tuning or further training. Whenever you are working with Tensorflow, you define a graph which is fed examples(training data) and some hyperparameters like learning rate, global step etc. It’s a standard practice to feed all the training data and hyperparameters using _placeholders_. Let’s build a small network using placeholders and save it. **Note that when the network is saved, values of the placeholders are not saved**.
```python
import tensorflow as tf

#Prepare to feed input, i.e. feed_dict and placeholders
w1 = tf.placeholder("float", name="w1")
w2 = tf.placeholder("float", name="w2")
b1= tf.Variable(2.0,name="bias")
feed_dict ={w1:4,w2:8}

#Define a test operation that we will restore
w3 = tf.add(w1,w2)
w4 = tf.multiply(w3,b1,name="op_to_restore")
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#Create a saver object which will save all the variables
saver = tf.train.Saver()

#Run the operation by feeding input
print sess.run(w4,feed_dict)
#Prints 24 which is sum of (w1+w2)*b1 

#Now, save the graph
saver.save(sess, 'my_test_model', global_step=1000)
```
Now, when we want to restore it, we not only have to restore the graph and weights, but also prepare a new feed_dict that will feed the new training data to the network. We can get reference to these saved operations and placeholder variables via **graph.get_tensor_by_name()** method.
```python
#How to access saved variable/Tensor/placeholders 
w1 = graph.get_tensor_by_name("w1:0")

## How to access saved operation
op_to_restore = graph.get_tensor_by_name("op_to_restore:0")
```
If we just want to run the same network with different data, you can simply pass the new data via feed_dict to the network.
```python
import tensorflow as tf
sess=tf.Session()    
# First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('my_test_model-1000.meta')
saver.restore(sess,tf.train.latest_checkpoint('./'))

# Now, let's access and create placeholders variables and create feed_dict to feed new data
graph = tf.get_default_graph()
w1 = graph.get_tensor_by_name("w1:0")
w2 = graph.get_tensor_by_name("w2:0")
feed_dict ={w1:13.0,w2:17.0}

#Now, access the op that you want to run. 
op_to_restore = graph.get_tensor_by_name("op_to_restore:0")

print sess.run(op_to_restore, feed_dict)
# This will print 60 which is calculated using new values of w1 and w2 and saved value of b1. 
```
What if you want to add more operations to the graph by adding more layers and then train it. Of course you can do that too. See here:
```python
import tensorflow as tf

sess=tf.Session()    
#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('my_test_model-1000.meta')
saver.restore(sess,tf.train.latest_checkpoint('./'))

# Now, let's access and create placeholders variables and
# create feed-dict to feed new data
graph = tf.get_default_graph()
w1 = graph.get_tensor_by_name("w1:0")
w2 = graph.get_tensor_by_name("w2:0")
feed_dict ={w1:13.0,w2:17.0}

#Now, access the op that you want to run. 
op_to_restore = graph.get_tensor_by_name("op_to_restore:0")

#Add more to the current graph
add_on_op = tf.multiply(op_to_restore,2)

print sess.run(add_on_op,feed_dict)
#This will print 120.
```
#### <a name="VGG"></a> VGG Example
How to restore part of the old graph and add-on to that for fine-tuning: just access the appropriate operation by **graph.get_tensor_by_name()** method and build graph on top of that. Here is a real world example. Here we load a vgg pre-trained network using meta graph and change the number of outputs to 2 in the last layer for fine-tuning with new data.
```python
......
......
saver = tf.train.import_meta_graph('vgg.meta')
# Access the graph
graph = tf.get_default_graph()
## Prepare the feed_dict for feeding data for fine-tuning 

# Access the appropriate output for fine-tuning
fc7= graph.get_tensor_by_name('fc7:0')

# use this if you only want to change gradients of the last layer
fc7 = tf.stop_gradient(fc7) # It's an identity function
fc7_shape= fc7.get_shape().as_list()

new_outputs=2
weights = tf.Variable(tf.truncated_normal([fc7_shape[3], num_outputs], stddev=0.05))
biases = tf.Variable(tf.constant(0.05, shape=[num_outputs]))
output = tf.matmul(fc7, weights) + biases
pred = tf.nn.softmax(output)

# Now, you run this with fine-tuning data in sess.run()
```


