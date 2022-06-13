---
layout: page
title: "Tensorflow-Pix2pix"
---

- [Exponential Moving Average (EMA)](#ema)  
- [Training supervisor](#sup)  
- [Control Dependencies](#depend)  
- [Variable scope](#scope)  
  
## <a name="ema"></a> Exponential Moving Average (EMA)
When training a model, it is often beneficial to maintain moving averages of the trained parameters. Evaluations that use averaged parameters sometimes produce significantly better results than the final trained values.   
> shadow_variable = decay * shadow_variable + (1 - decay) * variable  
```python  
ema = tf.train.ExponentialMovingAverage(decay=0.99)  
update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])  
```

## <a name="sup"></a> Training supervisor
A training helper that checkpoints models and computes summaries.  

> tf.train.Supervisor (newer version tf.train.MonitoredTrainingSession)  

The Supervisor is a small wrapper around a Coordinator, a Saver, and a SessionManager that takes care of common needs of TensorFlow training programs.

### Exporting and Serving Models with TensorFlow
Saver adds operations that allow us to save and restore the model’s parameters by using binary files called checkpoint files, mapping the tensor values to the names of the variables. (Here take note that the the saving use sess as the handle, the saved parameter values to have to exactly match the current graph when restored.)  
```python
saver = tf.train.Saver(max_to_keep=7, keep_checkpoint_every_n_hours=0.5)
```
## checkpoints and summary
You save the model to checkpoints because the Variables in the model, including neural network weights and biases and the global_step counter, keep changing during the training process. The structure of the model doesn't change. The saved checkpoints allow you to load the trained model for serving and to resume training later.

Supervisor帮助我们处理一些事情
（1）自动去checkpoint加载数据或初始化数据  
（2）自身有一个Saver，可以用来保存checkpoint  
（3）有一个summary_computed用来保存Summary  
所以，我们就不需要：  
（1）手动初始化或从checkpoint中加载数据  
（2）不需要创建Saver，使用sv内部的就可以  
（3）不需要创建summary writer  
  

## <a name="depend"></a> control_dependencies

TF可以**协调**多个数据流，在存在依赖的节点下非常有用，例如节点B要读取模型参数值V更新后的值，而节点A负责更新参数V，所以节点B就要等节点A执行完成后再执行，不然读到的就是更新以前的数据。这时候就需要个运算控制器 
```python
tf.control_dependencies(self, control_inputs)  
```
**Arguments**：control_inputs: A list of 'Operation' or 'Tensor' objects which must be executed or computed before running the operations defined in the context. (*note* control_inputs is a list)  
**Return**: A context manager that specifies control dependencies for all operations constructed within the context.  

Example 
```python
 with tf.control_dependencies([a, b, c]):
       # 'd' and 'e' will only run after 'a', 'b', and 'c' have executed.  
       d = ...  
       e = ...  
```

## <a name="scope"></a> Variable Scope 
name_scope, variable_scope目的：  
1. 减少训练参数的个数  
2. 区别同名变量  
  
为什么要共享变量？我举个简单的例子：例如，当我们研究生成对抗网络GAN的时候，判别器的任务是，如果接收到的是生成器生成的图像，判别器就尝试优化自己的网络结构来使自己输出0，如果接收到的是来自真实数据的图像，那么就尝试优化自己的网络结构来使自己输出1(两次调用，产生两个不同的图的节点)。也就是说，生成图像和真实图像经过判别器的时候，要共享同一套变量，所以TensorFlow引入了变量共享机制。  
  
**变量共享**主要涉及到两个函数： 
```python
tf.get_variable(<name>, <shape>, <initializer>)  
tf.variable_scope(<scope_name>)  
```
**tf.get_variable 和tf.Variable的区别**    
tf.get_variable 拥有一个**变量检查机制**，会检测已经存在的变量是否设置为共享变量，如果已经存在的变量没有设置为共享变量，TensorFlow 运行到第二个拥有相同名字的变量的时候，就会报错。  
例如如下代码：
```python
def my_image_filter(input_images):
    conv1_weights = tf.Variable(tf.random_normal([5, 5, 32, 32]),
        name="conv1_weights")
    conv1_biases = tf.Variable(tf.zeros([32]), name="conv1_biases")
    conv1 = tf.nn.conv2d(input_images, conv1_weights,
        strides=[1, 1, 1, 1], padding='SAME')
    return  tf.nn.relu(conv1 + conv1_biases)
```
有两个变量(Variables) conv1_weighs, conv1_biases和一个操作(Op) conv1，如果你直接调用两次，不会出什么问题，但是**会生成两套变量**；
```  
# First call creates one set of 2 variables.
result1 = my_image_filter(image1)
# Another set of 2 variables is created in the second call.
result2 = my_image_filter(image2)
```
如果把 tf.Variable 改成 tf.get_variable，直接调用两次，就会出问题了：
```python
result1 = my_image_filter(image1)
result2 = my_image_filter(image2)
# Raises ValueError(... conv1/weights already exists ...)
```
为了解决这个问题，TensorFlow 又提出了 tf.variable_scope 函数：它的主要作用是，**在一个作用域 scope 内共享一些变量**，
可以有如下几种用法
```python
with tf.variable_scope("image_filters") as scope:  
    result1 = my_image_filter(image1)  
    scope.reuse_variables() # or   
    #tf.get_variable_scope().reuse_variables()  
    result2 = my_image_filter(image2)  
```
需要注意的是：最好不要设置 reuse 标识为 False，只在需要的时候设置 reuse 标识为 True。  
```python
with tf.variable_scope("image_filters1") as scope1:  
    result1 = my_image_filter(image1)  
with tf.variable_scope(scope1, reuse = True)  
    result2 = my_image_filter(image2)  
``` 
通常情况下，tf.variable_scope 和 tf.name_scope 配合，能画出非常漂亮的流程图，但是他们两个之间又有着细微的差别，那就是 **name_scope 只能管住操作 Ops 的名字，而管不住变量 Variables 的名字**：
```python
with tf.variable_scope("foo"):  
    with tf.name_scope("bar"):  
        v = tf.get_variable("v", [1])  
        x = 1.0 + v  
assert v.name == "foo/v:0"  
assert x.op.name == "foo/bar/add"  
```
