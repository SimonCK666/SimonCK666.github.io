---
layout: page
title: "Tensorflow-Datasets and piplines"
---

[](#top)
- [The TensorFlow Dataset framework – main components](#main)
- [Simple TensorFlow Dataset examples](#simple)

Consuming data efficiently becomes really paramount to training performance in deep learning. This is an important topic which isn’t covered very well in most TensorFlow tutorials – rather, these tutorials will often use the `feed_dict` and `placeholder` method of feeding data into the model. This method of feeding data into your network in TensorFlow is inefficient and will likely slow down your training for large, realistic datasets. Why is this framework better than the `feed_dict` method that is so commonly used? Simply, all of the operations to transform data and feed it into the model which can be performed with the Dataset API i.e. reading the data from arrays and files, transforming it, shuffling it etc. can all be automatically optimized and paralleled to provide efficient consumption of data.  

## <a name="main"></a> The TensorFlow Dataset framework – main components
The TensorFlow Dataset framework has two main components:
- The **Dataset**  
- An associated **Iterator**  
The **Dataset** is basically where the data resides. This data can be loaded in from a number of sources – existing tensors, numpy arrays and numpy files, the `TFRecord` format and direct from text files. Once you’ve loaded the data into the Dataset object, you can string together various operations to apply to the data, these include operations such as:  
- `batch()` – this allows you to consume the data from your TensorFlow Dataset in batches  
- `map()` – this allows you to transform the data using lambda statements applied to each element    
- `zip()` – this allows you to zip together different Dataset objects into a new Dataset, in a similar way to the Python zip function
- `filter()` – this allows you to remove problematic data-points in your data-set, again based on some lambda function  
- `repeat()` – this operation restricts the number of times data is consumed from the Dataset before a tf.errors.OutOfRangeError error is thrown  
- `shuffle()` – this operation shuffles the data in the Dataset  
There are many other methods that the Dataset API includes. The next component in the TensorFlow Dataset framework is the **Iterator**. This creates operations which can be called during the training, validation and/or testing of your model in TensorFlow.  
[[back to top]](#top)  

## <a name="simple"></a> Simple TensorFlow Dataset examples 
In the first simple example, we’ll create a dataset out of numpy ranges:
```python
x = np.arange(0, 10)
```
We can create a TensorFlow Dataset object straight from a numpy array using `from_tensor_slices()`:   
### create dataset object from numpy array
```python
dx = tf.data.Dataset.from_tensor_slices(x)
```
The object dx is now a TensorFlow Dataset object. The next step is to create an Iterator that will extract data from this dataset. In the code below, the iterator is created using the method `make_one_shot_iterator()`.  The iterator arising from this method can only be initialized and run once – it can’t be re-initialized. The importance of being able to re-initialize an iterator will be explained more later.  

### create a one-shot iterator
```python
iterator = dx.make_one_shot_iterator()
```
### extract an element
next_element = iterator.get_next()
After the iterator is created, the next step is to setup a TensorFlow operation which can be called from the training code to extract the next element from the dataset. Finally, the dataset operation can be examined by running the following code:
```python
with tf.Session() as sess:
    for i in range(11):
        val = sess.run(next_element)
        print(val)
```
This code will print out integers from 0 to 9 but then throw an `OutOfRangeError`. This is because the code extracted all the data slices from the dataset and it is now out of range or “empty”.  

If we want to repeatedly extract data from a dataset, one way we can do it is to make the dataset **re-initializable**. We can do that by first adjusting the `make_one_shot_iterator()` line to the following:
```python
iterator = dx.make_initializable_iterator()
```
Then, within the TensorFlow session, the code looks like this:
```python
with tf.Session() as sess:
    sess.run(iterator.initializer)
    for i in range(15):
        val = sess.run(next_element)
        print(val)
        if i % 9 == 0 and i > 0:
            sess.run(iterator.initializer)
```
Note that the first operation run is the iterator.initializer operation. This is required to get your iterator ready for action and if you don’t do this before running the next_element operation it will throw an error. The final change is the last two lines: this if statement ensures that when we know that the iterator has run out of data (i.e. i == 9), the iterator is re-initialized by the iterator.initializer operation. Running this new code will produce: 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4. No error this time!

There are also other things that can be done to manipulate the dataset and how it can be used. First, the batch function:
```python
dx = tf.data.Dataset.from_tensor_slices(x).batch(3)
```
After this change, when the next_element operation is run, a batch of length 3 will be extracted from the data. Running the code below:
```python
with tf.Session() as sess:
    sess.run(iterator.initializer)
    for i in range(15):
        val = sess.run(next_element)
        print(val)
        if (i + 1) % (10 // 3) == 0 and i > 0:
            sess.run(iterator.initializer)
```
Will produce an output like:  
> [0 1 2] [3 4 5] [6 7 8] [0 1 2] [3 4 5] [6 7 8]  
and so on.  

Next, we can zip together datasets. This is useful when pairing up input-output training/validation pairs of data (i.e. input images and matching labels for each image). The code below does this:
```python
def simple_zip_example():
    x = np.arange(0, 10)
    y = np.arange(1, 11)
    # create dataset objects from the arrays
    dx = tf.data.Dataset.from_tensor_slices(x)
    dy = tf.data.Dataset.from_tensor_slices(y)
    # zip the two datasets together
    dcomb = tf.data.Dataset.zip((dx, dy)).batch(3)
    iterator = dcomb.make_initializable_iterator()
    # extract an element
    next_element = iterator.get_next()
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        for i in range(15):
            val = sess.run(next_element)
            print(val)
            if (i + 1) % (10 // 3) == 0 and i > 0:
                sess.run(iterator.initializer)
```
The zip combination of the two datasets (dx, dy) can be seen in the line where dcomb is created. Note the chaining together of multiple operations – first the zip method, then the batching operation. The rest of the code is the same. This code will produce an output like the following:  
> (array([0, 1, 2]), array([1, 2, 3]))  
> (array([3, 4, 5]), array([4, 5, 6]))  
> (array([6, 7, 8]), array([7, 8, 9]))  
> (array([0, 1, 2]), array([1, 2, 3]))  
  
and so on. As you can observe, the batching takes place appropriately within the zipped together datasets i.e. 3 items from dx, 3 items from dy. As stated above, this is handy for combining input data and matching labels.  

Note, the re-initialization if statement on the last two lines is a bit unwieldy, we can actually get rid of it by replacing the dcomb dataset creation line with the following:  
```python
dcomb = tf.data.Dataset.zip((dx, dy)).repeat().batch(3)
```
Note the addition of the `repeat()` method to the operation list. When this method is applied to the dataset with no argument, it means that the dataset can be repeated indefinitely without throwing an OutOfRangeError. This will be shown in the next more detailed example – using the sci-kit learn MNIST dataset to create a hand-written digits classifier.
