# B_Class

[Tribe of AI - B Class - Deep Learninig](https://github.com/tribeofai/workshops/tree/master/B_Class)




## Sprint 1
### [Kaggle - Deeplearninig]()

**[Intro to DL for Computer Vision](https://www.kaggle.com/dansbecker/intro-to-dl-for-computer-vision)**

* Tensorflow - Dominant AI Engine
* Keras - popular AI api for specifing deep learninig models

**Images**:
- composed of pixels
- Rows and columns, baiscally a Matrix of numbers, where each number represents brightness of that pixel. (Grayscale image) 
![grayscale](grayscale.png)

- Colour images 3D (Blue, Green Red matrixes), where each pixel stores how Blue, Green or Red it is.

**Convolutions** - is a small tensor which can be applied of a little sections of an image. They are also called filters.


You don't directly choose the numbers to go into your convolutions for deep learning... instead the deep learning technique determines what convolutions will be useful from the data (as part of model-training)

**[Building Models From Convolutions](https://www.kaggle.com/dansbecker/building-models-from-convolutions)**

![aiview](aiview.png)

Such networks in images can help object detection. **IMAGENET** is a popular dataset with a lot of labeled images used to create models able to recognize new same labeled images at 80% accuracy. We can use Imagenet dataset to train new models, but we can also use already trained models for our predictions.

### FAST.AI

Practical Deep Learning for Coders, v3
**[Lesson 1: Image classification](https://course.fast.ai/videos/?lesson=1)**

- Pytorch popular DL library, fastai uses it.

- Fastai is library that supports: Vision, NLP(Text), tabular data and collaborative filtering.

Fine-grained classification - in basically where you have to distinguish between similar categories. (Like differentianting breed of dogs and cats, to which they belong)

For comupter vision we have to make **images of the same size and shape**. Mostly making images square. (Common size is 224) - Generaly this size will work most of the time.

**Why do you use normalize images?**
Some channels tend to be very bright, others not etc, it helps DL by making channels mean of 9 and std of 1.

if you data is not normalized, it may be problem for model to learn.

Restnet - (34 or 50) convolutional model which works mostly very well on classification problems. (Restnet is a Pretrained model, based on millions of data trained on imagenet)

Metrics, are just metrics which are printed out. We always print out metrics on validation set.

**Transfer learninig** - taking a model that already knows how to do something pretty well and make it so that it can do your thing really well.

So you take already trained model, fit it with your data and boom. You make a amazing model that took thousands or less of the time/data of regular model traininig.

in practice you use .fit_one_cycle() instead of .fit() (In 2018 best mothod)

```.fit_one_cycle(4)``` 4 - indicates that it will run 4 cycles.


**Power of Deep learninig**
Top researchers of Oxford in 2012 built best experimental models that predicted cat/dog breed with around 60% accuracy.

Today, with few lines of DL code, using transfer learninig, and 1-2 minutes of traininig, we can achieve 94% of accuracy.

**Keras** is alternative to FASTAI, a good library too.

**Loss function** - tells how good is your prediction. 

.most_confused(min_val=2) - grabs from confussion matrix which were mistaken and brings you back a list, based on your min value. Easier than confusion matrix to distinguish which were wrong.


```.unfreeze()``` - a Thing that says please train the whole model.

**Visualizing and understanding CNN**

* 1st layers. Simply finds gradients of few colours, or some lines, like horizontal vertical etc.

![layer1](layer1.png)

* 2nd layer. Takes the results of those filters and does a second layer of computation. Allows to create, learned to create, recognize corners, or circles or slightly more complex things.

![layer2](layer2.png)


* 3rd layer. You can find combinations of those of previous layers, even more complex structures. From examples parternns, car tyres, or something rounds, texts, people and more.. .

![layer3](layer3.png)

* 4/th layers could identified even more complex structures based on previosu computations. Like dog faces, like legs etc...

![layer4](layer4.png)

and by the time we get to high layers, model is capable of differentiating dog/cat breeds.

**Learning rate** - how quickly am I updating parameters in my model.


**batchsize (bs)**, you can indicate batchsize, to make traininig in batches, this removes error of memory problems.

### Deeplearning.ai

**[Introduction to Deep Learninig](https://www.coursera.org/learn/neural-networks-deep-learning)**


RELU - Rectified linear unit. (Taking a max of zero, thus values become positive?)

Example of simple NN where its gola to predict price based on inputs (Size,zip,bedrooms,etc)
![nn](nn.png)

Given enought traininig examples, NN are very good to make functions mapping A to Y.

So far most AI value came from **Supervised Learning.** Examples, below:

![sl](sl.png)

* For Home prices (Standard Neural networks)
* For images, CNN (Convolutional Neural Nets would be used)
* For Audio data, RNN (Recurrent Neural nets would be used)


**Unstructured data:** Audio, Images, Text. 
Historically it has been hard for machines to work with these data, but now with DL it got much easier, and thus new applications are available.

Why DL taking off?
* Large amount of Data (Labeled data for SL)
* Better algorithms (NN)
* Better computing

**m denote** - Number of training examples


### [An overview of gradient descent optimization algorithms](https://ruder.io/optimizing-gradient-descent/)

Gradient descent is one of the most popular algorithms to perform optimization and by far the most common way to optimize neural networks

**Gradient descent variants**
There are three variants of gradient descent, which differ in how much data we use to compute the gradient of the objective function. Depending on the amount of data, we make a trade-off between the accuracy of the parameter update and the time it takes to perform an update.

* **Batch gradient descent**
* **Stochastic gradient descent**
* **Mini-batch gradient descent** (Most popular)

## Sprint 2
### [Kaggle - Deeplearninig]()

**[Tensorflow programming](https://www.kaggle.com/dansbecker/tensorflow-programming)**

Basic intro to Tensorflow and quick hands on code to build Dog breed identification system, based on Resnet50.

```
import os
from os.path import join


hot_dog_image_dir = '../input/hot-dog-not-hot-dog/seefood/train/hot_dog'

hot_dog_paths = [join(hot_dog_image_dir,filename) for filename in 
                            ['1000288.jpg',
                             '127117.jpg']]

not_hot_dog_image_dir = '../input/hot-dog-not-hot-dog/seefood/train/not_hot_dog'
not_hot_dog_paths = [join(not_hot_dog_image_dir, filename) for filename in
                            ['823536.jpg',
                             '99890.jpg']]

img_paths = hot_dog_paths + not_hot_dog_paths
```


```
from IPython.display import Image, display
from learntools.deep_learning.decode_predictions import decode_predictions
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array


image_size = 224

def read_and_prep_images(img_paths, img_height=image_size, img_width=image_size):
    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]
    img_array = np.array([img_to_array(img) for img in imgs])
    output = preprocess_input(img_array)
    return(output)


my_model = ResNet50(weights='../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels.h5')
test_data = read_and_prep_images(img_paths)
preds = my_model.predict(test_data)

most_likely_labels = decode_predictions(preds, top=3)
```

Vizualize predictions:
```
for i, img_path in enumerate(img_paths):
    display(Image(img_path))
    print(most_likely_labels[i])
```


### FAST.AI

Practical Deep Learning for Coders, v3
**[Lesson 2: Data cleaning and production; SGD from scratch](https://course.fast.ai/videos/?lesson=2)**

Some interersting examples, where sound is turned into images, and DL is used with some pretrained models (Transfer learninig) to tackle new problems. What solutions could you think of?

**Learninig rate** choosing, based on graph. According to Jeremy best practice is to choose the steep long drop, where it's learninig the most and somewhere in the middle of the drop, use that learninig rate.
![lr](lr.png)

Built simple model distinguishing Teddy bear vs Black bear vs Grizly. (2-4% error).

Possiblities to improve model performance:
* Clean data better, get better images

**In production, you are going to use CPU!** Because you need to make only one prediction at a time. 

**Things that can go wrong**
* Most of the time things will train fine with the defaults
* There's not much you really need to tune (despite what you've heard!)
* Most likely are
* Learning rate 
    * Too high
    * Too low (You never want to have traininig loss higher than validation loss, so lr too low or epochs to low)
* Number of epochs
    * too few epochs
    * too many epochs, overfitting (Will learn your teddy bears to well rather than general understanding of teddybears). You will see error rating falling, but then climbing back again.

* Train_loss > valid_loss is totally fine and not an overfiting criterion, as long as error rate is falling down.

When model is making prediction, he makes predictions for all possible classes and their probabilities. And then, it uses Argmax function, returns the index(label) of something which had the highest probability. In our case either 'Teddy bear', Black Bear, or Grizzly bear.
![argmax](argmax.png)

Metrics included in model, are always going to be applied onto **Validation set.**

Most of the time you usually need less data than you think. 


**What do you do when data is unclasified?**

* Try with unbalanced data, usually it works, despite unlabalanced data (Probably it's not like in machine learning)

**When we upload/use restnet model as based, do we load it, store it's size?**

* No, we just storer the algorithm, function, thus we are not storing the restnet model.

## **What is SGD?** 
* Stochastic Gradient Descent

Tensor in DL/ML, means an array.
```
[1,
 2,
 3]

 tensor(3,2) - in pytorch, creates tensor.
```

E.g. image is a 3d tersor (Height x Width x Channels (Red,Green,Blue))

With **Tensors we say ranks. Images is Rank 3 Tensor.**


```a = tensor(-1.,1)``` - Dot after 1. gives python understading that the number is float, rather than integer.

SGD in general, taking guess and trying to make loss better - optimize it. It reduces weights, increases weights, test bigger or smaller slopes and calulates loss.


in python ```a.sub_(lr * a.grad)``` _after a variable means replace a value, ineased of returning a new one.

**Graph illustrates the imprtance of learning rate.** Its important to set one, it's also important to make sure it's not too large, so it doesnt jump too far and mises your optimal point. Hence slightly lower may be a good option. (Too small may cause, slow training)
![lr2](lr2.png)
Ideally you want somthing that is nor too small, nor too big, so it finds best point asap.


**Mini-batches** - when we have a lot of data, like million points, calculating gradient on the whole dataset makes no sense. 

But what makes sense is to use a **Mini-batches**, like 64 images at a time, at random, and calulate loss, and update the weights. and this apporach is called SGD **(Stochastic gradien descent)**

**Learninig rate** - is the thing we multiply gradient by, to decide how much update weights by.

**Epoch** - is one complete run over all our images. (But if we use minibatch, and have 1000 datapoint, use 100 as mini-batch, it would take 10 epochs to go through all points) 

**Too many epochs**, you start going over the same images again, and thus model may remember them and can begin overfitting. Generally you don't want to do too many epochs.

**Mini-bacth** - just random bunch of points you use to update your weights.

**SGD(Stochastic gradient descent)** - just gradient descent using mini-batches.

**Model/Architecture** - Architecture is a mathematical function.

**Parameters** - also known as weights, also know as coeficients, the numbers that you are updating.

**Loss fucntion** - is the thing that tells you how far or close you are to the correct answer.

![uoj](uoj.jpg)

### Deeplearning.ai

**[Neural Networks Basics](https://www.coursera.org/learn/neural-networks-deep-learning)**

**Binary Classification**
Where you get image and predict 1(cat) vs 0 (non-cat)

Image is repressented as 3D Tensor(3D matrix). Where it has Height, Width, and 3 channels for (Red, Green, Blue) colors.

We then put this into 3D tensor into 1 long array.
![bin](bin.png)


#gradient descent