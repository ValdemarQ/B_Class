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


**### Power of Deep learninig**
Top researchers of Oxford in 2012 built best experimental models that predicted cat/dog breed with around 60% accuracy.

Today, with few lines of DL code, using transfer learninig, and 1-2 minutes of traininig, we can achieve 94% of accuracy.

**Keras** is alternative to FASTAI, a good library too.

**Loss function** - tells how good is your prediction. 

.most_confused(min_val=2) - grabs from confussion matrix which were mistaken and brings you back a list, based on your min value. Easier than confusion matrix to distinguish which were wrong.


```.unfreeze()``` - a Thing that says please train the whole model.

**### Visualizing and understanding CNN**

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