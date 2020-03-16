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
**[Lesson 1: Image classification]()**