## Art Generation -- Neural Style Transfer

The goal is to generate an image by merging a "content" image (image C) with a "style" image (image S). 
The generated image will contain the content of image C in the style of image S. 

We will use a previously trained convolutional network to capture, or encode, the content of "image C" and the style of "image S". 
In our case, we will use the VGG19 model trained on imagenet. 

