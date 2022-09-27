## Art Generation -- Neural Style Transfer

- The goal is to generate an image (**G**) by merging a "content" image (**C**) with a "style" image (**S**). The generated image will contain the content of C in the style of S. 

- We will use a previously trained convolutional network to capture, or encode, the content of C and the style of S. In our case, we will use the VGG19 model trained on the ImageNet database. 

### VGG19 architecture overview
- The VGG19 model consists of five convolutional blocks followed by three fully connected layers. The model contains 19 layers in total. For the full details see the original paper [here] (https://arxiv.org/abs/1508.06576).

### Optimization objective: Defining the cost function

We essentially want to minimise the "difference" between the content of G and the content of C, as well as the difference between the style of G and the style of S. 

Our loss function is therefore: 

J_total = alpha * J_content + beta * J_style

where J_content is the content cost function, representing the "difference" between the content of G and the content of C, and J_style is the style cost function, representing the "difference" between the style of G and the style of C. 

The coefficients "alpha" and "beta" can be adapted depending on whether we prioritise G's similarity with the content of C or the style of S. 


#### Content cost function (J_content)

- We choose a hidden layer of the VGG19 model with which to encode the content of C and G. 
- For this project, I chose the layer 'block5_conv4'
- Forward propagate image "C" and let **a_C** be the activations of the chosen hidden layer. The tensor a_C has the dimensions (n_H, n_W, n_C) and represents the "content" of image "C". 
- Forward propagate image "G" and let **a_G** be the activations of the chosen hidden layer. The tensor a_G has the dimensions (n_H, n_W, n_C) and represents the "content" of image "G".
- Compute J_content based on the squared norm of (a_C - a_G). 

#### Style cost function (J_style)

- Before we define the style cost function, we first need to discuss what we mean by the "style" of an image. 

##### The "style" matrix (Gram matrix)


- We choose several hidden layers of the VGG19 model with which to encode the style of C and G.
- For this project, I chose the layers 'block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1'.
- 
