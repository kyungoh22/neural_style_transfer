import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

def load_resize(file_path, img_size):
    """
    Loads image, resizes, and saves as numpy array. Then converts into Tensor. 

    Arguments
    file_path
    img_size -- integer

    Returns
    Image as Tensor of shape (1, img_size, img_size, 3)
    """
    # Load image, resize, and convert into numpy array
    image = np.array(Image.open(file_path).resize((img_size, img_size)))        # (img_size, img_size, 3)

    # Reshape
    image = np.reshape(image, ((1,) + image.shape))                             # (1, img_size, img_size, 3)

    # Convert to Tensor
    image = tf.constant(image)

    #imshow(image[0])
    #plt.show()

    return image


def content_cost (content, generated, content_model):
    """
    Compute a_C -- hidden layer activations representing the content of the content image.
    Compute a_G -- hidden layer activations representing the content of the generated image.
    Compute the content cost by comparing a_G with a_C. 

    Arguments:
    content -- (1, n_H, n_W, n_C)
    generated -- (1, n_H, n_W, n_C)
    content_model -- Model that outputs the activations of a VGG19 hidden layer
    
    Returns:
    content_Cost -- scalar
    """
    
    a_C = content_model(content)[0]                     # a_C -- (n_H, n_W, n_C)
    a_G = content_model (generated)[0]                  # a_G -- (n_H, n_W, n_C)

    # Retrieve dimensions
    n_H, n_W, n_C = a_C.shape                    
    
    # Compute the content cost -- based on the square of L2 norm of (a_C - a_G)
    content_cost = (1/(4*n_H*n_W*n_C))* tf.reduce_sum(tf.square(tf.subtract(a_C, a_G)))
    return content_cost


def gram_matrix (A):

    """
    Computes the Gram matrix of A.
    In our case, A will be the "unrolled" filter matrix with the dimensions (n_C, n_H * n_W).

    Arguments:
    A -- matrix of shape (n_C, n_H * n_W)
    
    Returns:
    gram -- Gram matrix of A with shape (n_C, n_C)
    """
    gram = tf.linalg.matmul(A, tf.transpose(A))
    return gram

def style_cost(style_image, generated_image, style_models, style_weights = [0.2, 0.2, 0.2, 0.2, 0.2]):

    """
    Compute a_S -- hidden layer activations representing the style of the style image.
    Compute a_G -- hidden layer activations representing the style of the generated image.
    Compute the style cost by comparing a_S with a_G.

    Arguments:
    style_image -- preprocessed_image, 4D tensor (1, n_H, n_W, n_C), with values from 0 to 1
    generated_image -- ditto
    style_models -- list of models that each outputs the activations of a VGG19 hidden layer
    style_weights -- list of weights for the five VGG19 hidden layers 

    Returns:
    J_style_total -- total style cost, scalar
    """

    J_style_total = 0

    # For each of the five VGG hidden layers
    for i, style_model in enumerate(style_models):
        
        # Retrieve hidden layer activations for style image and generated image
        a_S = style_model(style_image)[0]       # a_S = (n_H, n_W, n_C)
        a_G = style_model(generated_image)[0]   # a_G = (n_H, n_W, n_C)

        # Retrieve dimensions
        n_H, n_W, n_C = a_S.shape
        
        # Reshape the images from (n_H * n_W, n_C) to have them of shape (n_C, n_H * n_W)
        a_S = tf.reshape(a_S, shape = [n_H * n_W, n_C])
        a_S = tf.transpose(a_S, perm = [1,0])   # a_S = (n_C, n_H * n_W)

        a_G = tf.reshape(a_G, shape = [n_H * n_W, n_C])
        a_G = tf.transpose(a_G, perm = [1,0])   # a_G = (n_C, n_H * n_W)

        # Compute gram_matrices for both images S and G 
        GS = gram_matrix(a_S)
        GG = gram_matrix(a_G)

        # Compute style cost for current layer
        J_style_layer = (1/(4*n_C*n_C*(n_H*n_W)**2)) * tf.reduce_sum(tf.multiply(tf.subtract(GS, GG), tf.subtract(GS, GG)))

        # Update total

        J_style_total += style_weights[i]*J_style_layer

    return J_style_total


def clip_0_1(image):
    """
    Truncate all the pixels in the tensor to be between 0 and 1
    During optimisation, we want to make sure the updated pixel values for the image stay within the range 0 to 1.
    """
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

def tensor_to_image(tensor):
    """
    Scales the tensor values (input values are between 0 and 1) back to range of 0 to 255.
    Also convert the values into integers.
    If tensor is 4D, then take only last 3 values.
    Convert this to PIL image.
    
    Arguments:
    tensor -- Tensor
    
    Returns:
    Image: A PIL image
    """
    tensor = tensor * 255                       # Convert values to range 0–255
    tensor = np.array(tensor, dtype=np.uint8)   # Convert into integer type
    if np.ndim(tensor) > 3:                     # If more than 3 dimensions, take only last 3 dimensions.
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)



@tf.function()
def train_step(content, style, generated, content_model, style_models, alpha, beta, optimizer):
    """
    One training step. Does the following:
    - Compute total loss J_total
    - Compute gradient of J_total wrt generated image 
    - Update the generated image based on the gradient

    Arguments:
    content -- content image
    style -- style image
    generated -- generated image
    content_model -- Model that outputs the activations of a VGG19 hidden layer
    style_models -- list of Models that each outputs the activations of a VGG19 hidden layer
    alpha -- coefficient for J_cost
    beta -- coefficient for J_style
    optimizer -- initialized optimizer

    Returns:
    J_total -- scalar
    """
    with tf.GradientTape() as tape:
        J_content = content_cost (content, generated, content_model)
        J_style = style_cost(style, generated, style_models, style_weights = [0.2, 0.2, 0.2, 0.2, 0.2])
        J_total = alpha * J_content + beta * J_style

    grads = tape.gradient(J_total, generated)
    optimizer.apply_gradients([(grads, generated)])
    generated.assign(clip_0_1(generated))

    return J_total



def plot_gen_image(content_image, style_image, generated_image, 
                    epochs, alpha, beta, learning_rate, output_file_name,
                    caption_1, caption_2):
    """
    Plots the original content image, original style image, and final generated image side by side.
    """
    fig, ax = plt.subplots(1, 3, figsize = (12, 6), dpi = 300)
    ax[0].imshow(content_image)
    ax[0].set_title('Content image')
    ax[0].set_xlabel(f'\n \n{caption_1}')
    ax[1].imshow(style_image)
    ax[1].set_title('Style image')
    ax[1].set_xlabel(f'\n \n{caption_2}')
    ax[2].imshow(generated_image)
    ax[2].set_title('Generated image')
    ax[2].set_xlabel(f'epochs: {epochs} \n alpha: {alpha} \n beta: {beta} \n learning_rate: {learning_rate}')
    plt.tight_layout()
    plt.savefig(output_file_name)
    plt.show()
