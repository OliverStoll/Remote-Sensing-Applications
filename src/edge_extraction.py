#!/usr/bin/env python
# coding: utf-8

# ## Import Packages

# In[35]:


# used for submission variables
from copy import deepcopy


# In[36]:


# importing libraries
# you can import any that are installed in Docker/Paperspace environment
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


# ## Manually applying filters
# 
# In the theoretical lecture, you have seen a way to visualize and _manually_ calculate the convolution results.
# Please apply this _manual_ convolution method to the following input array.
# For this task, you can only use basic numpy operators (`+ - / @`) and for loops.
# 
# Apply the manual convolution method using:
# - 0-padding
# - Reflection
#     - `a b c d` as input leads to the following reflections on the left and right-hand sides:
#         - `d c b a | a b c d | d c b a`

# In[37]:


inp_arr = np.array(
    [
        [1, 0, 1],
        [0, 1, 0],
        [0, 0, 0],
    ]
)

kernel = np.array(
    [
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]
)


# In[38]:


# Add your code
input_size = inp_arr.shape[0]
kernel_size = kernel.shape[0]

padding_size = int((kernel_size - 1) / 2)
total_size = input_size + 2 * padding_size

input_0 = np.zeros((total_size, total_size))
input_refl = np.zeros((total_size, total_size))

for row in range(input_size):
    for column in range(input_size):
        input_0[row + padding_size][column + padding_size] = inp_arr[row][column]
        input_refl[row + padding_size][column + padding_size] = inp_arr[row][column]

for row in range(total_size):
    for column in range(total_size):

        # skip coordinate pairs inside the content
        inner_indexes = list(range(padding_size, input_size + padding_size))
        if row in inner_indexes and column in inner_indexes:
            continue

        # calculate reflection indexes for row and column
        coordinate_values = [row, column]
        new_coordinates = []
        for index in coordinate_values:

            start_edge_index = padding_size  # first index inside content
            end_edge_index = input_size + padding_size - 1  # last index inside content

            if index < start_edge_index:
                distance_to_edge = (start_edge_index - 1) - index
                index = start_edge_index + distance_to_edge

            elif index >= input_size + padding_size:
                distance_to_edge = index - (end_edge_index + 1)
                index = end_edge_index - distance_to_edge

            else:
                pass  # this coordinate doesnt get mirrored

            # now subtract the padding from index and save it back
            index = index - padding_size
            new_coordinates.append(index)


        # now set the
        new_row, new_col = new_coordinates[0], new_coordinates[1]
        input_refl[row][column] = inp_arr[new_row][new_col]


# In[39]:


# apply convolution
def convolve(input_matrix, kernel):
    output_matrix = np.zeros((input_size, input_size))
    for row in range(input_size):
        for column in range(input_size):
            mini_matrix = input_matrix[row:row + kernel_size, column:column + kernel_size]
            single_convolve = mini_matrix * kernel
            single_convolve_sum = np.sum(single_convolve)
            output_matrix[row][column] = single_convolve_sum
    return output_matrix


# print(kernel.sum(), kernel, "\n")
print(input_0)
print(input_refl)

kernel_flipped = np.flip(kernel, axis=(0,1))
output_zero = convolve(input_matrix=input_0, kernel=kernel_flipped)
print(output_zero, "\n")

output_refl = convolve(input_matrix=input_refl, kernel=kernel_flipped)
print(output_refl)

output_0_pad = output_zero / kernel.sum()
output_ref_pad = output_refl / kernel.sum()


# In[40]:


# DO NOT EDIT
submission_output_arr_0_pad = deepcopy(output_0_pad)
submission_output_reflect = deepcopy(output_ref_pad)
###


# Visualize the outputs of the convolution operation and plot the results _next_ to each other in a single row.

# In[41]:


# Add code to visualize the results next to each other
# Create two subplots
from matplotlib import pyplot as plt

fig, ax = plt.subplots(1, 2)
ax[0].imshow(input_0, cmap="gray")
ax[1].imshow(input_refl, cmap="gray")



# ## Applying filters to satellite data
# 
# After manually applying the convolution operation to the input, you can now use the functions provided by `scipy` for the remaining tasks.
# 
# 1. Download the image from the following link:
#     - https://tubcloud.tu-berlin.de/s/mPLKMiHdSbbnNcH/download/noisy_river.npz
#     - Note: The image data has already been converted to the [0, 1] range, so there is no need to apply any _preprocessing/normalization_ to the data!
# 1. Extract and visualize the channel that is named `blue`
# 1. Apply filters that are designed for _denoising_ the input image
#     - Select a single filter for generating the denoised output
#     - Explain why you have selected the filter and why you have chosen the specific kernel size
#     - Visualize the denoised image
# 2. Apply filters that are not designed for denoising with varying sizes and padding strategies to the _denoised_ image
#     - What is the effect of the padding strategy on the output?
#     - What do you notice when changing the padding strategy?
# 

# In[42]:


# Download the noisy image from within the notebook
from simple_downloader import download
TUB_URL = "https://tubcloud.tu-berlin.de/s/mPLKMiHdSbbnNcH/download/noisy_river.npz"
output_file = download(TUB_URL, "./data/ha2-data")
npzfiles = np.load(output_file)


# And visualize the blue band
noisy_blue_band = npzfiles['blue']

plt.imshow(noisy_blue_band, cmap="gray")


# In[43]:


# DO NOT EDIT
submission_noisy_img = deepcopy(noisy_blue_band)
###


# In[44]:


# Denoise the image with a _single_ filter and visualize the result
print(noisy_blue_band.shape)
smaller_img_size = 300

kernel_size = int(smaller_img_size / 50)
avg_kernel = np.ones((kernel_size, kernel_size))
denoised_img = ndimage.convolve(noisy_blue_band, avg_kernel) / avg_kernel.sum()

plt.imshow(denoised_img, vmin=0, vmax=1, cmap="gray")


# Add a short discussion about _why_ you selected the filter and _why_ the chosen kernel size was selected, not a different one.
# 
# # Discussion
# 
# I chose an averaging filter as it seems like a straightforward way to reduce noise and a kernel size of 2% of the smaller image dimension as it seemed to result in a denoising effect that does not loose too much information about the main features.

# In[45]:


# DO NOT EDIT
submission_denoised_img = deepcopy(denoised_img)
###


# In[46]:


# apply different filter operations with various sizes and
# padding strategies to the denoised image
# try to visualize a couple of outputs that support your
# arguments in the padding-strategy discussion!

# bigger kernel size with averaging
kernel_factor = 12
kernel_size = int(smaller_img_size / kernel_factor)
avg_kernel = np.ones((kernel_size, kernel_size))


# In[47]:


denoised_img = ndimage.convolve(noisy_blue_band, avg_kernel, mode="constant", cval=0) / avg_kernel.sum()
plt.imshow(denoised_img, cmap="gray")


# In[48]:


denoised_img = ndimage.convolve(noisy_blue_band, avg_kernel, mode="reflect") / avg_kernel.sum()
plt.imshow(denoised_img, cmap="gray")


# Add a short discussion about the padding strategy here.
# 
# # Discussion:
# 
# After experimenting with multiple kernel sizes and both the reflection and the zero-padding strategy (implemented with ndimage and mode constant), I found that with a greater kernel size the difference between the sharpness of both modes became more and more obvious.
# Here I find that while both images lost information on the river edges during the denoising, the reflection padding strategy produced better results while the zero-padding strategy disproportionally more information and produces a result with less sharp edges.

# ### Creating features
# 
# Create an output feature map that highlights the river edges.
# - You can apply multiple filter operations after each other to generate the output feature map
# - You can arbitrarily compose the intermediate results
#     - For example, by adding feature maps of filter responses together
# - The final feature map should be normalized to the range between 0 -- 1
# - Your strategy _must_ include _custom_ kernels that were not presented in the theoretical lecture
# 
# Visually compare your output to the following image:
# 
# ![](river_edges.png)
# 
# > If the image isn't displayed, make sure that the image file is located next to this notebook! Otherwise, manually inspect the provided image file: `river_edges.png`
# 
# What is more/less prominent in your image?
# Try to provide an intuitive explanation for why your filter composition choice generates the given output.
# Motivate your choices and clarify that you haven't applied _random_ kernels to the output!

# 
# # Discussion:
# 
# I choose a stronger version of the diagonal sobel filter discussed in the lecture.
# By this, I hoped to enhance the mostly diagonal edges in the image even more in comparison to the existing noise of the blue band.
# After applying two mirrored versions of this modified sobel filter, I averaged the image and took the absolute value to accentuate both strongly positive and negative values from the convolutional operations.
# 
# In my result, the edges of the river bank are strongly recognisable with a high contrast to the rest of the image. On the other hand, they are widened due to the average filtering I applied as the first step to reduce noise. Even after applying this moderate average filtering, there is still more noise than in the comparison image.

# In[49]:


# TODO: create complex filter output
def convolve(input, kernel):
    return ndimage.convolve(input, kernel, mode="reflect")

avg_size = 5
kernel_avg = np.ones((avg_size,avg_size))
kernel_sobel = np.array(
    [[3, 1, 0],
    [1, 0, -1],
    [0, -1, -3],]
)
kernel_sobel_2 = np.fliplr(kernel_sobel)
img = noisy_blue_band
img = convolve(img, kernel_avg) / kernel_avg.sum()


img_leftup = convolve(img, kernel_sobel)
img_rightup = convolve(img, kernel_sobel_2)
img_total = img_leftup + img_rightup

img = (img_total / np.mean(img_total))
img = np.abs(img)

lq, uq = np.quantile(img, (0.01, 0.99))
img = np.clip(img, a_min=lq, a_max=uq)
img = img / img.max()


print(img.shape[0])

plt.imshow(img, cmap="gray", vmin=0, vmax=1)
plt.axis('off')


river_border_features = img


# In[50]:


# DO NOT EDIT
submission_river_border_features = deepcopy(river_border_features)
###


# ## Submission
# 
# - Follow the instructions mentioned in this notebook
# - You are _not_ allowed to install other libraries
#     - The submitted jupyter notebook should be able to run as-is on Paperspace and/or the provided Docker image.
# - Explain the steps that went into the decision making of the specific values
# - Each homework must be done individually!
# - The notebook should have been executed once before submitting it
#     - The notebook should _not_ be _cleaned_ before submitting
# 
# Each student should submit the following materials to the ISIS system:
# - One `zip` file containing the jupyter notebook named as: `IP4RS2022-HW2_<NAME_SURNAME>.zip`
# 
# 
# 
# ### Submission variables
# 
# Please ensure that these variables are correct after you have restarted your notebook and executed all cells from top to bottom.
# You are _not allowed_ to modify the following cells:

# In[51]:


import rich

rich.print(f"{submission_output_arr_0_pad=}")
rich.print(f"{submission_output_reflect=}")
rich.print(f"{submission_noisy_img}")
rich.print(f"{submission_denoised_img=}")
rich.print(f"{submission_river_border_features}")


# In[51]:




