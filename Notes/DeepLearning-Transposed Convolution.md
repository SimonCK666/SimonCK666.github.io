---
layout: page
title: "DL-Transposed Convolution"
---

- [Exponential Moving Average (EMA)](#ema)  

{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Up-sampling with Transposed Convolution\n",
    "\n",
    "When we use neural networks to generate images, it usually involves up-sampling from low resolution to high resolution.\n",
    "\n",
    "There are various methods to conduct up-sample operation:\n",
    "\n",
    "* Nearest neighbor interpolation\n",
    "* Bi-linear interpolation\n",
    "* Bi-cubic interpolation\n",
    "\n",
    "All these methods involve some interpolation which we need to chose like a manual feature engineering that the network can not change later on.\n",
    "\n",
    "Instead, we could use the transposed convolution which has learnable parameters [[1]](#ref1).\n",
    "\n",
    "Examples of the transposed convolution usage:\n",
    "\n",
    "* the generator in DCGAN takes randomly sampled values to produce a full-size image [[2]](#ref2).\n",
    "* the semantic segmentation uses convolutional layers to extract features in the encoder and then restores the original image size in the encoder so that it can classify every pixel in the original image [[3]](#ref3).\n",
    "\n",
    "The transposed convolution is also known as:\n",
    "\n",
    "* Fractionally-strided convolution\n",
    "* Deconvolution\n",
    "\n",
    "But we will only use the word **transposed convolution** in this notebook.\n",
    "\n",
    "One caution: the transposed convolution is the cause of the checkerboard artifacts in generated images [[4]](#ref4). The paper recommends an up-sampling followed by convolution to reduce such issues.  If the main objective is to generate images without such artifacts, it is worth considering one of the interpolation methods."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Using TensorFlow backend.\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import keras\n",
    "import keras.backend as K\n",
    "from keras.layers import Conv2D\n",
    "from keras.models import Sequential\n",
    "\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Convolution Operation\n",
    "\n",
    "### Input Matrix\n",
    "\n",
    "We define a 4x4 matrix as the input.  We randomly generate values for this matrix using 1-5."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[3, 3, 3, 6],\n",
       "       [7, 4, 3, 2],\n",
       "       [7, 2, 8, 5],\n",
       "       [1, 7, 3, 1]])"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "inputs = np.random.randint(1, 9, size=(4, 4))\n",
    "inputs"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The matrix is visualized as below.  The higher the intensity the bright the cell color is."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "def show_matrix(m, color, cmap, title=None):\n",
    "    rows, cols = len(m), len(m[0])\n",
    "    fig, ax = plt.subplots(figsize=(cols, rows))\n",
    "    ax.set_yticks(list(range(rows)))\n",
    "    ax.set_xticks(list(range(cols)))\n",
    "    ax.xaxis.tick_top()  \n",
    "    if title is not None:\n",
    "        ax.set_title('{} {}'.format(title, m.shape), y=-0.5/rows)\n",
    "    plt.imshow(m, cmap=cmap, vmin=0, vmax=1)\n",
    "    for r in range(rows):\n",
    "        for c in range(cols):\n",
    "            text = '{:>3}'.format(int(m[r][c]))\n",
    "            ax.text(c-0.2, r+0.15, text, color=color, fontsize=15)\n",
    "    plt.show()\n",
    "    \n",
    "def show_inputs(m, title='Inputs'):\n",
    "    show_matrix(m, 'b', plt.cm.Vega10, title)\n",
    "    \n",
    "def show_kernel(m, title='Kernel'):\n",
    "    show_matrix(m, 'r', plt.cm.RdBu_r, title)\n",
    "    \n",
    "def show_output(m, title='Output'):\n",
    "    show_matrix(m, 'g', plt.cm.GnBu, title)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f8c0d73fcf8>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "show_inputs(inputs)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We are using small values so that the display look simpler than with big values.  If we use 0-255 just like an gray scale image, it'd look like below."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\nYEWxB94AAAAASUVORK5CYII=\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f8c0a38d240>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "show_inputs(np.random.randint(100, 255, size=(4, 4)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Apply a convolution operation on these values can produce big values that are hard to nicely display.\n",
    "\n",
    "Also, we are ignoring the channel dimension usually used in image processing for a simplicity reason."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Kernel\n",
    "\n",
    "We use a 3x3 kernel (filter) in this example (again no channel dimension).\n",
    "\n",
    "We only use 1-5 to make it easy to display the calculation results."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1, 2, 4],\n",
       "       [1, 1, 3],\n",
       "       [1, 2, 4]])"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "kernel = np.random.randint(1, 5, size=(3, 3))\n",
    "kernel"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\nRU5ErkJggg==\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f8c0a280358>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "show_kernel(kernel)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Convolution\n",
    "\n",
    "With padding = 0 (padding='VALID') and strides = 1, the convolution produces a 2x2 matrix.\n",
    "\n",
    "--- \n",
    "\n",
    "$H_m, W_m$: height and width of the input\n",
    "\n",
    "$H_k, W_k$: height and width of the kernel\n",
    "\n",
    "$P$: padding\n",
    "\n",
    "$S$: strides\n",
    "\n",
    "$H, W$: height and width of the output\n",
    "\n",
    "$W = \\frac{W_m - W_k + 2P}{S} + 1$\n",
    "\n",
    "$H = \\frac{H_m - H_k + 2P}{S} + 1$\n",
    "\n",
    "---\n",
    "\n",
    "With the 4x4 matrix and 3x3 kernel with no zero padding and stride of 1:\n",
    "\n",
    "$\\frac{4 - 3 + 2\\cdot 0}{1} + 1 = 2$\n",
    "\n",
    "So, with no zero padding and strides of 1, the convolution operation can be defined in a function like below:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def convolve(m, k):\n",
    "    m_rows, m_cols = len(m), len(m[0]) # matrix rows, cols\n",
    "    k_rows, k_cols = len(k), len(k[0]) # kernel rows, cols\n",
    "\n",
    "    rows = m_rows - k_rows + 1 # result matrix rows\n",
    "    cols = m_rows - k_rows + 1 # result matrix cols\n",
    "    \n",
    "    v = np.zeros((rows, cols), dtype=m.dtype) # result matrix\n",
    "    \n",
    "    for r in range(rows):\n",
    "        for c in range(cols):\n",
    "            v[r][c] = np.sum(m[r:r+k_rows, c:c+k_cols] * k) # sum of the element-wise multiplication\n",
    "    return v"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The result of the convolution operation is as follows:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[84, 84],\n",
       "       [87, 60]])"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "output = convolve(inputs, kernel)\n",
    "output"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\ny8zELo7wMQAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f8c0a21d940>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "show_output(output)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "One important point of such convolution operation is that it keeps the positional connectivity between the input values and the output values.\n",
    "\n",
    "For example, `output[0][0]` is calculated from `inputs[0:3, 0:3]`.  The kernel is used to link between the two."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "84"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "output[0][0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[3, 3, 3],\n",
       "       [7, 4, 3],\n",
       "       [7, 2, 8]])"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "inputs[0:3, 0:3]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1, 2, 4],\n",
       "       [1, 1, 3],\n",
       "       [1, 2, 4]])"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "kernel"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "84"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.sum(inputs[0:3, 0:3] * kernel) # sum of the element-wise multiplication"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "So, 9 values in the input matrix is used to produce 1 value in the output matrix.\n",
    "\n",
    "### Going Backward\n",
    "\n",
    "Now, suppose we want to go the other direction.  We want to associate 1 value in a matrix to 9 values to another matrix while keeping the same positional association.\n",
    "\n",
    "For example, the value in the left top corner of the input is associated with the 3x3 values in the left top corner of the output.  \n",
    "\n",
    "This is the core idea of the transposed convolution which we can use to up-sample a small image into a larger one while making sure the positional association (connectivity) is maintained.\n",
    "\n",
    "Let's first define the convolution matrix and then talk about the transposed convolution matrix."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Convolution Matrix\n",
    "\n",
    "We can express a convolution operation using a matrix.  It is nothing but a kernel matrix rearranged so that we can use a matrix multiplication to conduct convolution operations."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "def convolution_matrix(m, k):\n",
    "    m_rows, m_cols = len(m), len(m[0]) # matrix rows, cols\n",
    "    k_rows, k_cols = len(k), len(k[0]) # kernel rows, cols\n",
    "\n",
    "    # output matrix rows and cols\n",
    "    rows = m_rows - k_rows + 1 \n",
    "    cols = m_rows - k_rows + 1\n",
    "    \n",
    "    # convolution matrix\n",
    "    v = np.zeros((rows*cols, m_rows, m_cols)) \n",
    "\n",
    "    for r in range(rows):\n",
    "        for c in range(cols):\n",
    "            i = r * cols + c\n",
    "            v[i][r:r+k_rows, c:c+k_cols] = k\n",
    "\n",
    "    v = v.reshape((rows*cols), -1)\n",
    "    return v, rows, cols"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "C, rows, cols = convolution_matrix(inputs, kernel)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\nAj+kfjrb/wBuLTaqpvlSRCwGhoF3lXnSo2ONy8B/Br4QEb8N/Bx4c3ERTs5x8toF/DWwBPh6RGxJ\nKa0vLsrJOU5uvwe0A3c1Pv//OaX0zsKCnITj5HVFY2fgGPV1sVQ5HZLr9u9x3rNXRsS51E87f4xZ\nMq5FnkfPJUmSJEkTUdbTRyVJkiRJTWBTKEmSJEkVZlMoSZIkSRVmUyhJkiRJFWZTKEmSJEkVVvqm\nMCKuKzqG6WBe5ZNrbrnmBfnmZl7lk2tu5lU+ueZmXuWTa26zNa/SN4XArPzDNoF5lU+uueWaF+Sb\nm3mVT665mVf55JqbeZVPrrnNyrxyaAolSZIkSZM0LV9e3xEtaT6tTV/usRxklA5aZuS1ZpJ5Nce/\ndM6fsddKIweJ1o4Zea3FB/pn5HUg33UR8s0t1zqbyRoD66wZzKs5ch3LZpJ5lU+uuc1kXmloH2nk\nYEzkudPSuc2nlatZMR2Llk7KbWsuKTqEaXH1lk1FhyAdZp1J0yvXGpM0vUa23Tnh53r6qCRJkiRV\nmE2hJEmSJFWYTaEkSZIkVZhNoSRJkiRVmE2hJEmSJFWYTaEkSZIkVZhNoSRJkiRVmE2hJEmSJFWY\nTaEkSZIkVZhNoSRJkiRVmE2hJEmSJFWYTaEkSZIkVZhNoSRJkiRVmE2hJEmSJFWYTaEkSZIkVZhN\noSRJkiRVmE2hJEmSJFWYTaEkSZIkVZhNoSRJkiRVmE2hJEmSJFWYTaEkSZIkVdiEmsKIuDwitkXE\nTyLig9MdlCRJkiRpZrSe6AkR0QJsAC4DngR+EBF3ppQenO7gJuJi+jmLA6xmkHmMcQvLeISOosOa\nkm5GuJQ+1nKQHkYYoMY2OvgKC9l74resVN7ELl5FP3exgC+xqOhwpiYlPtB7H9ftfJiekUE2d/Xw\n3pUXsrVrcdGRTUmONXZIjrm1kLiWnaxiiG5GGST4OXO4k4U8TnvR4U1NpjUGea6LVRnLshrHIN86\nyzUvyDc385pREzlS+FLgJymln6WUhoC/A14/vWFN3IXsYy6jPFjywXO8VQxxDgf4PnPZwFK+zCJW\nM8gNbKedsaLDa5oVDPFy9nGAKDqUprih935u3L6Vjyx9CVedcSn7aq1886cbWTa8v+jQpiTHGjsk\nx9xqJBKwkQVsYAmfZTFzSLyHHfQwXHR4U5JrjUGe62IVxrLcxjHIt85yzQvyzc28ZtZEmsKVwBPj\nbj/ZuG9WuJnl3MQKvkV30aE0zU/o4EOcwka6eYQONjOXj7KUxYxyHuUuhPHewm42sYD9GVza2j42\nwvt33MdNy87mY0vWsmn+KVxz+joS8Ds7Hyo6vCnJscYOyTG3YWp8nCV8j/lso5OtdLGBpbSSOIcD\nRYc3aTnXGOS5LlZhLMtpHIN86yzXvCDf3Mxr5jXtUywirouIzRGx+SCjzVrsCaWM9s4dcoAaY0fl\n1UsbgwTdM/i3nU7nM8ByhtnIgqJDaYqLBnrpHhvmiwtPP3zf/pY2vt69ivV9TxUXWBPkWGOH5Jzb\neIMEIwStpKJDmbScawzyXBdzH8tyG8cg3zrLNS/INzfzmnkTaQqfAk4bd/vUxn1HSCndmlK6IKV0\nQQctzYpPDSsZop1EbwbXYbQxxtXs5g4WMpTJ3tU1g3sZIfhx+5EbBw93dLNmcG9BUanaEjUSCxjl\njexhjOAHzC06qEmzxvKQy1iW4zgG+dZZrnlBvrmZ18ybyKfyD4AzI2I19WbwrcDbpjUqHSFIvJld\n7KCVrXQVHc6UXU4fe2nhnhJvoB5t0cgQ+2ptjMWRGwe7W9qZOzZC29gowzV3lmjmrKePq9gDQB81\n/oal7Crxhrg1Vn45jWU5jmOQb53lmhfkm5t5zbwT7t5KKY0A7wY2Ag8BX0gpPTDdgekX3sAezmCQ\nT9LznFNxymYxw1xGH1/gBVDyXKTZ7G7m8WGWs4ElPM4c3kUvKxgqOixVWC5jmeOYpBxN6JyHlNI3\nUkovTim9MKX0p9MdlH7hYvq5jD5up4fHyj6dPHAVe/gRneygjU7G6GxsGrSS6GQMSnrN0+7WOcwb\nG6aWjpxRb9HoIAO11lLuzVK59dHC47RzP118lKXso8Z6+ooOa9KssXLLaSzLdRyDfOss17wg39zM\na+aV91yiCjiPAd7CLu5gIfdmcorKMoY5jWHOP2rmuXX0s45+PshK9pRwtdzW3k0riRcN9vNIxy9m\nD1xzcC/b2vOZTVDlNEbwNHPoYaToUCbNGiuv3MayXMcxyLfOcs0L8s3NvGZePldHZ+bFHORadvJd\n5nNXRlOUf4bF3MKyI372UmMzXdzCMvaVdJKiu+cuZW+tjav3PHr4vs6xEV7T9wQbF8yab3BRRbWS\nOI0hdpZ0QxWssbLKcSzLdRyDfOss17wg39zMa+aVdwuhYRWD9DDCosb01mdykHmMspNWHi/pKSrL\nGead9LKdNjYzl9UMHn6snxo7aSswuqk51nsyQrCbVh4p8Zc2D9ZauXnZ2dy4fQu7W9rZ1tHN9c8+\nQC0lNvSsLTq8Kcmxxg7JMbcLGOAsDvAAneylhQWMcjH9dDPKd0o8dX7ONQZ5rou5jmW5jmOQb53l\nmhfkm5t5zbzSN4Xr6OciBg7ffh316VzvZi6fKulAuppBukh0McwH2H7EY/W8egqKTM/npqUvoZYS\nN/Tex+KRQe7t6uGKF66nt62z6NCmJMcaOyTH3HbQxoUM8Kvsoosx+mjhUdr5M5bzDHOKDm9Kcq0x\nyHNddCwrp1zrLNe8IN/czGtmRUrNvyB6SbSnq1nR9OVKJ+u2cy8pOoRpce2WTUWHIB1mnUnTK9ca\nkzS9Rrbdydj+nROaJtlrCiVJkiSpwmwKJUmSJKnCbAolSZIkqcJsCiVJkiSpwmwKJUmSJKnCbAol\nSZIkqcJsCiVJkiSpwmwKJUmSJKnCbAolSZIkqcJsCiVJkiSpwmwKJUmSJKnCbAolSZIkqcJsCiVJ\nkiSpwmwKJUmSJKnCbAolSZIkqcJsCiVJkiSpwmwKJUmSJKnCIqXU9IXWunpS65orm77col27ZVPR\nIUgA3HbuJUWHMG2sM80WudaZNabZItcak2aLkW13MrZ/Z0zkuR4plCRJkqQKsymUJEmSpAqzKZQk\nSZKkCrMplCRJkqQKsymUJEmSpAqzKZQkSZKkCrMplCRJkqQKsymUJEmSpAqzKZQkSZKkCrMplCRJ\nkqQKsymUJEmSpAqzKZQkSZKkCrMplCRJkqQKsymUJEmSpAqzKZQkSZKkCrMplCRJkqQKsymUJEmS\npAqzKZQkSZKkCrMplCRJkqQKsymUJEmSpAqzKZQkSZKkCrMplCRJkqQKsymUJEmSpAprPdETIuI2\n4LVAb0rprOkP6SSlxAd67+O6nQ/TMzLI5q4e3rvyQrZ2LS46sim5mH7O4gCrGWQeY9zCMh6ho+iw\npqybES6lj7UcpIcRBqixjQ6+wkL2nnh1LI03sYtX0c9dLOBLLCo6nKnJtMYg3zrLMa8WEteyk1UM\n0c0ogwQ/Zw53spDHaS86vKnLtM5yXBerMo6BY1lp5Jqbec2oiRwpvB24fJrjmLQbeu/nxu1b+cjS\nl3DVGZeyr9bKN3+6kWXD+4sObUouZB9zGeXBkg+eR1vFEOdwgO8zlw0s5cssYjWD3MB22hkrOrym\nWMEQL2cfB4iiQ2mKXGsM8q2zHPOqkUjARhawgSV8lsXMIfEedtDDcNHhTVmudZbjuliFcQwcy8ok\n19zMa2adsClMKf0TsGsGYjlp7WMjvH/Hfdy07Gw+tmQtm+afwjWnryMBv7PzoaLDm5KbWc5NrOBb\ndBcdSlP9hA4+xClspJtH6GAzc/koS1nMKOdR7iI/5C3sZhML2J/B2dk51xjkW2c55jVMjY+zhO8x\nn210spUuNrCUVhLncKDo8KYk5zrLcV2swjgGjmVlkWtu5jXzSl3pFw300j02zBcXnn74vv0tbXy9\nexXr+54qLrAmSJnsmTvaAWqMHZVbL20MEnQzWlBUzXM+AyxnmI0sKDqUpsi5xiDfOss1r6MNEowQ\ntJKKDmVKcq6zHNfF3McxcCwrk1xzM6+Z17SmMCKui4jNEbE5jRxs1mKf15rBvYwQ/Lj9yA+thzu6\nWTO4d0Zi0NStZIh2Er0lvxajjTGuZjd3sJChcu9vOcwa0+yTqJFYwChvZA9jBD9gbtFBTYl1Vn65\njGPgWFY2ueZmXjOvaZ9eKaVbgVsBal09M7LbdtHIEPtqbYzFkR9au1vamTs2QtvYKMO1lpkIRZMU\nJN7MLnbQyla6ig5nSi6nj720cE/JN1DHs8Y026ynj6vYA0AfNf6Gpewq+Ya4dVZuOY1j4FhWNrnm\nZl4zL49dQCqtN7CHMxjkk/Q853ScMlnMMJfRxxd4AZQ4D2m2u5t5fJjlbGAJjzOHd9HLCoaKDksV\nlss4Bo5lUpWdsCmMiM8DdwNrIuLJiPjt6Q9rYna3zmHe2DC1dORsX4tGBxmotZZyD0KVXEw/l9HH\n7fTwWMmnlL+KPfyITnbQRidjdDY2DVpJdDIGJb3myRrTbNNHC4/Tzv108VGWso8a6+krOqwpsc7K\nK6dxDBzLyijX3Mxr5p3wnJuU0jUzEchkbGvvppXEiwb7eaTjFzObrTm4l23t+cx0lqPzGOAt7OIO\nFnJvBqeoLGOY0xjm/KNmnltHP+vo54OsZE8JT3GzxjSbjRE8zRx6GCk6lCmxzsopt3EMHMvKKNfc\nzGvmlfr00bvnLmVvrY2r9zx6+L7OsRFe0/cEGxesLDAyPZ8Xc5Br2cl3mc9dmUxT/hkWcwvLjvjZ\nS43NdHELy9hHOfdoWWOazVpJnMYQO0u4kTqedVY+OY5j4FhWRrnmZl4zr9Qj6WCtlZuXnc2N27ew\nu6WdbR3dXP/sA9RSYkPP2qLDm5JVDNLDCIsa01ufyUHmMcpOWnm8xKeoLGeYd9LLdtrYzFxWM3j4\nsX5q7KStwOgm71jvyQjBblp5pMRf2pxzjUG+dZZjXhcwwFkc4AE62UsLCxjlYvrpZpTvlHza/Jzr\nLMd1MddxDBzLyijX3Mxr5pW6KQS4aelLqKXEDb33sXhkkHu7erjihevpbessOrQpWUc/FzFw+Pbr\nqE9Tezdz+VRJB1KA1QzSRaKLYT7A9iMeq+fWU1BkOp5cawzyrbMc89pBGxcywK+yiy7G6KOFR2nn\nz1jOM8wpOrwpy7XOclwXHcfKKdcag3xzM6+ZFSk1/6LhWldPal1zZdOXW7Rrt2wqOgQJgNvOvaTo\nEKaNdabZItc6s8Y0W+RaY9JsMbLtTsb275zQVMKlvqZQkiRJkjQ1NoWSJEmSVGE2hZIkSZJUYTaF\nkiRJklRhNoWSJEmSVGE2hZIkSZJUYTaFkiRJklRhNoWSJEmSVGE2hZIkSZJUYTaFkiRJklRhNoWS\nJEmSVGE2hZIkSZJUYTaFkiRJklRhNoWSJEmSVGE2hZIkSZJUYTaFkiRJklRhNoWSJEmSVGE2hZIk\nSZJUYa1FB1Amt517SdEhSABcu2VT0SFIh+X62WidabawxqTpl2udTZRHCiVJkiSpwmwKJUmSJKnC\nbAolSZIkqcJsCiVJkiSpwmwKJUmSJKnCbAolSZIkqcJsCiVJkiSpwmwKJUmSJKnCbAolSZIkqcJs\nCiVJkiSpwmwKJUmSJKnCbAolSZIkqcJsCiVJkiSpwmwKJUmSJKnCbAolSZIkqcJsCiVJkiSpwmwK\nJUmSJKnCbAolSZIkqcJsCiVJkiSpwmwKJUmSJKnCbAolSZIkqcJsCiVJkiSpwlpP9ISIOA34NLAM\nSMCtKaW/mu7AJiwlPtB7H9ftfJiekUE2d/Xw3pUXsrVrcdGRTU2ueUG+uWWa18X0cxYHWM0g8xjj\nFpbxCB1FhzVl3YxwKX2s5SA9jDBAjW108BUWsvfEH42l8SZ28Sr6uYsFfIlFRYczNZnWGORbZznm\n1ULiWnayiiG6GWWQ4OfM4U4W8jjtRYc3dZnWWY7rIjiWldIsrbGJHCkcAd6XUloLvAx4V0Ssnd6w\nJu6G3vu5cftWPrL0JVx1xqXsq7XyzZ9uZNnw/qJDm5Jc84J8c8s1rwvZx1xGeTCDwXO8VQxxDgf4\nPnPZwFK+zCJWM8gNbKedsaLDa4oVDPFy9nGAKDqUpsi1xiDfOssxrxqJBGxkARtYwmdZzBwS72EH\nPQwXHd6U5VpnOa6L4FhWRrO1xk7YFKaUnkkp/bDxez/wELByugObiPaxEd6/4z5uWnY2H1uylk3z\nT+Ga09eRgN/Z+VDR4U1arnlBvrnlmhfAzSznJlbwLbqLDqWpfkIHH+IUNtLNI3Swmbl8lKUsZpTz\nKPfGzyFvYTebWMD+DK4UyLnGIN86yzGvYWp8nCV8j/lso5OtdLGBpbSSOIcDRYc3JTnXWY7rIjiW\nlc1srrGT+utGxOnAecA90xHMybpooJfusWG+uPD0w/ftb2nj692rWN/3VHGBTVGueUG+ueWaF0DK\nZM/c0Q5QY+yo3HppY5Cgm9GComqe8xlgOcNsZEHRoTRFzjUG+dZZrnkdbZBghKCVVHQoU5JzneW6\nLjqWlctsrrEJN4URMQ/4EvC7KaW+Yzx+XURsjojNaeRgM2M8rjWDexkh+HH7kSvKwx3drBncOyMx\nTIdc84J8c8s1r6pZyRDtJHpLfh1GG2NczW7uYCFDGexZBWtMs1GiRmIBo7yRPYwR/IC5RQc1JdZZ\nHhzLZq/ZXGMTWlsioo16Q/i5lNKXj/WclNKtwK0Ata6eGdlVtmhkiH21NsbiyBVld0s7c8dGaBsb\nZbjWMhOhNFWueUG+ueWaV5UEiTezix20spWuosOZksvpYy8t3FPyDdTxrDHNNuvp4yr2ANBHjb9h\nKbtKvhFunZWfY9nsNptr7IRtd0QE8AngoZTSLdMfkiTNvDewhzMY5JP0POdUnDJZzDCX0ccXeAGU\nOA9ptrubeXyY5WxgCY8zh3fRywqGig5LFedYpsmayLHYXwHeAVwSEVsaP1dMc1wTsrt1DvPGhqml\nI2dXWjQ6yECttbR7s3LNC/LNLde8quJi+rmMPm6nh8dKPqX8VezhR3SygzY6GaOzsVnQSqKTMSjp\nNU/WmGabPlp4nHbup4uPspR91FjPc66uKRXrrNwcy2a/2VxjJzzPIaX0PWZpi76tvZtWEi8a7OeR\njl/MJrXm4F62tZd3dqlc84J8c8s1ryo4jwHewi7uYCH3ZnCKyjKGOY1hzj9q1rl19LOOfj7ISvaU\n8BQ3a0yz2RjB08yhh5GiQ5kS66y8HMvKYTbXWKmv2rx77lL21tq4es+jh+/rHBvhNX1PsHHBrPjW\njEnJNS/IN7dc88rdiznItezku8znrkymKf8Mi7mFZUf87KXGZrq4hWXso5x7+q0xzWatJE5jiJ0l\n3EgdzzorJ8ey8pjNNVbqT6/BWis3LzubG7dvYXdLO9s6urn+2QeopcSGnrVFhzdpueYF+eaWa14A\nqxikhxEWNaa2PpODzGOUnbTyeIlPT1nOMO+kl+20sZm5rGbw8GP91NhJW4HRTd6x3pMRgt208kiJ\nv7Q55xqDfOssx7wuYICzOMADdLKXFhYwysX0080o3yn5tPk511mO6yI4lpXNbK6xUjeFADctfQm1\nlLih9z4Wjwxyb1cPV7xwPb1tnUWHNiW55gX55pZrXuvo5yIGDt9+HfUpk+9mLp8q8UC6mkG6SHQx\nzAfYfsRj9dx6CopMx5NrjUG+dZZjXjto40IG+FV20cUYfbTwKO38Gct5hjlFhzdludZZjusiOJaV\n0WytsUip+Rdq1rp6UuuaK5u+XEl1127ZVHQI0mG3nXtJ0SFMC+tMs4U1Jk2/HOtsZNudjO3fOaG5\nYUp9TaEkSZIkaWpsCiVJkiSpwmwKJUmSJKnCbAolSZIkqcJsCiVJkiSpwmwKJUmSJKnCbAolSZIk\nqcJsCiVJkiSpwmwKJUmSJKnCbAolSZIkqcJsCiVJkiSpwmwKJUmSJKnCbAolSZIkqcJsCiVJkiSp\nwmwKJUmSJKnCbAolSZIkqcJsCiVJkiSpwmwKJUmSJKnCIqXU9IXWunpS65orm75cSfm7dsumokOQ\nALjt3EuKDmFaWGOaLXKtMbDONDt8iWd4Ng3GRJ7rkUJJkiRJqjCbQkmSJEmqMJtCSZIkSaowm0JJ\nkiRJqjCbQkmSJEmqMJtCSZIkSaowm0JJkiRJqjCbQkmSJEmqMJtCSZIkSaowm0JJkiRJqjCbQkmS\nJEmqMJtCSZIkSaowm0JJkiRJqjCbQkmSJEmqMJtCSZIkSaowm0JJkiRJqjCbQkmSJEmqMJtCSZIk\nSaowm0JJkiRJqjCbQkmSJEmqMJtCSZIkSaowm0JJkiRJqrATNoUR0RER34+IrRHxQET80UwEJkmS\nJEmafq0TeM4gcElKaV9EtAHfi4hvppT+eZpjm5iU+EDvfVy382F6RgbZ3NXDe1deyNauxUVHNjW5\n5gX55mZepXMx/ZzFAVYzyDzGuIVlPEJH0WFNSTcjXEofazlIDyMMUGMbHXyFheyd0Ed+ObyJXbyK\nfu5iAV9iUdHhTF2mdZZjjUG+ebWQuJadrGKIbkYZJPg5c7iThTxOe9HhTU2mNQZ5ro+OZTPvhEcK\nU92+xs0aUpIrAAANSElEQVS2xk+a1qhOwg2993Pj9q18ZOlLuOqMS9lXa+WbP93IsuH9RYc2Jbnm\nBfnmZl7lcyH7mMsoD5Z88BxvFUOcwwG+z1w2sJQvs4jVDHID22lnrOjwmmIFQ7ycfRwgig6laXKt\nsxxrDPLNq0YiARtZwAaW8FkWM4fEe9hBD8NFhzcludYY5Lk+OpbNvAldUxgRLRGxBegF7kop3TO9\nYU1M+9gI799xHzctO5uPLVnLpvmncM3p60jA7+x8qOjwJi3XvCDf3MyrnG5mOTexgm/RXXQoTfMT\nOvgQp7CRbh6hg83M5aMsZTGjnEf5N34A3sJuNrGA/ZlcFp9zneVYY5BvXsPU+DhL+B7z2UYnW+li\nA0tpJXEOB4oOb9JyrjHIc310LJt5E4oipTSaUjoXOBV4aUScdfRzIuK6iNgcEZvTyMFmx3lMFw30\n0j02zBcXnn74vv0tbXy9exXr+56akRimQ655Qb65mVc5pVmyd66ZDlBj7Ki8emljkKCb0YKiap7z\nGWA5w2xkQdGhNE3OdZZjjUG+eR3LIMEIQevsOUnspOVcY5Dn+uhYNvNOqjVNKe0BvgtcfozHbk0p\nXZBSuiBaZ+bw9ZrBvYwQ/Lj9yD/owx3drBncOyMxTIdc84J8czMvzWYrGaKdRG/Jr8NoY4yr2c0d\nLGRoluxZbQbrTLNPokZiAaO8kT2MEfyAuUUHNWnWWB4cy6bXCf+qEbEEGE4p7YmITuAy4L9Me2QT\nsGhkiH21NsbiyD/o7pZ25o6N0DY2ynCtpaDoJi/XvCDf3MxLs1WQeDO72EErW+kqOpwpuZw+9tLC\nPSXeOD0W60yzzXr6uIo9APRR429Yyq4Sb4hbY+XnWDb9JtKergC+GxH3AT+gfk3h16Y3LElSM7yB\nPZzBIJ+k5zmn4pTJYoa5jD6+wAugxHlIZXA38/gwy9nAEh5nDu+ilxUMFR2WKsyxbPqdcLdPSuk+\n4LwZiOWk7W6dw7yxYWpp7Ii9P4tGBxmotZZ2r0+ueUG+uZmXZqOL6ecy+vgEPTxW8unkr2IPP6KT\nHbTR2Zh5LoBWEp2MNWZvm10D7ERZZ5pt+mihj/p69wCd/CFPs54+bqen4MgmxxorN8eymVHecwGA\nbe3dtJJ40WA/j3T8YsalNQf3sq29vDMw5ZoX5JubeWm2OY8B3sIu7mAh986yU1QmYxnDnMYw5x81\n69w6+llHPx9kJXtKOqRZZ5rNxgieZg49jBQdyqRZY+XlWDZzZs/VjZNw99yl7K21cfWeRw/f1zk2\nwmv6nmDjgpUFRjY1ueYF+eZmXppNXsxBrmUn32U+d2UyRflnWMwtLDviZy81NtPFLSxjH+Xd02+d\naTZrJXEaQ+ws6U4XsMbKyrFsZpW3woHBWis3LzubG7dvYXdLO9s6urn+2QeopcSGnrVFhzdpueYF\n+eZmXuW0ikF6GGFRY3rrMznIPEbZSSuPl/QUleUM80562U4bm5nLagYPP9ZPjZ20FRjd5B3r/Rgh\n2E0rj5T8C5tzrrMcawzyzesCBjiLAzxAJ3tpYQGjXEw/3YzynVk0df7JyrnGIM/10bFs5pW6KQS4\naelLqKXEDb33sXhkkHu7erjihevpbessOrQpyTUvyDc38yqfdfRzEQOHb7+O+tTkdzOXT5V0IF3N\nIF0kuhjmA2w/4rF6XuW8Jih3udZZjjUG+ea1gzYuZIBfZRddjNFHC4/Szp+xnGeYU3R4U5JrjUGe\n66Nj2cyLlJr/ZaS1rp7UuubKpi9XUv6u3bKp6BAkAG4795KiQ5gW1phmi1xrDKwzzQ5f4hmeTYMT\nmrmm1NcUSpIkSZKmxqZQkiRJkirMplCSJEmSKsymUJIkSZIqzKZQkiRJkirMplCSJEmSKsymUJIk\nSZIqzKZQkiRJkirMplCSJEmSKsymUJIkSZIqzKZQkiRJkirMplCSJEmSKsymUJIkSZIqzKZQkiRJ\nkirMplCSJEmSKsymUJIkSZIqzKZQkiRJkirMplCSJEmSKixSSs1faMSzwM+bvmBJkiRJ0kT8q5TS\nkok8cVqaQkmSJElSOXj6qCRJkiRVmE2hJEmSJFWYTaEkSZIkVZhNoSRJkiRVmE2hJEmSJFWYTaEk\nSZIkVZhNoSRJkiRVmE2hJEmSJFWYTaEkSZIkVZhNoSRpxkXEKyPiySn8/xsj4uPNjKkIEfFARLxy\nkv93bURsjohoclgnG8frIuLvi4xBkjQ1NoWSVAER8bZGA7EvIp6JiG9GxL8pOq6JOFYDmVL6cErp\n303Da/1mRKSI+Iuj7n994/7bJ7ic2yPiT070vJTSL6eU/nFy0fLHwEdSSumo1z4zIg5GxGcnuqCI\n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\nhQAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f8c0a25bcf8>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "show_kernel(C, 'Convolution Matrix')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "If we reshape the input into a column vector, we can use the matrix multiplication to perform convolution."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "def column_vector(m):\n",
    "    return m.flatten().reshape(-1, 1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[3],\n",
       "       [3],\n",
       "       [3],\n",
       "       [6],\n",
       "       [7],\n",
       "       [4],\n",
       "       [3],\n",
       "       [2],\n",
       "       [7],\n",
       "       [2],\n",
       "       [8],\n",
       "       [5],\n",
       "       [1],\n",
       "       [7],\n",
       "       [3],\n",
       "       [1]])"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x = column_vector(inputs)\n",
    "x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\nBnkUg/4/TCE7iwOzGZsAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f8c0a25b668>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "show_inputs(x)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 84.],\n",
       "       [ 84.],\n",
       "       [ 87.],\n",
       "       [ 60.]])"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "output = C @ x\n",
    "output"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n03tCo/XTUgAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f8c0a2ae438>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "show_output(output)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We reshape it into the desired shape."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 84.,  84.],\n",
       "       [ 87.,  60.]])"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "output = output.reshape(rows, cols)\n",
    "output"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\ny8zELo7wMQAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f8c0a087320>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "show_output(output)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This is exactly the same output as before."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Transposed Convolution Matrix\n",
    "\n",
    "Let's transpose the convolution matrix."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f8c09ffcf98>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "show_kernel(C.T, 'Transposed Convolution Matrix')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let's make a new input whose shape is 4x1."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[2],\n",
       "       [3],\n",
       "       [4],\n",
       "       [3]])"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x2 = np.random.randint(1, 5, size=(4, 1))\n",
    "x2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\nDflfsNx3LtdOTbIAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f8c09ffceb8>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "show_inputs(x2)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We matrix-multiply `C.T` with `x2` to up-sample `x2` from 4 (2x2) to 16 (4x4).  This operation has the same connectivity as the convolution but in the backward direction.  \n",
    "\n",
    "As you can see, 1 value in the input `x2` is connected to 9 values in the output matrix via the transposed convolution matrix."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[  2.],\n",
       "       [  7.],\n",
       "       [ 14.],\n",
       "       [ 12.],\n",
       "       [  6.],\n",
       "       [ 16.],\n",
       "       [ 31.],\n",
       "       [ 21.],\n",
       "       [  6.],\n",
       "       [ 14.],\n",
       "       [ 29.],\n",
       "       [ 21.],\n",
       "       [  4.],\n",
       "       [ 11.],\n",
       "       [ 22.],\n",
       "       [ 12.]])"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "output2 = (C.T @ x2)\n",
    "output2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f8c09e53f60>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "show_output(output2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[  2.,   7.,  14.,  12.],\n",
       "       [  6.,  16.,  31.,  21.],\n",
       "       [  6.,  14.,  29.,  21.],\n",
       "       [  4.,  11.,  22.,  12.]])"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "output2 = output2.reshape(4, 4)\n",
    "output2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n34HLsn0AAAAASUVORK5CYII=\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f8c0a2ae748>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "show_output(output2)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Summary\n",
    "\n",
    "As discussed at the beginning of this notebook the weights in the tranposed convolution matrix can be trained as part of a neural network back-propagation process.  As such, it eliminates the necessity for fixed up-sampling methods.\n",
    "\n",
    "Note: we can emulate the transposed convolution using a direct convolution.  We first up-sample the input by adding zeros between the original values in a way that the direct convolution produces the same effect as the transposed convolution.  However, it is less efficient due to the need to add zeros to up-sample the input before the convolution."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## References\n",
    "\n",
    "<a id=\"ref1\"></a>\n",
    "### [1] A guide to convolution arithmetic for deep learning\n",
    "\n",
    "Vincent Dumoulin, Francesco Visin\n",
    "\n",
    "https://arxiv.org/abs/1603.07285\n",
    "\n",
    "<a id=\"ref2\"></a>\n",
    "### [2] Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks\n",
    "\n",
    "Alec Radford, Luke Metz, Soumith Chintala\n",
    "\n",
    "https://arxiv.org/pdf/1511.06434v2.pdf\n",
    "\n",
    "<a id=\"ref3\"></a>\n",
    "### [3] Fully Convolutional Networks for Semantic Segmentation\n",
    "\n",
    "Jonathan Long, Evan Shelhamer, Trevor Darrell\n",
    "\n",
    "https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf\n",
    "\n",
    "<a id=\"ref4\"></a>\n",
    "### [4] Deconvolution and Checkerboard Artifacts\n",
    "\n",
    "Augustus Odena, Vincent Dumoulin, Chris Olah\n",
    " \n",
    "https://distill.pub/2016/deconv-checkerboard/\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
