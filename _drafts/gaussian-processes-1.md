# Introduction to Gaussian Processes - Part I
Author: [@alex_bridgland](https://twitter.com/alex_bridgland) | [@bridgo](https://github.com/Bridgo)
***
This is a notebook that I originally wrote when learning about Gaussian Processes. I thought it might be helpful for others in the same situation so I added a writeup to go with the code - hopefully it offers a bit of a new perspective!

I aimed for it to give a more visual / intuitive introduction without totally abandoning the theory. And since it's a notebook you should be able to get an even better understanding by downloading the notebook and playing with the code. Any/all feedback appreciated!

## What is a Gaussian Process and why would I use one?
***
A Gaussian process is a powerful model that can be used to represent a distribution over functions. Most techniques in machine learning tend to avoid this by parameterising functions and modeling their parameters (e.g. the weights in linear regression). However by modeling the function itself we can get useful uncertainty information. Quantifying uncertainty can be extremely valuable - for example if we can explore (request more data) we can chose to explore the areas we are least certain about to be as efficient as possible. For more information on the importance of uncertainty modeling see [this article](http://mlg.eng.cam.ac.uk/yarin/blog_3d801aa532c1ce.html) by Yarin Gal.

For this introduction we will consider the standard supervised learning setting (without noise):
- We assume there is some hidden function $f:\mathbb{R}^D\rightarrow\mathbb{R}$ that we want to model.
- We have data $\mathcal{D}=\{(\mathbf{x}_i, y_i):i=1, \ldots, N\}$ where $y_i = f(\mathbf{x}_i)$.
- We want to predict the value of $f$ at some new, unobserved points.

## Modelling Functions
***
The key idea behind GPs is that a function can be modeled using an infinite dimensional multivariate Gaussian distribution. In other words, every point in the input space is associated with a random variable and the joint distribution of these is modeled as a multivariate Gaussian.

Ok, so what does that mean and what does it actually look like? Well lets start with a simpler case: a unit 2D Gaussian. How can we start to view this as a distribution over functions? Here's what we have:

$$\left(\array{y_0\\ y_1}\right)\sim\mathcal{N}\left(\left(\array{0\\ 0}\right), \left(\array{1 & 0\\0 & 1}\right)\right)$$

Normally this is visualised as a 3D bell curve with the probability density represented as height. But what if instead of visualising the whole distribution we just sample from the distribution. Then we will have two values which we can plot on an graph. Let's do this 10 times, putting the first value at $x=0$ and the second at $x=1$ and then drawing a line between them.

```python
def plot_unit_gaussian_samples(D):
    p = figure(plot_width=800, plot_height=500, title='Samples from a unit {}D Gaussian'.format(D))

    xs = np.linspace(0, 1, D)
    for color in Category10[10]:
        ys = np.random.multivariate_normal(np.zeros(D), np.eye(D))
        p.line(xs, ys, line_width=1, color=color)
    return p

show(plot_unit_gaussian_samples(2))
```
