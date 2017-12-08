import numpy as np
from bokeh.plotting import figure
from bokeh.resources import CDN
from bokeh.embed import file_html
from bokeh.models import LinearColorMapper, BasicTicker, ColorBar
from bokeh.palettes import Category10

from jinja2 import Template

template = Template('{{ plot_div }}\n{{ plot_script }}')

def save_plot(plot, title):
    plot_html = file_html(plot, CDN, template=template)
    with open(f'{title}.html', 'w') as html_file:
        html_file.write(plot_html)

def plot_unit_gaussian_samples(D):
    p = figure(plot_width=800, plot_height=500, title='Samples from a unit {}D Gaussian'.format(D), sizing_mode='scale_width')

    xs = np.linspace(0, 1, D)
    for color in Category10[10]:
        ys = np.random.multivariate_normal(np.zeros(D), np.eye(D))
        p.line(xs, ys, line_width=1, color=color)
    return p

save_plot(plot_unit_gaussian_samples(2), '2d_samples')

save_plot(plot_unit_gaussian_samples(20), '20d_samples')

def k(xs, ys, sigma=1, l=1):
    """Sqared Exponential kernel as above but designed to return the whole
    covariance matrix - i.e. the pairwise covariance of the vectors xs & ys.
    Also with two parameters which are discussed at the end."""
    dx = np.expand_dims(xs, 1) - np.expand_dims(ys, 0)  # Pairwise difference matrix.
    return (sigma ** 2) * np.exp(-((dx / l) ** 2) / 2)

def m(x):
    """As discussed, we can let the mean always be zero."""
    return np.zeros_like(x)

N = 100
x = np.linspace(-2, 2, N)
y = np.linspace(-2, 2, N)
d = k(x, y)

color_mapper = LinearColorMapper(palette="Plasma256", low=0, high=1)

p = figure(plot_width=400, plot_height=400, x_range=(-2, 2), y_range=(-2, 2),
           title='Visualisation of k(x, x\')', x_axis_label='x', y_axis_label='x\'', toolbar_location=None, sizing_mode='scale_width')
p.image(image=[d], color_mapper=color_mapper, x=-2, y=-2, dw=4, dh=4)

color_bar = ColorBar(color_mapper=color_mapper, ticker=BasicTicker(),
                     label_standoff=12, border_line_color=None, location=(0,0))

p.add_layout(color_bar, 'right')

save_plot(p, 'kernel_viz')

p = figure(plot_width=800, plot_height=500, title='Samples from a 20D Gaussian with kernel smoothing', sizing_mode='scale_width')
D = 20
xs = np.linspace(0, 1, D)
for color in Category10[10]:
    ys = np.random.multivariate_normal(m(xs), k(xs, xs))
    p.circle(xs, ys, size=3, color=color)
    p.line(xs, ys, line_width=1, color=color)

save_plot(p, '20d_samples_smooth')

n = 100
xs = np.linspace(-5, 5, n)
K = k(xs, xs, sigma=1, l=1)
mu = m(xs)

p = figure(plot_width=800, plot_height=500, title='5 samples from GP prior', sizing_mode='scale_width')

for color in Category10[5]:
    ys = np.random.multivariate_normal(mu, K)
    p.line(xs, ys, line_width=2, color=color)

save_plot(p, 'prior_samples')

# coefs[i] is the coefficient of x^i
coefs = [6, -2.5, -2.4, -0.1, 0.2, 0.03]

def f(x):
    total = 0
    for exp, coef in enumerate(coefs):
        total += coef * (x ** exp)
    return total

xs = np.linspace(-5.0, 3.5, 100)
ys = f(xs)

p = figure(plot_width=800, plot_height=400, x_axis_label='x', y_axis_label='f(x)',
           title='The hidden function f(x)', sizing_mode='scale_width')
p.line(xs, ys, line_width=2)

save_plot(p, 'hidden_function')

x_obs = np.array([-4, -1.5, 0, 1.5, 2.5, 2.7])
y_obs = f(x_obs)

x_s = np.linspace(-8, 7, 80)

K = k(x_obs, x_obs) + np.eye(x_obs.shape[0]) * 1e-8
K_s = k(x_obs, x_s)
K_ss = k(x_s, x_s)

K_sTKinv = np.matmul(K_s.T, np.linalg.pinv(K))

mu_s = m(x_s) + np.matmul(K_sTKinv, y_obs - m(x_obs))
Sigma_s = K_ss - np.matmul(K_sTKinv, K_s)

p = figure(plot_width=800, plot_height=600, y_range=(-7, 8), title='GP posterior', sizing_mode='scale_width')

y_true = f(x_s)
p.line(x_s, y_true, line_width=3, color='black', alpha=0.4, line_dash='dashed', legend='True f(x)')

p.cross(x_obs, y_obs, size=20, legend='Training data')

stds = np.sqrt(Sigma_s.diagonal())
err_xs = np.concatenate((x_s, np.flip(x_s, 0)))
err_ys = np.concatenate((mu_s + 2 * stds, np.flip(mu_s - 2 * stds, 0)))
p.patch(err_xs, err_ys, alpha=0.2, line_width=0, color='grey', legend='Uncertainty')

for color in Category10[3]:
    y_s = np.random.multivariate_normal(mu_s, Sigma_s)
    p.line(x_s, y_s, line_width=1, color=color)

p.line(x_s, mu_s, line_width=3, color='blue', alpha=0.4, legend='Mean')
save_plot(p, 'post_samples')
