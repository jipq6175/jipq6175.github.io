---
layout: post
title: Walking through score-based diffusion with SDE
date: 2023-09-09 21:45:00
description: Some derivations and calculations of DSM with SDE
tags: reading solving
categories: models
---




### 0. Diffusion Model using Score Matching and SDE


Score matching is a technique that is used for the score-based genertive models, which regress on the scores, $$\nabla_x\log p(x)$$ instead of modeling the original data distribution $$p(x)$$. The score is the gradient of the data distribution so if we can access this gradient on every point of the data space, we can just follow this gradient and achieve the points with large probability density. This can be achieved by gradient ascend or Langevin dynamics which is similar to the stochastic gradient ascend. 

$$x_{t+1} = x_t + \epsilon\nabla_x\log p(x) + \sqrt{2\epsilon} z$$

where $$z~N(0, I)$$. The goal of training the score-based generative model is to approximate the scores using a neural network $$s_\theta(x) \approx \epsilon\nabla_x\log p(x)$$ everywhere on $$x$$. If we have such model, we can use the above Langevin dynamics for sampling, just replacing the score with model estimates: 

$$x_{t+1} = x_t + \epsilon s_\theta(x) + \sqrt{2\epsilon} z$$

The score-matching objectve (or loss function) is the Fischer divergence between $$s_\theta(x)$$ and $$\nabla_x\log p(x)$$, weighted by $$p(x)$$. 

$$\mathbb{E}_{p(x)}||s_\theta(x) - \nabla_x\log p(x)||_2^2 = \int p(x)||s_\theta(x) - \nabla_x\log p(x)||_2^2$$

The main issue with this simple score-matching is that in regions where $$p(x) \approx 0$$, the score extimate will be inaccurate due to the zero weighting in these regions. So if we start our sample in such region and try to follow the score (or gradient), we are doing some random walk and will never get closer to the modes of data distribution. This is more severe in the high-dimensional case where data distribution is like a bunch of spikes (delta functions) in the data space.

The trick to solve this if to introduce noise to the data distribution, trying to widen the distribution and cover as much space as possible. The introduction of the noise will result in non-zero $$p_{\sigma_i}(\tilde{x}\mid x)$$, where $$\tilde{x}$$ is the noised data controlled by variance $$\sigma_i^2$$, and cover larger data space for more accurate score estimate.

<br>


#### Score Matching with Langevin Dynamics (SMLD)

One way is to add different noises with increasing variances $$\{\sigma_0^2, \sigma_1^2, ..., \sigma_N^2\}$$, as proposed in the Noise Conditional Score Network (NCSN). The network $$s_\theta(x)$$ now need to be noise-conditioned, $$s_\theta(x, \sigma)$$ in order to match the scores at diferent variances. The NSCN objective is now

$$\sum_{i=1}^N \sigma_i^2 \mathbb{E}_{p(x)}\mathbb{E}_{p_{\sigma_i}(\tilde{x} \mid x)} || s(\tilde{x}, \sigma_i) - \nabla_{\tilde{x}}\log p_{\sigma_i}(\tilde{x}\mid x) ||_2^2$$


#### Denoising Diffusion Probabilistic Model (DDPM)

Another way is to add noise via a discrete Markov chain, $$p(x_i \mid x_{i-1}) = \mathcal{N}(x_i, \sqrt{1-\beta_i}x_{i-1}, \beta_i I)$$. Notice that this Markov chain attenuates the signal and adds noise, instead of just adding noise to overwhelm the signal as done in SMLD. The DDPM objective is now

$$\sum_{i=1}^N (1-\alpha_i) \mathbb{E}_{p(x)}\mathbb{E}_{p_{\alpha_i}(\tilde{x}\mid x)} || s(\tilde{x}, i) - \nabla_{\tilde{x}}\log p_{\alpha_i}(\tilde{x}\mid x) ||_2^2$$

Notice the similarity between the objectives of SMLD and DDPM. 

<br>

### 1. General Continuous-Time Diffusion

The SMLD and DDPM were descrete-time, with a pre-defined variance schedule: $$\sigma_i^2$$ for SMLD and $$\beta_i$$ for DDPM. A general case of adding noise to data is to use stochastic differential equations (SDEs), which involves adding gaussian noise (or Brownian motion). 

$$dx = f(x, t)dt + g(t)dw$$

where $$f(x, t)$$ is the drift term that depends on current $$x$$ and time $$t$$. $$g(t)$$ is the diffusion term, controlling how much noise to add to the data at certain time $$t$$ and $$dw$$ is the infinitesimal Brownian motion. With this predefined stochastic process, the time reveral of the SDE has close form

$$dx = [f(x, t) - g^2(t) \nabla_x\log p_t(x)]dt + g(t)dw$$

Note that here, in the time reversal, $$dt < 0$$, so we are actually reversing the drift and following along the gradient (in the same direction of the score). We now just need to have a time-conditioned score model $$s_\theta(x, t)$$ that matches $$\nabla_x\log p_t(x)$$ everywhere, everytime. 

There is also a guarantee that the distribution of $$x(t)$$ following this SDE, is a normal distribution with mean $$m(t)$$ and variance $$v(t)$$. So we can write down the perturbation kernel or transitional kernel from data $$x(0)$$ to noised data $$x(t)$$: 

$$p_{0t}\left(x(t) | x(0) \right) = \mathcal{N}\left(x(t); m(t), v(t)I \right)$$

We are just writing this down for derivation of $$m(t)$$ and $$v(t)$$ later. Note here that the data is actually multi-dimensional, but each dimensional is treated as independent, so we can just treat everything in scalar form and write $I$ for the variance.

The denoising score matching objective is now

$$\mathcal{L}_{dsm} = \mathbb{E}_t \lambda(t) \mathbb{E}_{x(0)} \mathbb{E}_{x(t)\mid x(0)} || s_\theta(x(t), t) - \nabla_{x(t)}\log p_{0t}\left(x(t) \mid x(0) \right)||_2^2$$

$$\lambda(t)$$ is the positive time-dependent weighting and is proportional to the variance squared $$v^2(t)$$ as done in SMLD and DDPM. In the maximum likelihood training proposed later, $$\lambda(t)$$ is proportional to the diffusion term squared $$g^2(t)$$. 

The gradient term can be easily calculated in exact form since $$p_{0t}$$ is a gaussian: 

$$\nabla_{x(t)}\log p_{0t}\left(x(t) \mid x(0) \right) = \nabla_{x(t)}\log \mathcal{N}\left(x(t); m(t), v(t)I \right) = -\frac{x(t) - m(t)}{v(t)}$$

How do we compute $$x(t)$$ in practice? Again, using $$p_{0t}$$: 

$$x(t) = m(t) + \sqrt{v(t)}z; z\sim \mathcal{N}(z; 0, I)$$

Plug $$x(t) - m(t) = \sqrt{v(t)}z$$ above and then $$\mathcal{L}_{dsm}$$ yields:

$$\mathcal{L}_{dsm} = \mathbb{E}_t \lambda(t) \mathbb{E}_{x(0)} \mathbb{E}_{x(t)\mid x(0)} \left|\left| s_\theta(x(t), t) + \frac{z}{\sqrt{v(t)}}\right|\right|_2^2$$

The critical components are $$m(t)$$ and $$v(t)$$. Once we have them, we can compute the loss and train the score model. 

<br>

#### SDE for SMLD (VE-SDE)

The discrete-time Markov chain for SMLD is $$x_i = x_{i-1} + \sqrt{\sigma_i^2 - \sigma_{i-1}^2} z_{i-1}$$. In the continuous-time generalization: 

$$x(t+\Delta t) = x(t) + \sqrt{\sigma^2(t+\Delta t) - \sigma^2(t)}z(t) \approx x(t) + \sqrt{\frac{d\sigma^2(t)}{dt}\Delta t}z(t)$$

We combine $$\sqrt{\Delta t}z(t) = dw$$. The continuous-time SMLD is then

$$dx = \sqrt{\frac{d\sigma^2(t)}{dt}}dw$$

which is also called variance-exploding SDE (VE-SDE). 

#### SDE for DDPM (VP-SDE)

The discrete-time Markov chain for DDPM is $$x_i = \sqrt{1-\beta_i}x_{i-1} + \sqrt{\beta_i}z_{i-1}$$. In the continuous-time generalization:

$$x(t+\Delta t) = \sqrt{1-\beta(t+\Delta t)\Delta t}x(t) + \sqrt{\beta(t+\Delta t)\Delta t}z(t) \approx x(t) - \frac{1}{2}\beta(t+\Delta t)\Delta t x(t) + \sqrt{\beta(t)\Delta t}z(t)$$

We combine $$\sqrt{\Delta t}z(t) = dw$$. The continuous0time DDPM is then

$$dx = -\frac{1}{2}\beta(t)x dt + \sqrt{\beta(t)}dw$$

which is also called variance-preserving SDE (VP-SDE). 

<br>

### 2. Marginal mean and variance from SDE

Now we will use the SDEs to derive $$m(t)$$ and $$v(t)$$ for the mean and variance of the perturbation kernel $$p_{0t}$$

Given an SDE with affine continuous function $f$ and $g$, 

$$dx = f(x, t)dt + g(x, t)dw$$

Let 

$$f(x, t) = A(t)x(t) + a(t)$$

$$g(x, t) = B(t)x(t) + b(t)$$

and 

$$\mathbb{E}[x(t)] = m(t)$$

$$\mathbb{Var}[x(t)] = v(t)$$

will satisfy the following ODEs with initial conditions: 

$$m'(t) = A(t)m(t) + a(t); m(0) = m_0$$

$$v'(t) = 2A(t)v(t) + b^2(t); v(0) = v_0$$

<br>

### 3. Solving variable coefficient ODEs

The above ODEs are variable coefficient ODEs and have general solution. The general ODE $$y'(t) = a(t)y(t) + b(t)$$ has solution

$$y(t) = Ce^{A(t)} + e^{A(t)}\int e^{-A(t)}b(t)dt$$

where 

$$A(t) = \int a(t)dt$$

<br>

### 4. Deriving perturbation kernels from SDE

The perturbation kernel $$p_{0t}$$ for SDE here is Gaussian. We are after the mean $$m(t)$$ and variance $$v(t)$$ for the distribution of $$x(t)$$ given initial data point $$x(0)$$. Note that because $$x(0)$$ is out data point and we treat its distribution as a delta function, or a super tight gaussian with mean $$x(0)$$ and variance $$v(0)=0I$$.

<br>

### VE-SDE

$$dx = \sqrt{\frac{d\sigma^2(t)}{dt}}dw$$

Using the above notation, $$A(t) = a(t) = B(t) = 0$$ and $$b(t) = \sqrt{d\sigma^2(t)/dt}$$. 

$$m'(t) = 0 \Rightarrow m(t) = c = x(0)$$

$$v'(t) = b^2(t) = \frac{d\sigma^2(t)}{dt} \Rightarrow v(t) = \sigma^2(t) + c = \sigma^2(t)$$

Therefore, 

$$p_{0t}(x(t) | x(0)) = \mathcal{N}\left(x(t); x(0), \sigma^2(t)I \right)$$

<br>

### VP-SDE

$$dx = -\frac{1}{2}\beta(t)x dt + \sqrt{\beta(t)}dw$$

$$a(t) = B(t) = 0$$, $$A(t) = -\beta(t)/2$$ and $$b(t) = \sqrt{\beta(t)}$$. Plug in the ODEs: 

$$m'(t) = -\frac{1}{2} \beta(t)m(t) \Rightarrow m(t) = Ce^{\int_0^t-\frac{1}{2}\beta(s)ds} = x(0)e^{\int_0^t-\frac{1}{2}\beta(s)ds}$$

$$v'(t) = -\beta(t)v(t)+\beta(t)$$

$$v(t) = Ce^{-\int\beta(t)dt} + e^{-\int\beta(t)dt}\int e^{\int\beta(t)dt}\beta(t)dt = Ce^{-\int\beta(t)dt} + 1; v(0)=0 \Rightarrow C=-1$$

$$v(t) = 1 - e^{-\int\beta(t)dt}$$

Therefore, 

$$p_{0t}(x(t) | x(0)) = \mathcal{N}\left(x(t); x(0)e^{\int_0^t-\frac{1}{2}\beta(s)ds}, (1 - e^{-\int_0^t\beta(s)ds})I \right)$$

<br>

### sub VP-SDE

In the score-based SDE model, the author introduces another SDE called sub-VP SDE

$$dx = -\frac{1}{2}\beta(t)dt + \sqrt{\beta(t)\left(1-e^{-2\int_0^t\beta(s)ds} \right)}dw$$

Now we only change the $$b(t)$$ so $$m(t)$$ remains the same as VP-SDE. 

$$v'(t) = -\beta(t)v(t) + \beta(t)\left(1-e^{-2\int_0^t\beta(s)ds} \right)$$

$$v(t) = Ce^{-\int\beta(t)dt} + e^{-\int\beta(t)dt}\int e^{\int\beta(t)dt}\beta(t)dt + e^{-\int\beta(t)dt}\int e^{-\int\beta(t)dt}\beta(t)dt$$

The first two terms are the same as before: 

$$v(t) =Ce^{-\int_0^t\beta(s)ds} + 1 + e^{-\int_0^t\beta(s)ds}e^{-\int_0^t\beta(s)ds} = Ce^{-\int_0^t\beta(s)ds} + 1 + e^{-2\int_0^t\beta(s)ds}$$

With $v(0) = 0 \Rightarrow C = -2$

$$v(t) = -2e^{-\int_0^t\beta(s)ds} + 1 + e^{-2\int_0^t\beta(s)ds} = \left(1-e^{-\int_0^t\beta(s)ds}\right)^2$$

Therefore, 

$$p_{0t}(x(t) | x(0)) = \mathcal{N}\left(x(t); x(0)e^{\int_0^t-\frac{1}{2}\beta(s)ds}, \left(1 - e^{-\int_0^t\beta(s)ds}\right)^2I \right)$$

Note that the variance of sub VP-SDE is bounded by (or always less than) the variance of VP-SDE. 

$$\forall t > 0; \left(1 - e^{-\int_0^t\beta(s)ds}\right)^2 \le 1 - e^{-\int_0^t\beta(s)ds}$$

These 3 SDEs appear in Eq.(29) of [1] and Table 1 of [2]. In the following code, we will use VE-SDE as an example. 


### 5. Notebook

The original notebook is provided by the author: [Google Colab](https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing). 

Most of the cells contain similar code to the DDPM, especially the time-conditional UNet model. I'll just pick the cells I did not grasp during the first pass. If I have time after work, I might implement all the training and sampling for the above 3 SDEs. 

<br>

#### Cell #5

Cell #5 sets up the VE-SDE scheduling for the diffusion coefficient $$g(t)$$ and standard deviation $$\sqrt{v(t)}$$. Note that in VE-SDE $$f(x, t) = 0$$ so the mean of $$x(t)$$: $$m(t) = 0$$. 


We set up the Stochastic Differential Equation (VE-SDE) as the following

$$dx = \sigma^t dw$$

where $$\sigma > 1.0$$ is the standard deviation by design and $$dw$$ is the Wiener process. 

This setup is not unique. The marginal probability standard deviation can be customized. The marginal probability variance is then

$$v(t) = \int_0^t g(s)^2ds$$

One can try different type of SDEs. One can verify that if $$g(t) = \sigma^t$$ then 

$$v(t) = \frac{\sigma^{2t} - 1}{2\log{\sigma}}$$


{% highlight python %}
# Set up the VE-SDE: dx = sigma^t dw

def diffusion_coeff(t, sigma, device=device):
    """Compute the diffusion coefficient of our SDE.

    Args:
        t: A vector of time steps.
        sigma: The $\sigma$ in our SDE.

    Returns:
        The vector of diffusion coefficients.
    """
    return torch.tensor(sigma**t, device=device)


def marginal_prob_std(t, sigma, device=device):
    """Compute the standard deviation of $p_{0t}(x(t) | x(0))$.

    Args:    
        t: A vector of time steps.
        sigma: The $\sigma$ in our SDE.  

    Returns:
        The standard deviation.
    """    
    t = t.clone().to(device)
    return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))


sigma = 25.
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)
{% endhighlight %}


<br>


#### Cell #6

Cell #6 sets up the loss function for the training objective. Recall that the score-function $$s_\theta(x, t)$$ has to match $$\nabla_{x(t)}\log p_t(x(t))$$ at everytime for every training data $$x(0)$$. The DSM loss is then the regression loss. 


{% highlight python %}
# Loss function: similar to the MSE loss 3
def loss_fn(model, x, marginal_prob_std, eps=1e-5):
    """The loss function for training score-based generative models.

    Args:
        model: A PyTorch model instance that represents a time-dependent score-based model.
        x: A mini-batch of training data.    
        marginal_prob_std: A function that gives the standard deviation of the perturbation kernel.
        eps: A tolerance value for numerical stability.
    """
    # x is the original signal, without purturbation or noising

    # uniformly sample random_t in [0, 1] using the batch dimension
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps 

    # Random Gaussian Noises
    z = torch.randn_like(x) 

    # Compute the std at these time points
    std = marginal_prob_std(random_t) 

    # forward computation p(x(t) | x(0)) = N(x(0), v(t))
    perturbed_x = x + z * std[:, None, None, None] 

    # compute the score 
    score = model(perturbed_x, random_t) 

    # true_score = - (x(t) - x(0)) / (std_t ^ 2)
    #            = - z_t / std_t
    # L_dsm = (pred_score - true_score) ^ 2 = (pred_score + z_t / std_t) ^ 2
    # scaled_L_dsm = (pred_score * std_t + z_t) ^ 2
    loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1,2,3))) 
    return loss

{% endhighlight %}

<br>


#### Cell #8

Cell #8 prepares sampling with 3 different methods: Euler-Maruyama, Predictor-Corrector and ODE. 

##### Euler-Maruyama

Recall that SDE of the form 

$$dx = f(x, t)dt + g(t)dw$$

has the reverse-time SDE:

$$dx = \left[f(x, t) - g^2(t)\nabla_{x(t)}\log p_t(x(t)) \right]dt + g(t)dw$$

$$dx = -\sigma^{2t} s_\theta(x, t) dt + \sigma^t dw; dt < 0$$

$$x_{t-\Delta t} = \mathbf{x}_t + \sigma^{2t} s_\theta(x_t, t)\Delta t + \sigma^t\sqrt{\Delta t} z_t$$

where $$z_t \sim \mathcal{N}(0, I)$$. 

Euler-Maruyama applies $$dt \sim \Delta t$$ discretization. 

{% highlight python %}
num_steps = 500
def Euler_Maruyama_sampler(score_model, 
                           marginal_prob_std,
                           diffusion_coeff, 
                           batch_size=64, 
                           num_steps=num_steps, 
                           device='cuda', 
                           eps=1e-3):
    """Generate samples from score-based models with the Euler-Maruyama solver.

    Args:
        score_model: A PyTorch model that represents the time-dependent score-based model.
        marginal_prob_std: A function that gives the standard deviation of the perturbation kernel.
        diffusion_coeff: A function that gives the diffusion coefficient of the SDE.
        batch_size: The number of samplers to generate by calling this function once.
        num_steps: The number of sampling steps. Equivalent to the number of discretized time steps.
        device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
        eps: The smallest time step for numerical stability.
    
    Returns:
        Samples.
    """
    t = torch.ones(batch_size, device=device)
    init_x = torch.randn(batch_size, 1, 28, 28, device=device) * marginal_prob_std(t)[:, None, None, None]
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    with torch.no_grad():
        for time_step in tqdm(time_steps):      
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            g = diffusion_coeff(batch_time_step)
            
            # mean_x = x + g^2(t) s(x, t) dt
            mean_x = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step) * step_size 
            
            # x = mean_x + g(t) \sqrt{dt} z; z ~ N(0, I)
            x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x) 
    
    # Do not include any noise in the last sampling step.
    return mean_x

{% endhighlight %}

<br>

##### Predictor-Corrector

Recall that given the score function $$s_\theta(x, t)$$, we can sample via Langevin dynamics:

$$x_{i+1} = x_i + \epsilon \nabla_{x_i} \log p(x_i) + \sqrt{2\epsilon} z_i$$

The PC sampling combines the ODE/SDE solver (Predictor) with $N$ steps of local Langevin dynamics (Correcor). 

1. Predictor: Use ODE/SDE solver for the next time step $x(t-dt)$ using $s(x(t), t)$
2. Corrector: Still using the score $s(x(t-dt), t-dt)$ and Langevin dynamics to correct for $x(t-dt)$ for $N$ steps

The step size $$\epsilon$$ is determined with predefined $$r$$: 

$$\epsilon = 2 \left(r \frac{\|z\|_2}{\|\nabla_{x} \log p(x)\|_2}\right)^2$$

which is determined by the norm of the score. The idea behind is like the adaptive step size in the stochastic gradient descent where we want to take a smaller (more careful) step when the score/gradient (slope) is steep to avoid overshoot (or sliding).  

{% highlight python %}
signal_to_noise_ratio = 0.16 

## The number of sampling steps.
num_steps =  500
def pc_sampler(score_model, 
               marginal_prob_std,
               diffusion_coeff,
               batch_size=64, 
               num_steps=num_steps, 
               snr=signal_to_noise_ratio,                
               device='cuda',
               eps=1e-3):
    """Generate samples from score-based models with Predictor-Corrector method.

    Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that gives the standard deviation
      of the perturbation kernel.
    diffusion_coeff: A function that gives the diffusion coefficient 
      of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    num_steps: The number of sampling steps. 
      Equivalent to the number of discretized time steps.    
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    eps: The smallest time step for numerical stability.

    Returns: 
    Samples.
    """
    t = torch.ones(batch_size, device=device)
    init_x = torch.randn(batch_size, 1, 28, 28, device=device) * marginal_prob_std(t)[:, None, None, None]
    time_steps = np.linspace(1., eps, num_steps)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    with torch.no_grad():
        for time_step in tqdm(time_steps):      
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            
            
            # N = 1, Corrector step (Langevin MCMC)
            # for N > 1, just wrap this part in a loop, but it will be expensive
            grad = score_model(x, batch_time_step)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = np.sqrt(np.prod(x.shape[1:]))

            # eps = 2 (r |z| / |g|) ^ 2
            langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2 
            x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)


            # Predictor step (Euler-Maruyama)
            g = diffusion_coeff(batch_time_step)
            x_mean = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step) * step_size
            x = x_mean + torch.sqrt(g**2 * step_size)[:, None, None, None] * torch.randn_like(x)      

    # The last step does not include any noise
    return x_mean

{% endhighlight %}

<br>

##### Probability flow ODE

For probability flow ODE, the reverse process: 

$$dx = \left[f(x, t) - \frac{1}{2}g^2(t)\nabla_{x(t)}\log p_t(x(t)) \right]dt$$

$$dx = -\frac{1}{2}\sigma^{2t} s_\theta(x, t) dt$$

$$\frac{dx}{dt} = -\frac{1}{2}\sigma^{2t} s_\theta(x, t)$$

Now we need to integrate from $$t=T$$ to $$t=0$$

$$x(t) = \int_T^t\frac{dx}{dt} dt + x(T) = \int_T^t -\frac{1}{2}\sigma^{2t} s_\theta(x, t) dt + x(T)$$

$$x(0) = \int_T^0 -\frac{1}{2}\sigma^{2t} s_\theta(x, t) dt + x(T)$$

the above can be solved using existent ODE solver, such as Runge-Kutta. 

{% highlight python %}
from scipy import integrate

## The error tolerance for the black-box ODE solver
error_tolerance = 1e-5 
def ode_sampler(score_model,
                marginal_prob_std,
                diffusion_coeff,
                batch_size=64, 
                atol=error_tolerance, 
                rtol=error_tolerance, 
                device='cuda', 
                z=None,
                eps=1e-3):
    """Generate samples from score-based models with black-box ODE solvers.

    Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that returns the standard deviation 
      of the perturbation kernel.
    diffusion_coeff: A function that returns the diffusion coefficient of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    atol: Tolerance of absolute errors.
    rtol: Tolerance of relative errors.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    z: The latent code that governs the final sample. If None, we start from p_1;
      otherwise, we start from the given z.
    eps: The smallest time step for numerical stability.
    """
    t = torch.ones(batch_size, device=device)
    # Create the latent code
    if z is None:
        init_x = torch.randn(batch_size, 1, 28, 28, device=device) * marginal_prob_std(t)[:, None, None, None]
    else:
        init_x = z

    shape = init_x.shape

    def score_eval_wrapper(sample, time_steps):
        """A wrapper of the score-based model for use by the ODE solver."""
        sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
        time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))    
        with torch.no_grad(): 
            score = score_model(sample, time_steps)
        return score.cpu().numpy().reshape((-1,)).astype(np.float64)

    def ode_func(t, x):        
        """The ODE function for use by the ODE solver.
           dx = - 0.5 * g^2(x) * s(x, t) dt
        """
        time_steps = np.ones((shape[0],)) * t
        g = diffusion_coeff(torch.tensor(t, dtype=torch.float)).cpu().numpy()
        return  -0.5 * (g**2) * score_eval_wrapper(x, time_steps)

    # Run the black-box ODE solver.
    # solving x' = dx / dt = - 0.5 * g^2(x) * s(x, t); x(0) = x_0 = init_x
    # note solving from t = 1 to t = 0
    res = integrate.solve_ivp(ode_func, (1., eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45')  
    
    print(f"Number of function evaluations: {res.nfev}")
    x = torch.tensor(res.y[:, -1], dtype=torch.float, device=device).reshape(shape)
    return x

{% endhighlight %}

<br>

The results of the above sampling methods are shown below. From left to right are: Euler-Maruyama, Predictor-Corrector and Probability flow ODE. 

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/posts/sde/em.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/posts/sde/pc.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/posts/sde/ode.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>


<br>

### 6. Likelihood Computation

We can compute the likelihood $$\log p_0(x(0))$$ using

$$\log p_0(x(0)) = \log p_T(x(T)) + \int_0^T \nabla \cdot \left[ -\frac{1}{2}\sigma^{2t} s_\theta(x, t) \right] dt$$

{% highlight python %}
# likelihood

#@title Define the likelihood function (double click to expand or collapse)

def prior_likelihood(z, sigma):
    """The likelihood of a Gaussian distribution with mean zero and 
      standard deviation sigma."""
    shape = z.shape
    N = np.prod(shape[1:])
    return -N / 2. * torch.log(2*np.pi*sigma**2) - torch.sum(z**2, dim=(1,2,3)) / (2 * sigma**2)

def ode_likelihood(x, 
                   score_model,
                   marginal_prob_std, 
                   diffusion_coeff,
                   batch_size=64, 
                   device='cuda',
                   eps=1e-5):
    """Compute the likelihood with probability flow ODE.

    Args:
    x: Input data.
    score_model: A PyTorch model representing the score-based model.
    marginal_prob_std: A function that gives the standard deviation of the 
      perturbation kernel.
    diffusion_coeff: A function that gives the diffusion coefficient of the 
      forward SDE.
    batch_size: The batch size. Equals to the leading dimension of `x`.
    device: 'cuda' for evaluation on GPUs, and 'cpu' for evaluation on CPUs.
    eps: A `float` number. The smallest time step for numerical stability.

    Returns:
    z: The latent code for `x`.
    bpd: The log-likelihoods in bits/dim.
    """

    # Draw the random Gaussian sample for Skilling-Hutchinson's estimator.
    epsilon = torch.randn_like(x)
      
    def divergence_eval(sample, time_steps, epsilon):      
        """Compute the divergence of the score-based model with Skilling-Hutchinson."""
        with torch.enable_grad():
            sample.requires_grad_(True)
            score_e = torch.sum(score_model(sample, time_steps) * epsilon)
            grad_score_e = torch.autograd.grad(score_e, sample)[0]
        return torch.sum(grad_score_e * epsilon, dim=(1, 2, 3))    

    shape = x.shape

    def score_eval_wrapper(sample, time_steps):
        """A wrapper for evaluating the score-based model for the black-box ODE solver."""
        sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
        time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))    
        with torch.no_grad():    
            score = score_model(sample, time_steps)
        return score.cpu().numpy().reshape((-1,)).astype(np.float64)

    def divergence_eval_wrapper(sample, time_steps):
        """A wrapper for evaluating the divergence of score for the black-box ODE solver."""
        with torch.no_grad():
            # Obtain x(t) by solving the probability flow ODE.
            sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
            time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))    
            # Compute likelihood.
            div = divergence_eval(sample, time_steps, epsilon)
        return div.cpu().numpy().reshape((-1,)).astype(np.float64)

    def ode_func(t, x):
        """The ODE function for the black-box solver."""
        time_steps = np.ones((shape[0],)) * t    
        sample = x[:-shape[0]]
        logp = x[-shape[0]:]
        g = diffusion_coeff(torch.tensor(t, dtype=torch.float)).cpu().numpy()
        sample_grad = -0.5 * g**2 * score_eval_wrapper(sample, time_steps) # dx / dt
        logp_grad = -0.5 * g**2 * divergence_eval_wrapper(sample, time_steps) # d logp / dt
        return np.concatenate([sample_grad, logp_grad], axis=0) # d[x, logp] / dt

    # logp_0(x(0)) = logp_T(x(T)) + int_0^T \nabla f(x(t), t)
    init = np.concatenate([x.cpu().numpy().reshape((-1,)), np.zeros((shape[0],))], axis=0)
    
    # Black-box ODE solver
    # note solving from t = 0 to t = 1, different from the reverse one
    res = integrate.solve_ivp(ode_func, (eps, 1.), init, rtol=1e-5, atol=1e-5, method='RK45')  
    
    zp = torch.tensor(res.y[:, -1], dtype=torch.float, device=device) # [x(1), logp(1)]
    z = zp[:-shape[0]].reshape(shape) # x(1)
    delta_logp = zp[-shape[0]:].reshape(shape[0]) # logp(1)
    sigma_max = marginal_prob_std(torch.tensor([1.]))
    # print(sigma_max)
    
    # compute likelihood of x(T)
    prior_logp = prior_likelihood(z, sigma_max) 
    
    # bits per dimension
    bpd = -(prior_logp + delta_logp) / np.log(2) # log_2 (x) = log(x) / log(2)
    N = np.prod(shape[1:])

    # we treat the gray scale [0, 1.] as [0., 256], so we have to add log_2(256) = 8 to each dimension
    # or according to the def: bpd = -(prior_logp + delta_logp - log(256)) / log(2)
    #                              = -(prior_logp + delta_logp) / log(2) + 8.
    bpd = bpd / N + 8. 
    
    return z, bpd

{% endhighlight %}

We can use the above to compute the likelihood as bits per dimension, the lower the better. 



{% highlight python %}
batch_size = 256

dataset = MNIST('.', train=False, transform=transforms.ToTensor(), download=True)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)


all_bpds = 0.
all_items = 0
try:
    for i, (x, _) in enumerate(data_loader):
        if i == 2: break
        x = x.to(device)
        # uniform dequantization
        x = (x * 255. + torch.rand_like(x)) / 256.    
        _, bpd = ode_likelihood(x, score_model, marginal_prob_std_fn,
                                diffusion_coeff_fn,
                                x.shape[0], device=device, eps=1e-5)
        all_bpds += bpd.sum()
        all_items += bpd.shape[0]
        print("Average bits/dim: {:5f}".format(all_bpds / all_items))

except KeyboardInterrupt: raise

# Average bits/dim: 1.710447
# Average bits/dim: 1.658721
{% endhighlight %}

<br>

### 7. References

1. Song et al, Score-Based Generative Modeling through Stochastic Differential Equations, ([link](https://openreview.net/forum?id=PxTIG12RRHS))
2. Song et al, Maximum Likelihood Training of Score-Based Diffusion Models, ([link](https://arxiv.org/abs/2101.09258))
3. Original PyTorch Implementation: [Google Colab](https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing)
4. Blog post on Score-Based SDE [link](https://yang-song.net/blog/2021/score/)


