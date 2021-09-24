**What is a neural network neuron?**

In Artificial Neural network, Neuron is basic computation unit that consist of a set of inputs, a set of weights, and an activation function. The neuron translates these inputs into a single output, which can then be picked up as input for another layer of neurons later on.

Neuron function can be represented as weighted sum:
$$
f(X1, X2, ...Xn) = 1.X1+W2.X2+W3.X3+.....+Wn.Xn
$$
Each neuron has a *weight vector* w=(w1,w2,...,wn) where n is the number of inputs to that neuron. These inputs can be either the 'raw' input features such as binary value or the output of neurons from an earlier layer.



**What is the use of the learning rate?**

 The **learning rate** is a turning parameter in an optimization algoiritham that determines the step size at each iteration while moving toward a minimum of a loss function.

Deep learning models are  typically trained by a stochastic gradient descent optimizer. There are  many variations of stochastic gradient descent: Adam, RMSProp, Adagrad,  etc. All of them let you set the learning rate. This parameter tells the optimizer how far to move the weights in the direction opposite of the  gradient for a mini-batch.

If the learning rate is low, then training is more reliable, but  optimization will take a lot of time because steps towards the minimum  of the loss function are tiny.

If the learning rate is high, then training may not converge or even  diverge. Weight changes can be so big that the optimizer overshoots the  minimum and makes the loss worse.

The training should start from a relatively large learning rate in the beginning and then the  learning rate can decrease during training to allow more fine-grained  weight updates.

There are multiple ways to select a good starting point for the learning rate. A naive approach is to try a few different values and see which  one gives you the best loss without sacrificing speed of training. We  might start with a large value like 0.1, then try exponentially lower  values: 0.01, 0.001, etc.



**How are weights initialized?**

The nodes in neural networks are composed of parameters referred to as weights used to calculate a weighted sum of the inputs.

Weight initialization is an important consideration in the design of a neural network model. Weight initialization is a procedure to set the weights of a neural network to small random values that define the starting point for the optimization (learning or training) of the neural network model.

**Zero initialization:**

One can initialize all weights equals to zero. In this case, the equations of the learning algorithm would fail to  make any changes to the network weights, and the model will be stuck. It is important to note that the bias weight in each neuron is set to zero by default, not a small random value.

**Random initialization:**

Assigning random values to weights is better than just 0 assignment.

Historically, weight initialization follows simple heuristics, such as:

- Small random values in the range [-0.3, 0.3]
- Small random values in the range [0, 1]
- Small random values in the range [-1, 1]



**What is "loss" in a neural network?**

 The lower the **loss,** the better a model (unless the model has over-fitted to the training data). The loss is calculated on **training** and **validation** and its interperation is how well the model is doing for these two  sets. Unlike accuracy, loss is not a percentage. It is a summation of  the errors made for each example in training or validation sets.

In the case of neural networks, the loss is usually negative log-likelihood and residual sum of squares for classification and regression respectively. Then naturally, the  main objective in a learning model is to reduce (minimize) the loss  function's value with respect to the model's parameters by changing the  weight vector values through different optimization methods, such as  backpropagation in neural networks.

Loss value implies how well or poorly a certain model behaves after each iteration of optimization. Ideally, one would expect the reduction of  loss after each, or several, iteration(s).



**What is the "chain rule" in gradient flow?**

Our objective is to reduce the loss or prediction error of the optimization fuction to accompolish that we use gradient of the error w.r.t weight.

The gradient of a vector can be interpreted as the "direction and  rate of the fastest increase". If the gradient of a function is non-zero at a point p, the direction of the gradient in which the function  increases most quickly from p, and the magnitude of the gradient is the  rate of increase in that direction, the greatest absolute directional  derivative.

[![\large \nabla f (x, y, z) = \left ( \dfrac{\partial f}{ \partial x} , \dfrac{\partial f}{ \partial y} , \dfrac{\partial f}{\partial z} \right ) = \dfrac{\partial f}{\partial x}\textbf{i} + \dfrac{\partial f}{ \partial y}\textbf{j} + \dfrac{\partial f}{ \partial z}\textbf{k}](https://camo.githubusercontent.com/1784750b80126d25aba879a8ce015f51da60eb5db4b148319e85860ec6950f27/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f5c62675f77686974652673706163653b5c6c617267652673706163653b5c6e61626c612673706163653b662673706163653b28782c2673706163653b792c2673706163653b7a292673706163653b3d2673706163653b5c6c6566742673706163653b282673706163653b5c64667261637b5c7061727469616c2673706163653b667d7b2673706163653b5c7061727469616c2673706163653b787d2673706163653b2c2673706163653b5c64667261637b5c7061727469616c2673706163653b667d7b2673706163653b5c7061727469616c2673706163653b797d2673706163653b2c2673706163653b5c64667261637b5c7061727469616c2673706163653b667d7b5c7061727469616c2673706163653b7a7d2673706163653b5c72696768742673706163653b292673706163653b3d2673706163653b5c64667261637b5c7061727469616c2673706163653b667d7b5c7061727469616c2673706163653b787d5c7465787462667b697d2673706163653b2b2673706163653b5c64667261637b5c7061727469616c2673706163653b667d7b2673706163653b5c7061727469616c2673706163653b797d5c7465787462667b6a7d2673706163653b2b2673706163653b5c64667261637b5c7061727469616c2673706163653b667d7b2673706163653b5c7061727469616c2673706163653b7a7d5c7465787462667b6b7d)](https://camo.githubusercontent.com/1784750b80126d25aba879a8ce015f51da60eb5db4b148319e85860ec6950f27/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f5c62675f77686974652673706163653b5c6c617267652673706163653b5c6e61626c612673706163653b662673706163653b28782c2673706163653b792c2673706163653b7a292673706163653b3d2673706163653b5c6c6566742673706163653b282673706163653b5c64667261637b5c7061727469616c2673706163653b667d7b2673706163653b5c7061727469616c2673706163653b787d2673706163653b2c2673706163653b5c64667261637b5c7061727469616c2673706163653b667d7b2673706163653b5c7061727469616c2673706163653b797d2673706163653b2c2673706163653b5c64667261637b5c7061727469616c2673706163653b667d7b5c7061727469616c2673706163653b7a7d2673706163653b5c72696768742673706163653b292673706163653b3d2673706163653b5c64667261637b5c7061727469616c2673706163653b667d7b5c7061727469616c2673706163653b787d5c7465787462667b697d2673706163653b2b2673706163653b5c64667261637b5c7061727469616c2673706163653b667d7b2673706163653b5c7061727469616c2673706163653b797d5c7465787462667b6a7d2673706163653b2b2673706163653b5c64667261637b5c7061727469616c2673706163653b667d7b2673706163653b5c7061727469616c2673706163653b7a7d5c7465787462667b6b7d)

And also you must be wondering how the partial derivatives came in, this `grad` was the reason so,

Here would be `grad L` (gradient of Loss)

[![\large \nabla L=\begin{bmatrix} \frac{\partial L}{\partial \theta_{0} } \\ \frac{\partial L}{\partial \theta_{1} } \\ ... \\ \frac{\partial L}{\partial \theta_{n-1} } \\ \end{bmatrix}](https://camo.githubusercontent.com/6362585d5a6ee6482dca358510d176d2aa7a447baf4ec2d38503c74c240b806f/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f5c62675f77686974652673706163653b5c6c617267652673706163653b5c6e61626c612673706163653b4c3d5c626567696e7b626d61747269787d2673706163653b5c667261637b5c7061727469616c2673706163653b4c7d7b5c7061727469616c2673706163653b5c74686574615f7b307d2673706163653b7d2673706163653b5c5c2673706163653b5c667261637b5c7061727469616c2673706163653b4c7d7b5c7061727469616c2673706163653b5c74686574615f7b317d2673706163653b7d2673706163653b5c5c2673706163653b2e2e2e2673706163653b5c5c2673706163653b5c667261637b5c7061727469616c2673706163653b4c7d7b5c7061727469616c2673706163653b5c74686574615f7b6e2d317d2673706163653b7d2673706163653b5c5c2673706163653b5c656e647b626d61747269787d)](https://camo.githubusercontent.com/6362585d5a6ee6482dca358510d176d2aa7a447baf4ec2d38503c74c240b806f/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f5c62675f77686974652673706163653b5c6c617267652673706163653b5c6e61626c612673706163653b4c3d5c626567696e7b626d61747269787d2673706163653b5c667261637b5c7061727469616c2673706163653b4c7d7b5c7061727469616c2673706163653b5c74686574615f7b307d2673706163653b7d2673706163653b5c5c2673706163653b5c667261637b5c7061727469616c2673706163653b4c7d7b5c7061727469616c2673706163653b5c74686574615f7b317d2673706163653b7d2673706163653b5c5c2673706163653b2e2e2e2673706163653b5c5c2673706163653b5c667261637b5c7061727469616c2673706163653b4c7d7b5c7061727469616c2673706163653b5c74686574615f7b6e2d317d2673706163653b7d2673706163653b5c5c2673706163653b5c656e647b626d61747269787d)

Now does it make all sense?  Since the `grad F` gives the direction of highest increase, we multiple this `grad F` by `-ve 1`, thus now we have the direction of highest decrease in the loss value,  and we thus move towards that steepest decreasing loss! amazing right?

[![\large W_{new} = W-\eta \nabla L](https://camo.githubusercontent.com/a9cac04b130be547ff149cf1a63142a265036b820467369f2aae44593d4fef9a/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f5c62675f77686974652673706163653b5c6c617267652673706163653b575f7b6e65777d2673706163653b3d2673706163653b572d5c6574612673706163653b5c6e61626c612673706163653b4c)](https://camo.githubusercontent.com/a9cac04b130be547ff149cf1a63142a265036b820467369f2aae44593d4fef9a/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f5c62675f77686974652673706163653b5c6c617267652673706163653b575f7b6e65777d2673706163653b3d2673706163653b572d5c6574612673706163653b5c6e61626c612673706163653b4c)

But what is this chain rule?

The problem is that [![\frac{\partial L}{\partial \theta_{0}}](https://camo.githubusercontent.com/ba5381c9e743681f508b57f0d215dd61db38eeab0a91549d78fca335a32f53c3/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f5c667261637b5c7061727469616c2673706163653b4c7d7b5c7061727469616c2673706163653b5c74686574615f7b307d7d)](https://camo.githubusercontent.com/ba5381c9e743681f508b57f0d215dd61db38eeab0a91549d78fca335a32f53c3/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f5c667261637b5c7061727469616c2673706163653b4c7d7b5c7061727469616c2673706163653b5c74686574615f7b307d7d) may not be directly calculatable.

```mathematica
w0 -> i_0 -> activation -> o_0
w0`: weight 0 `i_0`: multiply the weights with output of previous layer, this is input to this neuron `activation`: a activation function is applied to `i_0` producing `o_0
```

This is the chain rule from maths

[![img](https://github.com/satyajitghana/TSAI-DeepNLP-END2.0/raw/main/01_VeryBasics/chain_rule.png)](https://github.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/01_VeryBasics/chain_rule.png)

see how [![\frac{\partial z}{\partial s} = \frac{\partial z}{\partial x}\frac{\partial x}{\partial s}](https://camo.githubusercontent.com/9782061dc94e117007d9bbd1b517aeaa98f64be7e0b0100ba0e5d52b4ae5dd38/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f5c667261637b5c7061727469616c2673706163653b7a7d7b5c7061727469616c2673706163653b737d2673706163653b3d2673706163653b5c667261637b5c7061727469616c2673706163653b7a7d7b5c7061727469616c2673706163653b787d5c667261637b5c7061727469616c2673706163653b787d7b5c7061727469616c2673706163653b737d)](https://camo.githubusercontent.com/9782061dc94e117007d9bbd1b517aeaa98f64be7e0b0100ba0e5d52b4ae5dd38/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f5c667261637b5c7061727469616c2673706163653b7a7d7b5c7061727469616c2673706163653b737d2673706163653b3d2673706163653b5c667261637b5c7061727469616c2673706163653b7a7d7b5c7061727469616c2673706163653b787d5c667261637b5c7061727469616c2673706163653b787d7b5c7061727469616c2673706163653b737d)

Similarly,

[![\frac{\partial L}{\partial \theta_{0}} = \frac{\partial L_{0}}{\partial o_{0}}\frac{\partial o_{0}}{\partial i_{0}} \frac{\partial o_{0}}{\partial \theta_{0}}](https://camo.githubusercontent.com/fea25c1da331251d3294b63f41950142f89e9515d9c71b04dc2815226bb94467/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f5c667261637b5c7061727469616c2673706163653b4c7d7b5c7061727469616c2673706163653b5c74686574615f7b307d7d2673706163653b3d2673706163653b5c667261637b5c7061727469616c2673706163653b4c5f7b307d7d7b5c7061727469616c2673706163653b6f5f7b307d7d5c667261637b5c7061727469616c2673706163653b6f5f7b307d7d7b5c7061727469616c2673706163653b695f7b307d7d2673706163653b5c667261637b5c7061727469616c2673706163653b6f5f7b307d7d7b5c7061727469616c2673706163653b5c74686574615f7b307d7d)](https://camo.githubusercontent.com/fea25c1da331251d3294b63f41950142f89e9515d9c71b04dc2815226bb94467/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f5c667261637b5c7061727469616c2673706163653b4c7d7b5c7061727469616c2673706163653b5c74686574615f7b307d7d2673706163653b3d2673706163653b5c667261637b5c7061727469616c2673706163653b4c5f7b307d7d7b5c7061727469616c2673706163653b6f5f7b307d7d5c667261637b5c7061727469616c2673706163653b6f5f7b307d7d7b5c7061727469616c2673706163653b695f7b307d7d2673706163653b5c667261637b5c7061727469616c2673706163653b6f5f7b307d7d7b5c7061727469616c2673706163653b5c74686574615f7b307d7d)

That's the chain rule, plain and simple

