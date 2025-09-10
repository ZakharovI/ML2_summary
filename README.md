- [\[Лекция 1\]](#лекция-1)
  - [Perceptron](#perceptron)

<!-- 

Ctrl+Shift+V - open preview

Ctrl+Shift+P + "Markdown All in One: Create Table of Content"

git add .
git commit --allow-empty-message -m ''
git push origin main

-->

# [Лекция 1] 
## Perceptron

<p align="center">
<img src="./src/img/01_lect/perceptron.png"  style="width: 50%">
</p>

Принимает $n$ чисел на вход $x_1$, $x_2$, ..., $x_n$ и возвращает одно число $y$.

$y = f(Z(x_1, x_2, ..., x_n))$

$f$ - "функция активации"

$Z$ - линейная функция вида $x_1\omega_1 + x_2\omega_2 + ... + x_n\omega_n + b$

Для удобства можно вынети коэффициент $b$ из $Z$ и добавить еще один параметр на вход $x_0=1$

Все это нужно, например, для задач бинарной классификации, когда классы друг от друга линейной функцией мы разделить не можем.

<p align="center">
<img src="./src/img/01_lect/mlp.png"  style="width: 50%">
</p>

"Multilayer Perceptron" или "Fully Connected Neural Network" - объединяем перцептроны в полносвязную сеть. Выход каждого перцептрона на одном слое подается в качетсве входа в каждый перцептрон на следующем слое.

<p align="center">
<img src="./src/img/01_lect/fcnn.png"  style="width: 50%">
</p>

Рассмотрим самую простую нейросеть с одним перцептроном:

* $x\in\mathbb{R}^n$ - фичи из датасета
* $y\in\mathbb{R}$ - target
* $\theta\in\Theta$ (или $w\in W$) - параметры нейросети
* $f_{\theta}(x) = \overline{x}\cdot\theta = x_1\theta_1 + x_2\theta_2 + ... + x_n\theta_n$ - нейросеть из одного перцептрона
* $\{(x_i,y_i)\}$ - training set
* $\mathcal{L}(\hat{y}, y)$ - loss function

Надо минимизировать "эмпирический риск" $\frac{1}{n}\sum_{i=1}^{n} \mathcal{L}(\hat{y_i}, y_i)$ \
Для этого можно использовать градиентный спуск, но сначала надо понять, как считать градиент

Воспользуемся свойством $y=f(g(x)) \Rightarrow \frac{dy}{dx} = \frac{df}{dg}\cdot\frac{dg}{dx}$

Тогда

$$y=f(g_1(x), g_2(x), ..., g_n(x))\;\;\Rightarrow\;\;\frac{dy}{dx} = \sum_{i=1}^{n}\frac{df}{dg_i}\cdot\frac{dg_i}{dx}$$