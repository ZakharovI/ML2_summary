- [Полезные ссылки](#полезные-ссылки)
- [\[Лекция 1\]](#лекция-1)
  - [Perceptron](#perceptron)
  - [Полносвязная сеть](#полносвязная-сеть)
  - [Функция активации](#функция-активации)

<!-- 

Ctrl+Shift+V - open preview

Ctrl+Shift+P + "Markdown All in One: Create Table of Content"

git add .
git commit --allow-empty-message -m ''
git push origin main

-->

# Полезные ссылки

* [Записи занятий](https://disk.360.yandex.ru/d/-i0quAMsXqNabw/2%20%D0%BA%D1%83%D1%80%D1%81%20%D0%9C%D0%9D%D0%90%D0%94-24/%D0%9C%D0%B0%D1%88%D0%B8%D0%BD%D0%BD%D0%BE%D0%B5%20%D0%BE%D0%B1%D1%83%D1%87%D0%B5%D0%BD%D0%B8%D0%B5%202?clckid=a86ec8e9)


# [Лекция 1] 

* [Запись 1 лекции](https://disk.360.yandex.ru/d/-i0quAMsXqNabw/2%20%D0%BA%D1%83%D1%80%D1%81%20%D0%9C%D0%9D%D0%90%D0%94-24/%D0%9C%D0%B0%D1%88%D0%B8%D0%BD%D0%BD%D0%BE%D0%B5%20%D0%BE%D0%B1%D1%83%D1%87%D0%B5%D0%BD%D0%B8%D0%B5%202/1%20%D0%BF%D0%BE%D1%82%D0%BE%D0%BA/1.%2006.09.25%20%D0%9C%D0%B0%D1%88%D0%B8%D0%BD%D0%BD%D0%BE%D0%B5%20%D0%BE%D0%B1%D1%83%D1%87%D0%B5%D0%BD%D0%B8%D0%B5%202_%D0%9B%D0%B5%D0%BA%D1%86%D0%B8%D1%8F_1%20%D0%BF%D0%BE%D1%82%D0%BE%D0%BA.mp4?clckid=a86ec8e9)

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

## Полносвязная сеть

"Multilayer Perceptron" или "Fully Connected Neural Network" - объединяем перцептроны в полносвязную сеть. Выход каждого перцептрона на одном слое подается в качетсве входа в каждый перцептрон на следующем слое.

<p align="center">
<img src="./src/img/01_lect/fcnn.png"  style="width: 50%">
</p>

Рассмотрим самую простую "нейросеть" с одним перцептроном:

* $x\in\mathbb{R}^n$ - фичи из датасета
* $y\in\mathbb{R}$ - target
* $\theta\in\Theta$ (или $w\in W$) - параметры нейросети
* $f_{\theta}(x) = x^\intercal\times\theta = x_1\theta_1 + x_2\theta_2 + ... + x_n\theta_n$ - нейросеть из одного перцептрона
* $\{(x_i,y_i)\}$ - training set
* $\mathcal{L}(\hat{y}, y)$ - loss function

Надо минимизировать "эмпирический риск" $\quad\frac{1}{n}\sum_{i=1}^{n} \mathcal{L}(\hat{y_i}, y_i)$ \
Для этого можно использовать градиентный спуск (для этого $f$ и $\mathcal{L}$ должны быть дифференцируемы)

Чтобы понять, как считать градиент $\nabla(\mathcal{L})$, воспользуемся свойством $y=f(g(x)) \quad\Rightarrow\quad \frac{dy}{dx} = \frac{df}{dg}\cdot\frac{dg}{dx}$

Тогда

$$y=f(g_1(x), g_2(x), ..., g_n(x)) \quad\Rightarrow\quad \frac{dy}{dx} = \sum_{i=1}^{n}\frac{df}{dg_i}\cdot\frac{dg_i}{dx}$$

Теперь перейдем к случаю c полносвязной нейросетью, в которой $m$ слоев

* $x\in\mathbb{R}^n$
* $\{z_i\}_{i=1}^{m}$ - "logit" или вектор выходов для каждого из слоев
  * $z_1 = x \times \theta_1$
  * $z_2 = z_1 \times \theta_2$
  * ...
  * $z_m = z_{m-1} \times \theta_m$
  
Пока все было достаточно линейно, а мы хотим привнести какую-нибуь нелинейность в нашу модель

## Функция активации

Существует много разных функций активаций. Самые распространенные:

**Sigmoid** \
$\sigma(x) = \frac{1}{1+e^{-x}}\quad\quad\sigma'=\sigma(x)(1-\sigma(x))$

**tanh** \
$\tanh(x) = \frac{e^x-e^{-x}}{e^x+e^{-x}}\quad\quad \tanh'(x)=1-\tanh^2(x)$

**ReLU** \
$ReLU(x) = max(0, x)$

**Leaky ReLU** \
$LeakyReLU(x) = max(\alpha x, x)$

<p align="center">
<img src="./src/img/01_lect/activation_functions.png"  style="width: 80%">
</p>

Пусть $\sigma(.)$ - "функция активации" ($\sigma:\mathbb{R}\mapsto\mathbb{R}$), тогда мы можем переопределить $z_i$:
* $z_i=\sigma(z_{i-1}\theta_i+b_i)$

На самом деле, иногда активацию выносят на отдельные слои, а в некоторых моделях может быть несколько линейных слоев подряд без активации. В pytorch слои с активацией надо указывать отдельно. 

Простейшие примеры нейросетей:

<p align="center">
<img src="./src/img/01_lect/simple_nn.png"  style="width: 50%">
</p>

Здесь `Linear (x, y)` означает, что мы $x$ признаков подаем на вход $y$ перцептронам (нейронам), то есть в следующий слой каждому нейрону на вход подастся $y$ признаков.