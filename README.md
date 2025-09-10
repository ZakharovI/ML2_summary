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