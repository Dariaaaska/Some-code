# Линейная алгебра на Python

Допустим, нам нужно запрограммировать базовые операции с объектами линейной алгебры. Самые полезные библиотеки для этой работы: numpy, scipy.linalg и import matplotlib

Для начала, добавим все нужные библиотеки:

```python
import numpy as np

import scipy.linalg as sla

import matplotlib.pyplot as plt
%matplotlib inline
```

Попробуем создать самый простой объект - нулевую матрицу $Z$ размера $3 \times 4$

```python
Z = np.zeros((3,4), dtype = int)
print(Z)
```

Теперь диагональную создадим матрицу $5 \times 5$, у которой на главной диагонали будут расположены цифры 1-5 в порядке возрастания

```python
diag_arr = np.diag(np.full(5, [1, 2, 3, 4, 5]))
print(diag_arr)
```

Найдём след последней матрицы:

```python
diag_arr = np.diag(np.full(5, [1, 2, 3, 4, 5]))
trace = np.trace(diag_arr)
print(trace)
```

И обратную к данной матрицу:

```python
diag_arr = np.diag(np.full(5, [1, 2, 3, 4, 5]))
inv_arr = np.linalg.inv(diag_arr)
print(inv_arr)
```

Но что, если нам нужна матрица со случайно сгенерированными значениями? Используем `np.random.rand`

```python
X = np.random.rand(4, 5)
print(X)
```

Иногда требуется работать не с целой матрицей, а с каким-то $k \times k$ блоком - то есть **подматрицей**. Найдём определитель такой подматрицы нашей предыдущей матрицы X, расположенной на пересечении 2-ой и 3-ей строк и 1-го и 2-го столбцов (фактически мы ищем минор матрицы Х)

```python
new_X = X[1:3, 0:2]
new_det = sla.det(new_X)
print(new_det)
```

Ещё одна важная операция над матрицами - это транспонирование. Посмотрим, как мы можем её легко закодировать:

```python
X_T = np.reshape(X, (5, 4))
ans = X_T.dot(X)
print(ans)
```

Перейдём к более сложным программам. Один из способов нахождения определителя матрицы - через элементарные преобразования (метод Гаусса). Напишем функцию, которая будет вычислять определитель таким способом и выкидывать ValueError в случаях, когда матрица не является квадратной

```python
def my_det(X):
    count = 1
    X = X.astype("float128")
    n = X.shape[0]
    m = X.shape[1]
    if n != m:
        raise ValueError("Matrix is not square")
    for i in range(n):
        if X[i, i] == 0:
            for j in range(i + 1, n):
                if X[j, i] != 0:
                    X_new = X[i, :].copy()
                    X[i, :] = X[j, :]
                    X[j, :] = X_new
                    count = count * (-1)
                    break
        for u in range(i + 1, n):
            if X[u, i] != 0:
                numb_koeff = X[u, i] / X[i, i]
                X[u, :] -= X[i, :] * numb_koeff
    det = count * np.prod(np.diag(X))
    return det
```

Теперь поработаем с перемножением матриц. Сделаем это двумя способами: через целые числа и через вещественные

```python
A = np.array([[1, 0], [10 ** 20, 1]])
B = np.array([[10 ** (-20), 1],[0, 1 - 10 ** 20]])
first_var = A.dot(B)
print(first_var)
C = np.array([[1., 0.], [10. ** 20, 1.]])
D = np.array([[10. ** (-20), 1.],[0., 1. - 10. ** 20]])
second_var = C.dot(D)
print(second_var)
```

*Небольшой комментарий: лучше производить вычисления с данными типа int, так как точность обработки данных этого типа выше, чем типа float*

Теперь рассмотрим, как можно кодировать одни элементы матриц на основании других. Ярким примером будет служить матрица Паскаля, где $P_{ij} = C_{i+j}^{i}$

Итак, напишем функцию, генерирующую *матрицу Паскаля* заданной размерности *n*, то есть матрицу $P$, в которой $P_{ij} = C_{i+j}^{i}$

```python
def my_pascal(dim):
    P = np.zeros((dim, dim), dtype = int)
    for i in range(dim):
        summa = 1
        for j in range(dim):
            if i == 0 or j == 0:
                P[i][j] = 1
            else:
                summa = summa + P[i - 1][j]
                P[i][j] = summa

    return P


n = int(input())
print(my_pascal(n))
```

Перейдём к матричным вычислениям.

Напишем функцию prod_and_sq_sum(A), вычисляющую произведение диагональных элементов, а также сумму квадратов диагональных элементов квадратной матрицы А

```python
def prod_and_sq_sum(A):
    elem = np.diagonal(A)
    el = pow(np.diagonal(A), 2)
    prod = elem.prod()
    summa = el.sum()
    return prod, summa


A = np.random.randn(5, 5)
print(f'A =\n {A}')
print()
print(f'Result: {prod_and_sq_sum(A)}')
```

Теперь возьмём две матрицы: A и B размера $m \times n$. Обозначим за $a_1, ..., a_n$ и $b_1, ..., b_n$ их столбцы (соответственно для А и В). Пусть $λ_1, ..., λ_n$ - некоторые числа. Напишем функцию $f(A, B, lmbd, k)$, вычисляющую
$\sum_{i=1}^{min(k, n)} λ_ia_ib_i^T$

```python
def f(A, B, lmbd, k):
    m_n = min(k, n)
    a_i = A[:, :m_n]
    b_i = B[:, :m_n]
    lmbd_a_i = a_i * lmbd[:m_n].reshape(1, m_n)
    res = lmbd_a_i.dot(b_i.T)
    return res


n = np.random.randint(2, 7)
m = np.random.randint(2, 7)
A = np.random.randn(m, n)
B = np.random.randn(m, n)
lmbd = np.random.randn(n)
k = np.random.randint(1, n)

print(f'n = {n}')
print(f'm = {m}')
print(f'A =\n {A}')
print(f'B =\n {B}')
print()
print(f'lmbd = {lmbd}')
print(f'k = {k}')
print()
print(f'Result:\n {f(A, B, lmbd, k)}')
```

Снова возьмём матрицы A и B, заполненные случайными числами. Иногда нужно посчитать вектор диагональных элементов произведения $A \times B$. Конечно, можно сначала вычислить результат перемножения матриц, но мы можем решить эту задачу короче и получить необходимый вектор, не потратив время на вычисление произведения исходных матриц:

```python
n = np.random.randint(2, 7)
A = np.random.randn(n, n)
B = np.random.randn(n, n)
def get_diag(A, B):
    return np.sum(A * B.T, axis=1)

print(f'A =\n {A}')
print()
print(f'B =\n {B}')
print()
print(f"Result:\n{get_diag(A, B)}")
```

Затронем геометрию и поле $\mathbb{C}$ комплексных чисел

Напишем функцию `shrink_rotate`, которая принимает на вход:
- заданную в виде комплексного числа точку $X$, которую мы подвергаем преобразованию,
- заданную в виде комплексного числа точку $A$,
- действительный коэффициент `coef`,
- угол `alpha`, заданный в радианах

и осуществляет следующее преобразование: мы берём вектор $AX$, умножаем его на `coef`, поворачиваем вокруг точки $A$ на угол `alpha` против часовой стрелки, после чего возвращаем конец полученного вектора

```python
def shrink_rotate(X, A, coef=1., alpha=0.):
    vect_AX = X - A
    coef_vect_AX = vect_AX * coef
    y = A + coef_vect_AX
    circ = np.exp(alpha * 1j)
    rotate = (circ * coef_vect_AX) + A
    return rotate
    raise NotImplementedError()
```

Немного поменяем предыдущий код и напишем функцию `shrink_rotate_conj`, которая сначала делает то же самое, что и `shrink_rotate`, а после этого отражает вектор $AY$ относительно горизонтальной прямой, проходящей через точку $A$, и возвращает точку $Y^`$

```python
def shrink_rotate_conj(X, A, coef=1., alpha=0.):
    vect_AX = X - A
    coef_vect_AX = vect_AX * coef
    y = A + coef_vect_AX
    circ = np.exp(alpha * 1j)
    rotate = (circ * coef_vect_AX) + A
    Y = rotate
    Y_ = (Y - A) * np.conj(Y - A)
    return Y_ + A
    raise NotImplementedError()
```

Одно из многочисленных преимуществ Python - его огромный функционал. Попробуем использовать его графический функционал для построения множеств точек в виде графика

```python
z = 0.5 + 0.*1j
max_iter = 100000
screen = np.zeros((1000, 1000))
a, b = -1, -1
c, d = 1, 1
r_x = r_y = 1000/ 2 #screen/ (c - a) = screen/ (d - b) = screen/ 2

funcs = [
    (lambda t: shrink_rotate(t, 0. + 1.*1j, coef=0.5, alpha=0.)),
    (lambda t: shrink_rotate(t, 1. + 0.*1j, coef=0.5, alpha=0.)),
    (lambda t: shrink_rotate(t, -1. + 0.*1j, coef=0.5, alpha=0.))
]

for n_iter in range(max_iter):
    n_func = np.random.choice(len(funcs))
    z = funcs[n_func](z)
    if n_iter >= 10:
      screen[int((z.imag * r_y) + r_y), int((z.real * r_x) + r_x)] = 1
plt.figure(figsize=(20, 20))
plt.imshow(screen, cmap='gist_gray')
```

С помощью этого кода мы получили знаменитый фрактал - треугольник Серпинского. Каждый новый вектор строится по правилу треугольника (правило сложения векторов), геометрически это и выглядит как треугольник. Итерация повторяется множество раз, а вектор каждый раз уменьшается, так как мы имеем коэффициент меньше единицы. А значит с каждой итерацией треугольники уменьшаются, что и образует получившуюся фигуру

И напоследок, напишем ещё один код, который сгенерирует нам изображение фрактала, но уже другого - множества Мандельброта

```python
def matrix(xmin, xmax, ymin, ymax, pixel_density):
    re = np.linspace(xmin, xmax, int((xmax - xmin) * pixel_density))
    im = np.linspace(ymin, ymax, int((ymax - ymin) * pixel_density))
    return re[np.newaxis, :] + im[:, np.newaxis] * 1j


def first_thing(c, num_iterations):
    z = 0
    for _ in range(num_iterations):
        z = z ** 2 + c
    return abs(z) <= 2


def second_thing(c, num_iterations):
    mask = first_thing(c, num_iterations)
    return c[mask]


c = matrix(-2, 0.5, -1.5, 1.5, pixel_density=210)
members = second_thing(c, num_iterations=20)

plt.scatter(members.real, members.imag, color="black", marker=",", s=1)
plt.gca().set_aspect("equal")
plt.axis("off")
plt.tight_layout()
plt.show()
```
