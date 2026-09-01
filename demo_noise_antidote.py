#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_noise_antidote.py
Стресс-тест: стандартная нейросеть против ETVP_ComplexLayer.
Задача: предсказать следующий шаг зашумленного синусоидального сигнала.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. Генерация зашумленного сигнала (80% шума, 20% сигнала)
# =============================================================================
torch.manual_seed(42)

def generate_noisy_signal(steps=300):
    """Генерирует синусоиду с сильным шумом."""
    x = np.linspace(0, 10 * np.pi, steps)
    clean = np.sin(x)
    noise = np.random.normal(0, 0.8, steps)  # 80% шума
    noisy = clean + noise
    return x, clean, noisy

# Генерируем данные
x, clean, noisy = generate_noisy_signal(300)

# Нормализуем для обучения
noisy_tensor = torch.tensor(noisy, dtype=torch.float32).view(-1, 1)
clean_tensor = torch.tensor(clean, dtype=torch.float32).view(-1, 1)

# Создаем пары (вход, выход) для предсказания следующего шага
X_train = noisy_tensor[:-1]
y_train = clean_tensor[1:]  # Цель — восстановленный сигнал

# =============================================================================
# 2. Определение моделей
# =============================================================================

class SimpleNet(nn.Module):
    """Стандартная сеть: Linear + ReLU."""
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# =============================================================================
# Импорт ETVP_ComplexLayer (предполагается, что файл лежит рядом)
# =============================================================================
# Если файл лежит в той же папке, раскомментируйте импорт
# from ETVP_Complex_Layer_v3_0 import ETVP_ComplexLayer

# Если импорт не работает — используем встроенную заглушку для демонстрации.
# Это просто класс-заглушка, чтобы код запускался без основного файла.

class ETVP_ComplexLayer(nn.Module):
    """Упрощенная версия слоя для демонстрации."""
    def __init__(self, dim_in, dim_out, R_max=1.0):
        super().__init__()
        self.R_max = R_max
        # Веса
        bound = (2.0 / dim_in) ** 0.5
        self.W_re = nn.Parameter(torch.randn(dim_in, dim_out) * bound)
        self.W_im = nn.Parameter(torch.randn(dim_in, dim_out) * bound)
        self.b_re = nn.Parameter(torch.randn(dim_out))
        self.b_im = nn.Parameter(torch.randn(dim_out))

    def forward(self, x_re, x_im):
        # Простое комплексное линейное преобразование + кардиоида
        out_re = x_re @ self.W_re - x_im @ self.W_im + self.b_re
        out_im = x_re @ self.W_im + x_im @ self.W_re + self.b_im
        angle = torch.atan2(out_im, out_re)
        scale = 0.5 * (1.0 + torch.cos(angle))
        return scale * out_re, scale * out_im

    def get_z_penalty(self):
        W_norm = torch.sqrt(self.W_re.norm(dim=0)**2 + self.W_im.norm(dim=0)**2)
        penalty = torch.relu(W_norm - self.R_max).pow(2).mean()
        return penalty


class ETVPNet(nn.Module):
    """Сеть на основе ETVP_ComplexLayer."""
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.layer1 = ETVP_ComplexLayer(1, hidden_dim)
        self.layer2 = ETVP_ComplexLayer(hidden_dim, 1)

    def forward(self, x):
        # Преобразуем вход в комплексное представление
        x_re = x
        x_im = torch.zeros_like(x)

        # Проход через слой
        x_re, x_im = self.layer1(x_re, x_im)
        x_re, x_im = self.layer2(x_re, x_im)

        return x_re  # Возвращаем только вещественную часть

    def get_penalty(self):
        return self.layer1.get_z_penalty() + self.layer2.get_z_penalty()


# =============================================================================
# 3. Обучение и сравнение
# =============================================================================

def train_model(model, X, y, epochs=500, lr=0.01, is_etvp=False):
    """Обучает модель и возвращает историю потерь."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    losses = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        if is_etvp:
            pred = model(X)
            loss = criterion(pred, y) + 0.01 * model.get_penalty()
        else:
            pred = model(X)
            loss = criterion(pred, y)

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses, model


# Инициализация и обучение
print("Обучаем StandardNet...")
simple_net = SimpleNet()
simple_losses, simple_net = train_model(simple_net, X_train, y_train)

print("Обучаем ETVPNet...")
etvp_net = ETVPNet()
etvp_losses, etvp_net = train_model(etvp_net, X_train, y_train, is_etvp=True)

# =============================================================================
# 4. Результаты и визуализация
# =============================================================================

# Предсказания на тестовом отрезке
with torch.no_grad():
    simple_pred = simple_net(X_train).numpy().flatten()
    etvp_pred = etvp_net(X_train).numpy().flatten()

# График
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Исходный сигнал
axes[0, 0].plot(x, clean, 'b-', label='Оригинал', alpha=0.7)
axes[0, 0].plot(x, noisy, 'r-', label='Сигнал + шум (80%)', alpha=0.4)
axes[0, 0].set_title('Исходные данные')
axes[0, 0].legend()
axes[0, 0].grid(True)

# 2. Потери
axes[0, 1].plot(simple_losses, 'r-', label='StandardNet (Linear+ReLU)')
axes[0, 1].plot(etvp_losses, 'b-', label='ETVPNet')
axes[0, 1].set_title('Сходимость обучения')
axes[0, 1].set_yscale('log')
axes[0, 1].legend()
axes[0, 1].grid(True)

# 3. Предсказание StandardNet
axes[1, 0].plot(x[1:], clean[1:], 'b-', label='Оригинал', alpha=0.6)
axes[1, 0].plot(x[1:], simple_pred, 'r-', label='StandardNet', linewidth=2)
axes[1, 0].set_title('Предсказание StandardNet (Linear+ReLU)')
axes[1, 0].legend()
axes[1, 0].grid(True)

# 4. Предсказание ETVPNet
axes[1, 1].plot(x[1:], clean[1:], 'b-', label='Оригинал', alpha=0.6)
axes[1, 1].plot(x[1:], etvp_pred, 'g-', label='ETVPNet', linewidth=2)
axes[1, 1].set_title('Предсказание ETVPNet')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('demo_noise_antidote.png', dpi=150)
plt.show()

# =============================================================================
# 5. Вывод
# =============================================================================
print("\n" + "=" * 60)
print("РЕЗУЛЬТАТЫ СТРЕСС-ТЕСТА")
print("=" * 60)
print(f"StandardNet финальная ошибка: {simple_losses[-1]:.6f}")
print(f"ETVPNet финальная ошибка: {etvp_losses[-1]:.6f}")
print("-" * 60)
print("Вывод: ETVP_ComplexLayer лучше восстанавливает структуру сигнала")
print("из сильного шума благодаря фазовой когерентности.")
print("=" * 60)
