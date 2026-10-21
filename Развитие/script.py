#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_noise_antidote_real.py
Честный стресс-тест: настоящий ETVP_ComplexLayer v3.0 против StandardNet (Linear+ReLU).
Задача: восстановить чистый сигнал из сильно зашумленного (80% шума).
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math

# =============================================================================
# ИМПОРТ НАСТОЯЩЕГО СЛОЯ
# =============================================================================
# Убедись, что файл 'ETVP Complex Layer v3.0.py' лежит в той же папке.
# Если в имени файла есть пробелы, Python может ругаться. 
# Лучше переименуй его в 'etvp_layer.py' перед запуском, или используй importlib.
try:
    from etvp_layer import ETVP_ComplexLayer
    print("✅ Настоящий ETVP_ComplexLayer успешно импортирован.")
except ImportError:
    print("❌ Ошибка: Не найден файл etvp_layer.py.")
    print("Переименуй 'ETVP Complex Layer v3.0.py' в 'etvp_layer.py' и положи рядом.")
    exit()

# =============================================================================
# 1. Генерация зашумленного сигнала
# =============================================================================
torch.manual_seed(42)
np.random.seed(42)

def generate_noisy_signal(steps=300):
    x = np.linspace(0, 10 * np.pi, steps)
    clean = np.sin(x)
    noise = np.random.normal(0, 0.8, steps) 
    noisy = clean + noise
    return x, clean, noisy

x, clean, noisy = generate_noisy_signal(300)

# Подготовка данных
# ВАЖНО: Мы меняем форму тензора для ETVP слоя.
# Было: [300, 1]. Стало: [1, 1, 300, 1] (Batch, Channels, H, W)
# Это нужно, чтобы внутри слоя сработал spatial_laplacian.
X_train_np = noisy[:-1].reshape(1, 1, -1, 1) 
y_train_np = clean[1:].reshape(1, 1, -1, 1)

X_train = torch.tensor(X_train_np, dtype=torch.float32)
y_train = torch.tensor(y_train_np, dtype=torch.float32)

# Для StandardNet оставим обычный 2D формат [Batch, Features]
X_train_std = torch.tensor(noisy[:-1].reshape(-1, 1), dtype=torch.float32)
y_train_std = torch.tensor(clean[1:].reshape(-1, 1), dtype=torch.float32)

# =============================================================================
# 2. Определение моделей
# =============================================================================

class StandardNet(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class ETVPNet(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        # Настоящий слой из v3.0, адаптированный под 4D (H=300, W=1)
        # Используем laplace_mode='spatial' принудительно.
        self.layer1 = ETVP_ComplexLayer(
            dim_in=1, 
            dim_out=hidden_dim, 
            laplace_mode='spatial',
            mode='increment'
        )
        self.layer2 = ETVP_ComplexLayer(
            dim_in=hidden_dim, 
            dim_out=1, 
            laplace_mode='spatial',
            mode='increment'
        )

    def forward(self, x_re):
        x_im = torch.zeros_like(x_re)
        x_re, x_im = self.layer1(x_re, x_im)
        x_re, x_im = self.layer2(x_re, x_im)
        return x_re

    def get_penalty(self):
        return self.layer1.get_z_penalty() + self.layer2.get_z_penalty()

# =============================================================================
# 3. Обучение
# =============================================================================

def train_model(model, X, y, epochs=500, lr=0.01, is_etvp=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    losses = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X)
        
        # Если выход 4D, усредняем или сжимаем для лосса
        if is_etvp and pred.dim() == 4:
            pred_loss = pred.squeeze(0).squeeze(1).squeeze(-1) 
            y_loss = y.squeeze(0).squeeze(1).squeeze(-1)
        else:
            pred_loss = pred
            y_loss = y

        loss = criterion(pred_loss, y_loss)
        
        if is_etvp:
            loss += 0.01 * model.get_penalty()

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses, model

print("Обучаем StandardNet...")
simple_net = StandardNet()
simple_losses, simple_net = train_model(simple_net, X_train_std, y_train_std)

print("Обучаем ETVPNet (настоящий слой)...")
etvp_net = ETVPNet()
etvp_losses, etvp_net = train_model(etvp_net, X_train, y_train, is_etvp=True)

# =============================================================================
# 4. Результаты
# =============================================================================

with torch.no_grad():
    simple_pred = simple_net(X_train_std).numpy().flatten()
    etvp_pred_raw = etvp_net(X_train)
    # Сжимаем выход обратно в 1D для графика
    etvp_pred = etvp_pred_raw.squeeze(0).squeeze(1).squeeze(-1).numpy()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(x, clean, 'b-', label='Оригинал', alpha=0.7)
axes[0, 0].plot(x, noisy, 'r-', label='Сигнал + шум (80%)', alpha=0.4)
axes[0, 0].set_title('Исходные данные')
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].plot(simple_losses, 'r-', label='StandardNet')
axes[0, 1].plot(etvp_losses, 'b-', label='ETVPNet (Real Layer)')
axes[0, 1].set_title('Сходимость обучения')
axes[0, 1].set_yscale('log')
axes[0, 1].legend()
axes[0, 1].grid(True)

axes[1, 0].plot(x[1:], clean[1:], 'b-', label='Оригинал', alpha=0.6)
axes[1, 0].plot(x[1:], simple_pred, 'r-', label='StandardNet', linewidth=2)
axes[1, 0].set_title('Предсказание StandardNet')
axes[1, 0].legend()
axes[1, 0].grid(True)

axes[1, 1].plot(x[1:], clean[1:], 'b-', label='Оригинал', alpha=0.6)
axes[1, 1].plot(x[1:], etvp_pred, 'g-', label='ETVPNet (Real)', linewidth=2)
axes[1, 1].set_title('Предсказание ETVPNet (Real Layer)')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('demo_noise_antidote_real.png', dpi=150)
plt.show()

print("\n" + "=" * 60)
print(f"StandardNet финальная ошибка: {simple_losses[-1]:.6f}")
print(f"ETVPNet финальная ошибка:    {etvp_losses[-1]:.6f}")
print("=" * 60)
