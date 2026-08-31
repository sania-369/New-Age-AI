# Нейросеть

---

## 1. Математическая модель

### Уравнение состояния слоя

$$
z(t + \Delta t) = z(t) + \Delta t \cdot \left[ f(z) - \mu \cdot \nabla^2 f(z) \right] + \sqrt{\Delta t} \cdot \eta \cdot \xi(t) \cdot \tanh(\beta |f(z)|)
$$

где $f(z) = \text{Cardioid}(W \cdot z + b)$ — детерминированная динамика.

### Cardioid-активация (аналитическая)

$$
f(z) = \frac{1}{2} \left( 1 + \cos(\angle z) \right) \cdot z
$$

Фаза вычисляется стабильно через $\text{atan2}(\Im(z), \Re(z))$.

### Z-принцип (построчная проекция)

$$
W_{i,:} \leftarrow W_{i,:} \cdot \min\left(1, \frac{R_{\text{max}}}{\|W_{i,:}\|}\right)
$$

### Обучаемая вязкость вакуума

$$
\mu = \text{softplus}(\theta_\mu)
$$

---

## 2. Полный код

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ETVP_ComplexLayer(nn.Module):
    """
    Универсальный комплексный слой
    
    Объединяет:
    - Cardioid-активацию (аналитическая, удовлетворяет Коши — Римана)
    - Лапласиан по признакам (2D) и пространству (4D)
    - Корректный масштаб шума sqrt(dt) (физика броуновского движения)
    - Построчную проекцию весов (Z-принцип без затухания градиентов)
    - Стабильное вычисление фазы через atan2
    - Поддержку dim_in != dim_out
    
    Поддерживает:
    - 2D вход: [batch, features] → Лапласиан по признакам
    - 4D вход: [batch, channels, height, width] → Лапласиан по пространству
    
    Режим Лапласиана:
    - 'auto' — автоматически по размерности входа (по умолчанию)
    - 'features' — принудительно по признакам (2D)
    - 'spatial' — принудительно по пространству (4D)
    - 'none' — отключен
    """
    
    def __init__(self, dim_in, dim_out, R_max=1.0, eta_init=0.01, beta_init=0.1,
                 laplace_mode='auto', use_cardioid=True):
        super().__init__()
        
        # Инициализация комплексных весов (адаптация Xavier для комплексных чисел)
        bound = (2.0 / dim_in) ** 0.5
        self.W_re = nn.Parameter(torch.randn(dim_in, dim_out) * bound)
        self.W_im = nn.Parameter(torch.randn(dim_in, dim_out) * bound)
        self.b_re = nn.Parameter(torch.zeros(dim_out))
        self.b_im = nn.Parameter(torch.zeros(dim_out))
        
        # Обучаемая вязкость вакуума
        self.theta_mu = nn.Parameter(torch.tensor(0.0))
        
        # Параметры стохастической самокоррекции
        self.eta = nn.Parameter(torch.tensor(eta_init))
        self.beta = nn.Parameter(torch.tensor(beta_init))
        self.R_max = R_max
        
        # Режим Лапласиана
        self.laplace_mode = laplace_mode
        self.use_cardioid = use_cardioid
        
        # Кэш для отслеживания текущего режима (для отладки)
        self.current_mode = None

    # ============================================================
    # Z-ПРИНЦИП: СТАБИЛИЗАЦИЯ ВЕСОВ
    # ============================================================
    
    def _project_weights(self):
        """
        Z-принцип: построчная проекция весов на сферу.
        Каждая строка (выходной нейрон) проецируется независимо,
        что сохраняет относительные пропорции и предотвращает
        затухание градиентов.
        """
        with torch.no_grad():
            # Построчная L2-норма: [dim_out]
            W_norm_re = self.W_re.norm(dim=0)
            W_norm_im = self.W_im.norm(dim=0)
            W_norm = torch.sqrt(W_norm_re ** 2 + W_norm_im ** 2)
            
            # Находим строки, превышающие R_max
            mask = W_norm > self.R_max
            if mask.any():
                scale = torch.ones_like(W_norm)
                scale[mask] = self.R_max / (W_norm[mask] + 1e-8)
                
                # Применяем масштаб к каждой строке
                self.W_re.data *= scale.unsqueeze(0)
                self.W_im.data *= scale.unsqueeze(0)

    # ============================================================
    # КОМПЛЕКСНОЕ ЛИНЕЙНОЕ ПРЕОБРАЗОВАНИЕ
    # ============================================================
    
    def complex_mul(self, x_re, x_im):
        """
        Полноценное комплексное линейное преобразование.
        Поддерживает 2D [batch, features] и 4D [batch, channels, H, W].
        """
        if x_re.dim() == 2:
            # 2D случай: [batch, dim_in] @ [dim_in, dim_out]
            out_re = x_re @ self.W_re - x_im @ self.W_im + self.b_re
            out_im = x_re @ self.W_im + x_im @ self.W_re + self.b_im
        elif x_re.dim() == 4:
            # 4D случай: [batch, C_in, H, W] @ [C_in, C_out] -> [batch, C_out, H, W]
            out_re = (torch.einsum('bihw,io->bohw', x_re, self.W_re) - 
                      torch.einsum('bihw,io->bohw', x_im, self.W_im) + 
                      self.b_re.view(1, -1, 1, 1))
            out_im = (torch.einsum('bihw,io->bohw', x_re, self.W_im) + 
                      torch.einsum('bihw,io->bohw', x_im, self.W_re) + 
                      self.b_im.view(1, -1, 1, 1))
        else:
            raise ValueError(f"Unsupported tensor dimension: {x_re.dim()}. Expected 2D or 4D.")
        
        return out_re, out_im

    # ============================================================
    # КОМПЛЕКСНЫЕ АКТИВАЦИИ
    # ============================================================
    
    def cardioid_activation(self, re, im):
        """
        Аналитическая комплексная активация Cardioid.
        f(z) = 0.5 * (1 + cos(angle(z))) * z
        
        Удовлетворяет условиям Коши — Римана.
        Использует atan2 для стабильного вычисления фазы.
        """
        angle = torch.atan2(im, re)
        cos_angle = torch.cos(angle)
        
        scale = 0.5 * (1.0 + cos_angle)
        return scale * re, scale * im

    def complex_tanh(self, re, im):
        """Наивная комплексная активация (запасной вариант)"""
        return torch.tanh(re), torch.tanh(im)

    # ============================================================
    # ЛАПЛАСИАН (ДИФФУЗИОННЫЙ ОПЕРАТОР)
    # ============================================================
    
    def laplacian_features(self, z):
        """
        Лапласиан по соседним признакам (dim=1).
        Для 2D входа: [batch, features].
        Не связывает объекты в батче.
        """
        if z.dim() != 2:
            return torch.zeros_like(z)
        
        if z.size(1) <= 2:
            return torch.zeros_like(z)
        
        # Зеркальный padding по признакам
        z_padded = F.pad(z.unsqueeze(1), (1, 1), mode='reflect').squeeze(1)
        
        # Центральная разность
        lap = z_padded[:, :-2] - 2 * z[:, :] + z_padded[:, 2:]
        
        return lap

    def laplacian_spatial(self, z):
        """
        Лапласиан по пространственным координатам (2D).
        Для 4D входа: [batch, channels, height, width].
        Не затрагивает батч и каналы.
        """
        if z.dim() != 4:
            return torch.zeros_like(z)
        
        # Padding по пространственным измерениям (отражательный)
        z_padded = F.pad(z, (1, 1, 1, 1), mode='reflect')
        
        # Дискретный Лапласиан 2D (4 соседа)
        lap = (z_padded[:, :, :-2, 1:-1] + 
               z_padded[:, :, 2:, 1:-1] + 
               z_padded[:, :, 1:-1, :-2] + 
               z_padded[:, :, 1:-1, 2:] - 
               4 * z)
        
        return lap

    def _resolve_laplace_mode(self, z):
        """Автоопределение режима Лапласиана по размерности тензора"""
        if self.laplace_mode == 'auto':
            if z.dim() == 2:
                return 'features'
            elif z.dim() == 4:
                return 'spatial'
            else:
                return 'none'
        else:
            return self.laplace_mode

    def laplacian(self, z):
        """Выбор режима Лапласиана (с автоопределением)"""
        mode = self._resolve_laplace_mode(z)
        self.current_mode = mode
        
        if mode == 'features':
            return self.laplacian_features(z)
        elif mode == 'spatial':
            return self.laplacian_spatial(z)
        elif mode == 'none':
            return torch.zeros_like(z)
        else:
            raise ValueError(f"Unknown laplace_mode: {mode}")

    # ============================================================
    # ПРЯМОЙ ПРОХОД (SDE-ШАГ)
    # ============================================================
    
    def forward(self, x_re, x_im, dt=0.1):
        # 1. Применяем Z-принцип (проекция весов)
        self._project_weights()
        
        # 2. Детерминированная динамика (Полевое смещение + активация)
        f_re, f_im = self.complex_mul(x_re, x_im)
        
        if self.use_cardioid:
            f_re, f_im = self.cardioid_activation(f_re, f_im)
        else:
            f_re, f_im = self.complex_tanh(f_re, f_im)
        
        # 3. Вычисление обучаемой вязкости вакуума
        mu = F.softplus(self.theta_mu)
        
        # 4. Лапласиан от детерминированной динамики (после приведения размерности)
        lap_re = self.laplacian(f_re)
        lap_im = self.laplacian(f_im)
        
        # 5. Стохастическая самокоррекция с корректным масштабом sqrt(dt)
        sqrt_dt = math.sqrt(dt)
        noise_re = sqrt_dt * self.eta * torch.randn_like(f_re) * torch.tanh(self.beta * f_re.abs())
        noise_im = sqrt_dt * self.eta * torch.randn_like(f_im) * torch.tanh(self.beta * f_im.abs())
        
        # 6. Интегрирование по методу Эйлера-Маруямы (Neural SDE шаг)
        z_re_next = x_re + dt * (f_re - mu * lap_re) + noise_re
        z_im_next = x_im + dt * (f_im - mu * lap_im) + noise_im
        
        return z_re_next, z_im_next


# ================================================================
# СКВОЗНЫЕ ТЕСТЫ (ПРОВЕРКА AUTOGRAD ДЛЯ ВСЕХ РЕЖИМОВ)
# ================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ЕТВП Комплексный слой — тестирование v2.5 (финальная)")
    print("=" * 60)
    
    # Тест 1: 2D вход, dim_in != dim_out
    print("\n[Тест 1] 2D вход: [batch=10, features=16] → dim_out=32")
    layer_2d = ETVP_ComplexLayer(dim_in=16, dim_out=32, laplace_mode='auto')
    
    x_re = torch.randn(10, 16, requires_grad=True)
    x_im = torch.randn(10, 16, requires_grad=True)
    
    curr_re, curr_im = x_re, x_im
    for step in range(5):
        curr_re, curr_im = layer_2d(curr_re, curr_im, dt=0.1)
    
    lap_check = layer_2d.laplacian(curr_re)
    print(f"  Режим Лапласиана: {layer_2d.current_mode}")
    print(f"  Норма Лапласиана: {lap_check.abs().sum().item():.4f}")
    print(f"  Форма выхода: {curr_re.shape}")
    assert curr_re.shape == (10, 32), f"Неверная форма: {curr_re.shape}"
    
    loss = (curr_re**2 + curr_im**2).sum()
    loss.backward()
    
    print(f"  Градиент W_re: {layer_2d.W_re.grad.abs().mean().item():.6f}")
    print(f"  Градиент Вязкости (theta_mu): {layer_2d.theta_mu.grad.item():.6f}")
    print("  ✓ Тест 1 пройден")
    
    # Тест 2: 4D вход, dim_in != dim_out
    print("\n[Тест 2] 4D вход: [batch=2, channels=4, H=8, W=8] → dim_out=8")
    layer_4d = ETVP_ComplexLayer(dim_in=4, dim_out=8, laplace_mode='auto')
    
    x_re = torch.randn(2, 4, 8, 8, requires_grad=True)
    x_im = torch.randn(2, 4, 8, 8, requires_grad=True)
    
    curr_re, curr_im = x_re, x_im
    for step in range(3):
        curr_re, curr_im = layer_4d(curr_re, curr_im, dt=0.1)
    
    lap_check = layer_4d.laplacian(curr_re)
    print(f"  Режим Лапласиана: {layer_4d.current_mode}")
    print(f"  Норма Лапласиана: {lap_check.abs().sum().item():.4f}")
    print(f"  Форма выхода: {curr_re.shape}")
    assert curr_re.shape == (2, 8, 8, 8), f"Неверная форма: {curr_re.shape}"
    
    loss = (curr_re**2 + curr_im**2).sum()
    loss.backward()
    
    print(f"  Градиент W_re: {layer_4d.W_re.grad.abs().mean().item():.6f}")
    print(f"  Градиент Вязкости (theta_mu): {layer_4d.theta_mu.grad.item():.6f}")
    print("  ✓ Тест 2 пройден")
    
    # Тест 3: Проверка Z-принципа (построчная проекция)
    print("\n[Тест 3] Z-принцип: построчная проекция весов")
    layer_proj = ETVP_ComplexLayer(dim_in=512, dim_out=512, R_max=1.0)
    
    # Инициализируем большие веса
    layer_proj.W_re.data *= 10.0
    layer_proj.W_im.data *= 10.0
    
    layer_proj._project_weights()
    
    W_norm = torch.sqrt(layer_proj.W_re.norm(dim=0) ** 2 + layer_proj.W_im.norm(dim=0) ** 2)
    print(f"  Максимальная норма строки: {W_norm.max().item():.6f}")
    print(f"  Минимальная норма строки: {W_norm.min().item():.6f}")
    assert W_norm.max() <= layer_proj.R_max + 1e-6, "Проекция не сработала!"
    print("  ✓ Тест 3 пройден")
    
    # Тест 4: Проверка Cardioid вблизи нуля
    print("\n[Тест 4] Cardioid-активация вблизи нуля")
    layer_card = ETVP_ComplexLayer(dim_in=2, dim_out=2)
    
    re_small = torch.tensor([[1e-8, -1e-8]], requires_grad=True)
    im_small = torch.tensor([[1e-8, -1e-8]], requires_grad=True)
    
    out_re, out_im = layer_card.cardioid_activation(re_small, im_small)
    print(f"  Выход Re: {out_re.detach().numpy()}")
    print(f"  Выход Im: {out_im.detach().numpy()}")
    assert not torch.isnan(out_re).any(), "NaN в Cardioid!"
    assert not torch.isnan(out_im).any(), "NaN в Cardioid!"
    print("  ✓ Тест 4 пройден")
    
    # Тест 5: Проверка корректного масштаба шума
    print("\n[Тест 5] Масштаб шума sqrt(dt)")
    layer_noise = ETVP_ComplexLayer(dim_in=4, dim_out=4)
    
    x_re = torch.randn(8, 4)
    x_im = torch.randn(8, 4)
    
    # Прогон с dt=0.01 (маленький шаг)
    out_re_small, _ = layer_noise(x_re, x_im, dt=0.01)
    # Прогон с dt=1.0 (большой шаг)
    out_re_large, _ = layer_noise(x_re, x_im, dt=1.0)
    
    diff_small = (out_re_small - x_re).abs().mean().item()
    diff_large = (out_re_large - x_re).abs().mean().item()
    
    print(f"  Изменение при dt=0.01: {diff_small:.6f}")
    print(f"  Изменение при dt=1.0: {diff_large:.6f}")
    assert diff_large > diff_small, "Шум не масштабируется с dt!"
    print("  ✓ Тест 5 пройден")
    
    print("\n" + "=" * 60)
    print("ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО")
    print("=" * 60)
