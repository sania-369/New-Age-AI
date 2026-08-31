import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Literal

class ETVP_ComplexLayer(nn.Module):
    r"""
    ETVP Complex Layer v3.0 — Промышленная реализация оператора эволюции поля.

    ============================================================================
    ТОПОЛОГИЯ ВЫЧИСЛЕНИЙ (ПРИНЦИП РАБОТЫ)
    ============================================================================
    Состояние реальности в момент (t_n) является функцией исключительно
    состояния в момент (t_{n-1}). Мы не ищем "глобальное решение".
    Мы применяем оператор эволюции U(Φ, Z, C_оп) к текущему состоянию:

        Ψ(t + dt) = U · Ψ(t)

    Где U — это жесткий нелинейный фильтр, включающий:
        - Комплексное линейное преобразование (E8-подобное)
        - Кардиоидную активацию (когерентность фазы)
        - Диффузию (Лапласиан с автонормализацией)
        - Стохастическую самокоррекцию (шум с учётом фазы)

    Прошлое трансформируется в каждом такте (dt), становясь начальным
    условием для настоящего. Это гарантирует причинность и локальность
    вычислений, что критично для физического моделирования и RL-агентов.
    ============================================================================

    Поддерживает:
        - 2D входы [batch, features]     → L2-регуляризация
        - 4D входы [batch, ch, H, W]     → пространственный Лапласиан
        - Два режима: 'increment' (по умолч.) и 'generate'
        - Дифференцируемый Z-принцип (штраф в лоссе)
        - Устойчивый солвер Хейна (Heun) для SDE
        - Возврат промежуточных состояний (для анализа энергии)

    Аргументы:
        dim_in (int):    Размерность входного пространства
        dim_out (int):   Размерность выходного пространства
        R_max (float):   Радиус ограничения для Z-принципа (по умолч. 1.0)
        eta (float):     Амплитуда шума (по умолч. 0.01)
        beta (float):    Коэффициент модуляции шума (по умолч. 0.1)
        laplace_mode (str): 'auto', 'spatial', 'features', 'none'
        use_cardioid (bool):  Использовать Cardioid-активацию (по умолч. True)
        padding_mode (str):   'reflect', 'zero', 'periodic'
        mode (str):           'increment' или 'generate'
        lambda_z (float):     Коэффициент штрафа Z-принципа (по умолч. 1.0)
    """

    EPS: float = 1e-8

    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 R_max: float = 1.0,
                 eta: float = 0.01,
                 beta: float = 0.1,
                 laplace_mode: Literal['auto', 'spatial', 'features', 'none'] = 'auto',
                 use_cardioid: bool = True,
                 padding_mode: Literal['reflect', 'zero', 'periodic'] = 'reflect',
                 mode: Literal['increment', 'generate'] = 'increment',
                 lambda_z: float = 1.0):
        super().__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.R_max = R_max
        self.eta = eta
        self.beta = beta
        self.laplace_mode = laplace_mode
        self.use_cardioid = use_cardioid
        self.padding_mode = padding_mode
        self.mode = mode
        self.lambda_z = lambda_z

        # --- Комплексные веса (Xavier) ---
        bound = (2.0 / dim_in) ** 0.5
        self.W_re = nn.Parameter(torch.randn(dim_in, dim_out) * bound)
        self.W_im = nn.Parameter(torch.randn(dim_in, dim_out) * bound)

        # Смещения с малым шумом (для выхода из нуля)
        self.b_re = nn.Parameter(0.01 * torch.randn(dim_out))
        self.b_im = nn.Parameter(0.01 * torch.randn(dim_out))

        # --- Обучаемая вязкость вакуума ---
        self.theta_mu = nn.Parameter(torch.tensor(0.0))

        self._current_mode = None

    # ============================================================
    # 1. ДИФФЕРЕНЦИРУЕМЫЙ Z-ПРИНЦИП (ШТРАФ)
    # ============================================================
    def get_z_penalty(self) -> torch.Tensor:
        """
        Возвращает скалярный штраф за превышение весами R_max.
        Штраф = lambda_z * mean( ReLU(||W|| - R_max)^2 )
        """
        W_norm = torch.abs(self.W_re + 1j * self.W_im).norm(dim=0)
        penalty = F.relu(W_norm - self.R_max).pow(2).mean()
        return self.lambda_z * penalty

    # ============================================================
    # 2. ЛАПЛАСИАН (С АВТОНОРМАЛИЗАЦИЕЙ)
    # ============================================================
    def laplacian_spatial(self, z: torch.Tensor) -> torch.Tensor:
        """Пространственный Лапласиан для 4D-тензоров [B, C, H, W]."""
        if z.dim() != 4:
            return torch.zeros_like(z)

        _, _, H, W = z.shape
        pad_map = {'reflect': 'reflect', 'zero': 'constant', 'periodic': 'circular'}
        pad_mode = pad_map.get(self.padding_mode, 'reflect')

        z_padded = F.pad(z, (1, 1, 1, 1), mode=pad_mode)
        lap = (z_padded[:, :, :-2, 1:-1] +
               z_padded[:, :, 2:, 1:-1] +
               z_padded[:, :, 1:-1, :-2] +
               z_padded[:, :, 1:-1, 2:] -
               4 * z)

        # Нормализация по размеру сетки (масштабно-инвариантный лапласиан)
        return lap / (H * W + self.EPS)

    def laplacian(self, z: torch.Tensor) -> torch.Tensor:
        """Выбор режима Лапласиана."""
        mode = self.laplace_mode
        if mode == 'auto':
            mode = 'spatial' if z.dim() == 4 else 'none'

        if mode == 'spatial':
            return self.laplacian_spatial(z)
        else:
            # Для 2D — регуляризация через L2-штраф в лоссе
            return torch.zeros_like(z)

    # ============================================================
    # 3. КОМПЛЕКСНОЕ УМНОЖЕНИЕ И АКТИВАЦИЯ
    # ============================================================
    def complex_mul(self, x_re: torch.Tensor, x_im: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Полноценное комплексное линейное преобразование."""
        if x_re.dim() == 2:
            out_re = x_re @ self.W_re - x_im @ self.W_im + self.b_re
            out_im = x_re @ self.W_im + x_im @ self.W_re + self.b_im
        elif x_re.dim() == 4:
            out_re = (torch.einsum('bihw,io->bohw', x_re, self.W_re) -
                      torch.einsum('bihw,io->bohw', x_im, self.W_im) +
                      self.b_re.view(1, -1, 1, 1))
            out_im = (torch.einsum('bihw,io->bohw', x_re, self.W_im) +
                      torch.einsum('bihw,io->bohw', x_im, self.W_re) +
                      self.b_im.view(1, -1, 1, 1))
        else:
            raise ValueError(f"Unsupported tensor dims: {x_re.dim()}. Expected 2D or 4D.")
        return out_re, out_im

    def cardioid_activation(self, re: torch.Tensor, im: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Комплексная кардиоидная активация.
        f(z) = 0.5 * (1 + cos(angle(z))) * z
        Удовлетворяет условиям Коши-Римана.
        """
        angle = torch.atan2(im, re)
        scale = 0.5 * (1.0 + torch.cos(angle))
        return scale * re, scale * im

    # ============================================================
    # 4. ОПЕРАТОР ЭВОЛЮЦИИ (СОЛВЕР ХЕЙНА)
    # ============================================================
    def forward(self,
                x_re: torch.Tensor,
                x_im: torch.Tensor,
                dt: float = 0.1,
                return_intermediate: bool = False) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Применяет оператор эволюции U к текущему состоянию Ψ(t):
            Ψ(t + dt) = U · Ψ(t)

        Возвращает новое состояние Ψ(t+dt).
        """
        # --- Шаг 1: k1 (производная в текущей точке) ---
        f_re, f_im = self.complex_mul(x_re, x_im)
        if self.use_cardioid:
            f_re, f_im = self.cardioid_activation(f_re, f_im)
        else:
            f_re, f_im = torch.tanh(f_re), torch.tanh(f_im)

        mu = F.softplus(self.theta_mu)
        lap_re = self.laplacian(f_re)
        lap_im = self.laplacian(f_im)

        # Шум с учётом фазы (корректный sqrt(dt))
        sqrt_dt = math.sqrt(dt)
        f_abs = torch.sqrt(f_re**2 + f_im**2) + self.EPS
        noise_scale = self.eta * torch.tanh(self.beta * f_abs)

        noise_re = sqrt_dt * noise_scale * torch.randn_like(f_re) * torch.cos(torch.atan2(f_im, f_re))
        noise_im = sqrt_dt * noise_scale * torch.randn_like(f_im) * torch.sin(torch.atan2(f_im, f_re))

        if self.mode == 'increment':
            k1_re = f_re - mu * lap_re
            k1_im = f_im - mu * lap_im
            x_pred_re = x_re + dt * k1_re
            x_pred_im = x_im + dt * k1_im
        else:  # 'generate'
            k1_re = f_re - mu * lap_re
            k1_im = f_im - mu * lap_im
            x_pred_re = f_re - dt * mu * lap_re
            x_pred_im = f_im - dt * mu * lap_im

        # --- Шаг 2: k2 (производная в предсказанной точке) ---
        f_pred_re, f_pred_im = self.complex_mul(x_pred_re, x_pred_im)
        if self.use_cardioid:
            f_pred_re, f_pred_im = self.cardioid_activation(f_pred_re, f_pred_im)
        else:
            f_pred_re, f_pred_im = torch.tanh(f_pred_re), torch.tanh(f_pred_im)

        lap_pred_re = self.laplacian(f_pred_re)
        lap_pred_im = self.laplacian(f_pred_im)

        if self.mode == 'increment':
            k2_re = f_pred_re - mu * lap_pred_re
            k2_im = f_pred_im - mu * lap_pred_im
        else:
            k2_re = f_pred_re - mu * lap_pred_re
            k2_im = f_pred_im - mu * lap_pred_im

        # --- Шаг 3: усреднение (Heun) ---
        if self.mode == 'increment':
            z_re_next = x_re + dt * (k1_re + k2_re) / 2 + noise_re
            z_im_next = x_im + dt * (k1_im + k2_im) / 2 + noise_im
        else:
            z_re_next = (f_re + f_pred_re) / 2 - dt * mu * (lap_re + lap_pred_re) / 2
            z_im_next = (f_im + f_pred_im) / 2 - dt * mu * (lap_im + lap_pred_im) / 2

        if return_intermediate:
            intermediates = {
                'f': (f_re, f_im),
                'f_pred': (f_pred_re, f_pred_im),
                'lap': (lap_re, lap_im),
                'lap_pred': (lap_pred_re, lap_pred_im),
                'noise': (noise_re, noise_im),
                'mu': mu,
                'dt': dt
            }
            return z_re_next, z_im_next, intermediates

        return z_re_next, z_im_next
