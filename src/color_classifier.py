"""
Módulo de Classificação de Cores de Veículos
Sistema Inteligente de Monitoramento Veicular (SIMV)

Classifica a cor predominante dos veículos detectados
usando análise de histograma de cores no espaço HSV.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from collections import Counter


# Definição de faixas de cores no espaço HSV
# Formato: (H_min, H_max, S_min, S_max, V_min, V_max)
COLOR_RANGES = {
    'vermelho': [(0, 10, 70, 255, 50, 255), (170, 180, 70, 255, 50, 255)],
    'laranja': [(10, 25, 70, 255, 50, 255)],
    'amarelo': [(25, 35, 70, 255, 50, 255)],
    'verde': [(35, 85, 70, 255, 50, 255)],
    'azul': [(85, 130, 70, 255, 50, 255)],
    'roxo': [(130, 160, 70, 255, 50, 255)],
    'rosa': [(160, 170, 70, 255, 50, 255)],
    'branco': [(0, 180, 0, 30, 200, 255)],
    'preto': [(0, 180, 0, 255, 0, 50)],
    'cinza': [(0, 180, 0, 30, 50, 200)],
    'prata': [(0, 180, 0, 40, 150, 220)],
}

# Traduções para exibição
COLOR_TRANSLATIONS = {
    'vermelho': 'Vermelho',
    'laranja': 'Laranja',
    'amarelo': 'Amarelo',
    'verde': 'Verde',
    'azul': 'Azul',
    'roxo': 'Roxo',
    'rosa': 'Rosa',
    'branco': 'Branco',
    'preto': 'Preto',
    'cinza': 'Cinza',
    'prata': 'Prata',
}


class ColorClassifier:
    """Classificador de cores de veículos"""

    def __init__(self):
        """Inicializa o classificador de cores"""
        self.color_history = {}

    def classify(self, frame: np.ndarray, bbox: list) -> str:
        """
        Classifica a cor de um veículo a partir de seu bounding box.

        Args:
            frame: Frame BGR do OpenCV
            bbox: Bounding box [x1, y1, x2, y2]

        Returns:
            Nome da cor classificada
        """
        x1, y1, x2, y2 = [int(c) for c in bbox]

        # Garantir coordenadas válidas
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            return 'indefinido'

        # Extrair região do veículo
        vehicle_roi = frame[y1:y2, x1:x2]

        if vehicle_roi.size == 0:
            return 'indefinido'

        # Recortar região central (evita bordas e rodas)
        h, w = vehicle_roi.shape[:2]
        margin_h = int(h * 0.2)
        margin_w = int(w * 0.15)

        center_roi = vehicle_roi[margin_h:h-margin_h, margin_w:w-margin_w]

        if center_roi.size == 0:
            center_roi = vehicle_roi

        # Converter para HSV
        hsv = cv2.cvtColor(center_roi, cv2.COLOR_BGR2HSV)

        # Contar pixels de cada cor
        color_counts = {}
        total_pixels = hsv.shape[0] * hsv.shape[1]

        for color_name, ranges in COLOR_RANGES.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

            for range_values in ranges:
                h_min, h_max, s_min, s_max, v_min, v_max = range_values
                lower = np.array([h_min, s_min, v_min])
                upper = np.array([h_max, s_max, v_max])

                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

            color_counts[color_name] = np.sum(mask > 0)

        # Encontrar cor predominante
        if color_counts:
            max_color = max(color_counts.items(), key=lambda x: x[1])
            if max_color[1] > total_pixels * 0.1:  # Pelo menos 10% dos pixels
                return max_color[0]

        return 'indefinido'

    def classify_with_smoothing(self, track_id: int, frame: np.ndarray, bbox: list) -> str:
        """
        Classifica cor com suavização temporal para maior precisão.

        Args:
            track_id: ID do veículo rastreado
            frame: Frame BGR
            bbox: Bounding box

        Returns:
            Cor classificada com suavização
        """
        current_color = self.classify(frame, bbox)

        if track_id not in self.color_history:
            self.color_history[track_id] = []

        self.color_history[track_id].append(current_color)

        # Manter apenas últimas 10 classificações
        if len(self.color_history[track_id]) > 10:
            self.color_history[track_id] = self.color_history[track_id][-10:]

        # Retornar cor mais frequente
        counter = Counter(self.color_history[track_id])
        most_common = counter.most_common(1)[0][0]

        return most_common

    def get_color_display_name(self, color: str) -> str:
        """Retorna nome de exibição da cor"""
        return COLOR_TRANSLATIONS.get(color, color.capitalize())

    def get_color_bgr(self, color: str) -> Tuple[int, int, int]:
        """Retorna cor BGR para visualização"""
        colors_bgr = {
            'vermelho': (0, 0, 255),
            'laranja': (0, 165, 255),
            'amarelo': (0, 255, 255),
            'verde': (0, 255, 0),
            'azul': (255, 0, 0),
            'roxo': (128, 0, 128),
            'rosa': (203, 192, 255),
            'branco': (255, 255, 255),
            'preto': (0, 0, 0),
            'cinza': (128, 128, 128),
            'prata': (192, 192, 192),
            'indefinido': (100, 100, 100),
        }
        return colors_bgr.get(color, (100, 100, 100))

    def reset(self):
        """Limpa histórico de cores"""
        self.color_history.clear()
