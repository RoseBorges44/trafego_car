"""
Módulo de Contagem de Veículos
Sistema Inteligente de Monitoramento Veicular (SIMV)

Implementa contagem de veículos que entram e saem usando
linhas de referência e tracking de trajetória.
"""

import cv2
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class VehicleRecord:
    """Registro de um veículo contado"""
    track_id: int
    direction: str  # 'entrada' ou 'saida'
    color: str
    vehicle_type: str
    timestamp: float


@dataclass
class CountingStats:
    """Estatísticas de contagem"""
    total_entrada: int = 0
    total_saida: int = 0
    por_cor: Dict[str, Dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: {'entrada': 0, 'saida': 0}))
    por_tipo: Dict[str, Dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: {'entrada': 0, 'saida': 0}))
    registros: List[VehicleRecord] = field(default_factory=list)


class VehicleCounter:
    """Sistema de contagem de veículos com linhas de referência"""

    def __init__(self, frame_height: int, line_position: float = 0.5):
        """
        Inicializa o contador de veículos.

        Args:
            frame_height: Altura do frame de vídeo
            line_position: Posição relativa da linha de contagem (0-1)
        """
        self.frame_height = frame_height
        self.line_y = int(frame_height * line_position)

        # Zona de contagem (margem acima e abaixo da linha)
        self.zone_margin = int(frame_height * 0.05)

        # Veículos já contados (evitar duplicidade)
        self.counted_vehicles: Set[int] = set()

        # Histórico de posições para determinar direção
        self.position_history: Dict[int, List[float]] = defaultdict(list)

        # Estatísticas
        self.stats = CountingStats()

    def update(self, tracked_vehicles: List[Dict], colors: Dict[int, str],
               timestamp: float = 0.0) -> List[Tuple[int, str]]:
        """
        Atualiza contagem com veículos detectados.

        Args:
            tracked_vehicles: Lista de veículos rastreados
            colors: Dicionário de cores por track_id
            timestamp: Timestamp atual do vídeo

        Returns:
            Lista de (track_id, direção) para veículos recém-contados
        """
        newly_counted = []

        for vehicle in tracked_vehicles:
            track_id = vehicle.get('track_id', -1)
            if track_id < 0:
                continue

            bbox = vehicle['bbox']
            center_y = (bbox[1] + bbox[3]) / 2

            # Atualizar histórico de posições
            self.position_history[track_id].append(center_y)

            # Manter apenas últimas 30 posições
            if len(self.position_history[track_id]) > 30:
                self.position_history[track_id] = self.position_history[track_id][-30:]

            # Verificar se veículo já foi contado
            if track_id in self.counted_vehicles:
                continue

            # Verificar se cruzou a linha de contagem
            if self._crossed_line(track_id, center_y):
                direction = self._get_direction(track_id)

                if direction:
                    self.counted_vehicles.add(track_id)

                    color = colors.get(track_id, 'indefinido')
                    vehicle_type = vehicle.get('class_name', 'car')

                    # Registrar contagem
                    record = VehicleRecord(
                        track_id=track_id,
                        direction=direction,
                        color=color,
                        vehicle_type=vehicle_type,
                        timestamp=timestamp
                    )
                    self.stats.registros.append(record)

                    # Atualizar estatísticas
                    if direction == 'entrada':
                        self.stats.total_entrada += 1
                    else:
                        self.stats.total_saida += 1

                    self.stats.por_cor[color][direction] += 1
                    self.stats.por_tipo[vehicle_type][direction] += 1

                    newly_counted.append((track_id, direction))

        return newly_counted

    def _crossed_line(self, track_id: int, current_y: float) -> bool:
        """Verifica se o veículo está na zona de contagem"""
        return abs(current_y - self.line_y) < self.zone_margin

    def _get_direction(self, track_id: int) -> Optional[str]:
        """
        Determina a direção do movimento do veículo.

        Returns:
            'entrada' (movendo para baixo), 'saida' (movendo para cima) ou None
        """
        positions = self.position_history.get(track_id, [])

        if len(positions) < 5:
            return None

        # Analisar movimento
        start_positions = positions[:len(positions)//2]
        end_positions = positions[len(positions)//2:]

        avg_start = sum(start_positions) / len(start_positions)
        avg_end = sum(end_positions) / len(end_positions)

        movement = avg_end - avg_start

        if movement > 20:  # Movendo para baixo (y aumenta)
            return 'entrada'
        elif movement < -20:  # Movendo para cima (y diminui)
            return 'saida'

        return None

    def draw_counting_line(self, frame: np.ndarray) -> np.ndarray:
        """
        Desenha a linha de contagem no frame.

        Args:
            frame: Frame BGR

        Returns:
            Frame com linha desenhada
        """
        height, width = frame.shape[:2]

        # Linha principal de contagem
        cv2.line(frame, (0, self.line_y), (width, self.line_y),
                 (0, 255, 255), 3)

        # Zona de contagem (área semi-transparente)
        overlay = frame.copy()
        cv2.rectangle(overlay,
                      (0, self.line_y - self.zone_margin),
                      (width, self.line_y + self.zone_margin),
                      (0, 255, 255), -1)
        cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)

        # Labels
        cv2.putText(frame, "LINHA DE CONTAGEM", (10, self.line_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return frame

    def draw_stats(self, frame: np.ndarray) -> np.ndarray:
        """
        Desenha estatísticas no frame.

        Args:
            frame: Frame BGR

        Returns:
            Frame com estatísticas
        """
        # Fundo semi-transparente para estatísticas
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Textos
        cv2.putText(frame, "SIMV - Contagem de Veiculos", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.putText(frame, f"Entrada: {self.stats.total_entrada}", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(frame, f"Saida: {self.stats.total_saida}", (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return frame

    def get_stats(self) -> Dict:
        """Retorna estatísticas em formato de dicionário"""
        return {
            'total_entrada': self.stats.total_entrada,
            'total_saida': self.stats.total_saida,
            'total_geral': self.stats.total_entrada + self.stats.total_saida,
            'por_cor': dict(self.stats.por_cor),
            'por_tipo': dict(self.stats.por_tipo),
            'registros': len(self.stats.registros)
        }

    def get_color_distribution(self) -> Dict[str, int]:
        """Retorna distribuição total de cores"""
        distribution = {}
        for color, counts in self.stats.por_cor.items():
            distribution[color] = counts['entrada'] + counts['saida']
        return distribution

    def reset(self):
        """Reseta todas as contagens"""
        self.counted_vehicles.clear()
        self.position_history.clear()
        self.stats = CountingStats()
