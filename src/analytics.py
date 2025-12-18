"""
Módulo de Analytics Avançado
Sistema Inteligente de Monitoramento Veicular (SIMV)

Calcula métricas avançadas como velocidade estimada,
tempo de permanência, fluxo por hora, densidade de tráfego.
"""

import time
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field
import numpy as np


@dataclass
class VehicleMetrics:
    """Métricas de um veículo individual"""
    track_id: int
    entry_time: float = 0.0
    exit_time: float = 0.0
    entry_position: Tuple[float, float] = (0, 0)
    exit_position: Tuple[float, float] = (0, 0)
    positions: List[Tuple[float, float, float]] = field(default_factory=list)  # (x, y, timestamp)
    color: str = 'indefinido'
    vehicle_type: str = 'car'
    speed_estimates: List[float] = field(default_factory=list)
    direction: str = ''
    counted: bool = False


class AdvancedAnalytics:
    """Sistema de análise avançada de tráfego"""

    def __init__(self, fps: int = 30, pixels_per_meter: float = 20.0):
        """
        Inicializa o módulo de analytics.

        Args:
            fps: Frames por segundo do vídeo
            pixels_per_meter: Estimativa de pixels por metro (para cálculo de velocidade)
        """
        self.fps = fps
        self.pixels_per_meter = pixels_per_meter

        # Métricas por veículo
        self.vehicles: Dict[int, VehicleMetrics] = {}

        # Estatísticas globais
        self.total_vehicles = 0
        self.vehicles_in_scene = 0
        self.flow_per_minute: List[Tuple[float, int]] = []
        self.speed_history: List[float] = []

        # Tempo de permanência
        self.dwell_times: List[float] = []

        # Densidade de tráfego
        self.density_history: List[Tuple[float, int]] = []

        # Alertas
        self.alerts: List[Dict] = []

        # Horários de pico
        self.hourly_flow: Dict[int, int] = defaultdict(int)

        # Timestamp inicial
        self.start_time = time.time()

    def update_vehicle(self, track_id: int, bbox: List[float],
                       timestamp: float, color: str = None,
                       vehicle_type: str = None):
        """
        Atualiza informações de um veículo.

        Args:
            track_id: ID do rastreamento
            bbox: Bounding box [x1, y1, x2, y2]
            timestamp: Timestamp atual do vídeo
            color: Cor do veículo
            vehicle_type: Tipo do veículo
        """
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2

        if track_id not in self.vehicles:
            # Novo veículo detectado
            self.vehicles[track_id] = VehicleMetrics(
                track_id=track_id,
                entry_time=timestamp,
                entry_position=(center_x, center_y)
            )
            self.total_vehicles += 1
            self.vehicles_in_scene += 1

        vehicle = self.vehicles[track_id]

        # Atualizar posições
        vehicle.positions.append((center_x, center_y, timestamp))

        # Atualizar cor e tipo se fornecidos
        if color:
            vehicle.color = color
        if vehicle_type:
            vehicle.vehicle_type = vehicle_type

        # Calcular velocidade instantânea
        if len(vehicle.positions) >= 2:
            speed = self._calculate_speed(vehicle.positions[-2], vehicle.positions[-1])
            if speed > 0:
                vehicle.speed_estimates.append(speed)
                self.speed_history.append(speed)

    def vehicle_exited(self, track_id: int, timestamp: float, direction: str):
        """
        Registra saída de um veículo da cena.

        Args:
            track_id: ID do veículo
            timestamp: Timestamp de saída
            direction: Direção ('entrada' ou 'saida')
        """
        if track_id in self.vehicles:
            vehicle = self.vehicles[track_id]
            vehicle.exit_time = timestamp
            vehicle.direction = direction
            vehicle.counted = True

            if vehicle.positions:
                vehicle.exit_position = vehicle.positions[-1][:2]

            # Calcular tempo de permanência
            dwell_time = timestamp - vehicle.entry_time
            self.dwell_times.append(dwell_time)

            self.vehicles_in_scene = max(0, self.vehicles_in_scene - 1)

            # Verificar alertas
            self._check_alerts(vehicle, dwell_time)

    def _calculate_speed(self, pos1: Tuple, pos2: Tuple) -> float:
        """
        Calcula velocidade entre duas posições.

        Returns:
            Velocidade em km/h
        """
        x1, y1, t1 = pos1
        x2, y2, t2 = pos2

        dt = t2 - t1
        if dt <= 0:
            return 0

        # Distância em pixels
        distance_pixels = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        # Converter para metros
        distance_meters = distance_pixels / self.pixels_per_meter

        # Velocidade em m/s -> km/h
        speed_ms = distance_meters / dt
        speed_kmh = speed_ms * 3.6

        # Limitar a valores razoáveis (0-200 km/h)
        return min(max(speed_kmh, 0), 200)

    def _check_alerts(self, vehicle: VehicleMetrics, dwell_time: float):
        """Verifica e gera alertas baseados nas métricas"""
        # Alerta de veículo parado muito tempo
        if dwell_time > 60:  # Mais de 60 segundos
            self.alerts.append({
                'type': 'PERMANENCIA_LONGA',
                'track_id': vehicle.track_id,
                'message': f'Veículo #{vehicle.track_id} permaneceu {dwell_time:.1f}s na área',
                'timestamp': time.time(),
                'severity': 'warning'
            })

        # Alerta de alta velocidade
        if vehicle.speed_estimates:
            avg_speed = np.mean(vehicle.speed_estimates)
            if avg_speed > 80:  # Acima de 80 km/h
                self.alerts.append({
                    'type': 'ALTA_VELOCIDADE',
                    'track_id': vehicle.track_id,
                    'message': f'Veículo #{vehicle.track_id} com velocidade média de {avg_speed:.1f} km/h',
                    'timestamp': time.time(),
                    'severity': 'danger'
                })

    def update_density(self, timestamp: float, vehicle_count: int):
        """Atualiza histórico de densidade de tráfego"""
        self.density_history.append((timestamp, vehicle_count))

        # Manter apenas últimos 5 minutos
        cutoff = timestamp - 300
        self.density_history = [(t, c) for t, c in self.density_history if t > cutoff]

    def get_average_speed(self) -> float:
        """Retorna velocidade média atual"""
        if not self.speed_history:
            return 0
        # Média das últimas 50 medições
        recent = self.speed_history[-50:]
        return np.mean(recent)

    def get_average_dwell_time(self) -> float:
        """Retorna tempo médio de permanência"""
        if not self.dwell_times:
            return 0
        return np.mean(self.dwell_times)

    def get_current_flow_rate(self) -> float:
        """Retorna taxa de fluxo atual (veículos/minuto)"""
        if len(self.dwell_times) < 2:
            return 0

        # Calcular baseado nos últimos veículos
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            return (self.total_vehicles / elapsed) * 60
        return 0

    def get_traffic_density(self) -> str:
        """Classifica densidade de tráfego atual"""
        if self.vehicles_in_scene <= 2:
            return "BAIXO"
        elif self.vehicles_in_scene <= 5:
            return "MODERADO"
        elif self.vehicles_in_scene <= 10:
            return "ALTO"
        else:
            return "CONGESTIONADO"

    def get_traffic_density_color(self) -> str:
        """Retorna cor baseada na densidade"""
        density = self.get_traffic_density()
        colors = {
            "BAIXO": "#27ae60",
            "MODERADO": "#f39c12",
            "ALTO": "#e67e22",
            "CONGESTIONADO": "#e74c3c"
        }
        return colors.get(density, "#95a5a6")

    def get_peak_hours(self) -> List[Tuple[int, int]]:
        """Retorna horários de pico"""
        if not self.hourly_flow:
            return []
        sorted_hours = sorted(self.hourly_flow.items(), key=lambda x: -x[1])
        return sorted_hours[:3]

    def get_vehicle_stats(self, track_id: int) -> Optional[Dict]:
        """Retorna estatísticas de um veículo específico"""
        if track_id not in self.vehicles:
            return None

        vehicle = self.vehicles[track_id]

        avg_speed = np.mean(vehicle.speed_estimates) if vehicle.speed_estimates else 0
        max_speed = max(vehicle.speed_estimates) if vehicle.speed_estimates else 0

        dwell_time = vehicle.exit_time - vehicle.entry_time if vehicle.exit_time else 0

        return {
            'track_id': track_id,
            'color': vehicle.color,
            'type': vehicle.vehicle_type,
            'avg_speed_kmh': round(avg_speed, 1),
            'max_speed_kmh': round(max_speed, 1),
            'dwell_time_seconds': round(dwell_time, 1),
            'direction': vehicle.direction,
            'positions_count': len(vehicle.positions)
        }

    def get_summary(self) -> Dict:
        """Retorna resumo completo das análises"""
        return {
            'total_vehicles': self.total_vehicles,
            'vehicles_in_scene': self.vehicles_in_scene,
            'average_speed_kmh': round(self.get_average_speed(), 1),
            'average_dwell_time_s': round(self.get_average_dwell_time(), 1),
            'flow_rate_per_minute': round(self.get_current_flow_rate(), 1),
            'traffic_density': self.get_traffic_density(),
            'alerts_count': len(self.alerts),
            'recent_alerts': self.alerts[-5:] if self.alerts else []
        }

    def get_recent_alerts(self, count: int = 10) -> List[Dict]:
        """Retorna alertas mais recentes"""
        return self.alerts[-count:]

    def reset(self):
        """Reseta todas as métricas"""
        self.vehicles.clear()
        self.total_vehicles = 0
        self.vehicles_in_scene = 0
        self.flow_per_minute.clear()
        self.speed_history.clear()
        self.dwell_times.clear()
        self.density_history.clear()
        self.alerts.clear()
        self.hourly_flow.clear()
        self.start_time = time.time()
