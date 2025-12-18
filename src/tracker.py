"""
Módulo de Tracking de Veículos
Sistema Inteligente de Monitoramento Veicular (SIMV)

Utiliza ByteTrack para rastreamento de múltiplos objetos,
garantindo que cada veículo seja contado apenas uma vez.
"""

import supervision as sv
import numpy as np
from typing import Dict, List, Any, Optional


class VehicleTracker:
    """Rastreador de veículos usando ByteTrack via supervision"""

    def __init__(self):
        """Inicializa o tracker ByteTrack"""
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=30
        )
        self.tracked_vehicles: Dict[int, Dict[str, Any]] = {}

    def update(self, detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        """
        Atualiza o tracker com novas detecções.

        Args:
            detections: Lista de detecções do detector
            frame: Frame atual para referência

        Returns:
            Lista de detecções com IDs de tracking
        """
        if not detections:
            # Criar detecção vazia para manter o tracker atualizado
            empty_detections = sv.Detections.empty()
            self.tracker.update_with_detections(empty_detections)
            return []

        # Converter detecções para formato supervision
        xyxy = np.array([d['bbox'] for d in detections])
        confidence = np.array([d['confidence'] for d in detections])
        class_id = np.array([d['class_id'] for d in detections])

        sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id
        )

        # Atualizar tracker
        tracked_detections = self.tracker.update_with_detections(sv_detections)

        # Converter de volta para formato de lista
        tracked_list = []
        if len(tracked_detections) > 0:
            for i in range(len(tracked_detections)):
                track_id = int(tracked_detections.tracker_id[i]) if tracked_detections.tracker_id is not None else -1

                tracked_list.append({
                    'bbox': tracked_detections.xyxy[i].tolist(),
                    'class_id': int(tracked_detections.class_id[i]) if tracked_detections.class_id is not None else 2,
                    'class_name': detections[0]['class_name'] if detections else 'car',
                    'confidence': float(tracked_detections.confidence[i]) if tracked_detections.confidence is not None else 0.5,
                    'track_id': track_id
                })

                # Armazenar informações do veículo rastreado
                if track_id not in self.tracked_vehicles:
                    self.tracked_vehicles[track_id] = {
                        'first_seen': True,
                        'positions': [],
                        'color': None
                    }
                else:
                    self.tracked_vehicles[track_id]['first_seen'] = False

                # Guardar posição central do veículo
                bbox = tracked_detections.xyxy[i]
                center_y = (bbox[1] + bbox[3]) / 2
                self.tracked_vehicles[track_id]['positions'].append(center_y)

        return tracked_list

    def get_vehicle_direction(self, track_id: int) -> Optional[str]:
        """
        Determina a direção do movimento de um veículo.

        Args:
            track_id: ID do veículo rastreado

        Returns:
            'down' (entrando), 'up' (saindo) ou None
        """
        if track_id not in self.tracked_vehicles:
            return None

        positions = self.tracked_vehicles[track_id]['positions']
        if len(positions) < 5:
            return None

        # Calcular direção média do movimento
        recent_positions = positions[-10:]
        if len(recent_positions) >= 2:
            movement = recent_positions[-1] - recent_positions[0]
            if movement > 10:
                return 'down'  # Movendo para baixo (entrando)
            elif movement < -10:
                return 'up'    # Movendo para cima (saindo)

        return None

    def set_vehicle_color(self, track_id: int, color: str):
        """Define a cor de um veículo rastreado"""
        if track_id in self.tracked_vehicles:
            self.tracked_vehicles[track_id]['color'] = color

    def get_vehicle_color(self, track_id: int) -> Optional[str]:
        """Obtém a cor de um veículo rastreado"""
        if track_id in self.tracked_vehicles:
            return self.tracked_vehicles[track_id].get('color')
        return None

    def reset(self):
        """Reseta o tracker"""
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=30
        )
        self.tracked_vehicles.clear()
