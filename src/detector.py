"""
Módulo de Detecção de Veículos usando YOLOv8
Sistema Inteligente de Monitoramento Veicular (SIMV)
"""

from ultralytics import YOLO
import numpy as np
from typing import List, Tuple, Dict, Any


# Classes de veículos no dataset COCO (usado pelo YOLOv8)
VEHICLE_CLASSES = {
    2: 'car',       # carro
    3: 'motorcycle', # moto
    5: 'bus',       # ônibus
    7: 'truck'      # caminhão
}


class VehicleDetector:
    """Detector de veículos usando YOLOv8"""

    def __init__(self, model_size: str = 'n', confidence: float = 0.5):
        """
        Inicializa o detector de veículos.

        Args:
            model_size: Tamanho do modelo YOLOv8 ('n', 's', 'm', 'l', 'x')
            confidence: Limiar de confiança para detecções (0-1)
        """
        self.confidence = confidence
        self.model = YOLO(f'yolov8{model_size}.pt')
        self.vehicle_class_ids = list(VEHICLE_CLASSES.keys())

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detecta veículos em um frame.

        Args:
            frame: Imagem BGR do OpenCV

        Returns:
            Lista de detecções com bounding boxes, classes e confiança
        """
        results = self.model(frame, conf=self.confidence, verbose=False)[0]

        detections = []

        for box in results.boxes:
            class_id = int(box.cls[0])

            # Filtrar apenas veículos
            if class_id in self.vehicle_class_ids:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])

                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'class_id': class_id,
                    'class_name': VEHICLE_CLASSES[class_id],
                    'confidence': confidence
                })

        return detections

    def get_boxes_for_tracking(self, detections: List[Dict]) -> np.ndarray:
        """
        Converte detecções para formato usado pelo tracker.

        Args:
            detections: Lista de detecções

        Returns:
            Array numpy com formato [x1, y1, x2, y2, confidence]
        """
        if not detections:
            return np.empty((0, 5))

        boxes = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            boxes.append([x1, y1, x2, y2, det['confidence']])

        return np.array(boxes)
