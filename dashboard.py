"""
SIMV Dashboard - Painel de Controle de Trafego
Sistema Inteligente de Monitoramento Veicular

Interface profissional para analise de trafego urbano
Desenvolvido para Prefeitura Municipal - IBM Smart Cities
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import ttkbootstrap as ttk
from ttkbootstrap.constants import *

import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
import json
from pathlib import Path
from datetime import datetime
from collections import deque

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib
matplotlib.use('TkAgg')

from src.detector import VehicleDetector
from src.tracker import VehicleTracker
from src.color_classifier import ColorClassifier
from src.counter import VehicleCounter
from src.analytics import AdvancedAnalytics


class SIMVDashboard:
    """Dashboard principal do SIMV"""

    def __init__(self):
        # Janela principal - MAIOR
        self.root = ttk.Window(
            title="SIMV - Sistema Inteligente de Monitoramento Veicular | Prefeitura Municipal",
            themename="darkly",
            size=(1800, 1000),
            minsize=(1600, 900)
        )
        self.root.place_window_center()

        # Variaveis de controle
        self.is_running = False
        self.is_paused = False
        self.video_path = None
        self.cap = None
        self.current_frame = None

        # Modulos de processamento
        self.detector = None
        self.tracker = None
        self.color_classifier = None
        self.counter = None
        self.analytics = None

        # Dados para graficos
        self.flow_data = deque(maxlen=100)
        self.time_labels = deque(maxlen=100)

        # Cores do veiculo
        self.vehicle_colors = {}

        # Thread de processamento
        self.processing_thread = None

        # Variaveis de ajuste (inicializar antes de build_ui)
        self.video_scale_var = ttk.DoubleVar(value=1.0)
        self.chart_scale_var = ttk.DoubleVar(value=1.0)
        self.panel_width_var = ttk.IntVar(value=550)

        # Construir interface
        self._build_ui()

        # Iniciar loop de atualizacao
        self._update_clock()

    def _build_ui(self):
        """Constroi a interface do usuario"""
        # Header
        self._build_header()

        # Container principal
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=BOTH, expand=YES, padx=15, pady=10)

        # Painel esquerdo (video + controles)
        left_panel = ttk.Frame(main_container)
        left_panel.pack(side=LEFT, fill=BOTH, expand=YES, padx=(0, 10))

        self._build_video_panel(left_panel)
        self._build_controls(left_panel)

        # Painel direito (estatisticas + graficos)
        self.right_panel = ttk.Frame(main_container, width=550)
        self.right_panel.pack(side=RIGHT, fill=BOTH, padx=(10, 0))
        self.right_panel.pack_propagate(False)

        self._build_stats_panel(self.right_panel)
        self._build_charts_panel(self.right_panel)
        self._build_alerts_panel(self.right_panel)

    def _build_header(self):
        """Constroi o cabecalho"""
        header = ttk.Frame(self.root, bootstyle="dark")
        header.pack(fill=X, padx=15, pady=(15, 10))

        # Logo e titulo
        title_frame = ttk.Frame(header)
        title_frame.pack(side=LEFT)

        ttk.Label(
            title_frame,
            text="SIMV",
            font=("Segoe UI", 32, "bold"),
            bootstyle="info"
        ).pack(side=LEFT, padx=(0, 15))

        subtitle_frame = ttk.Frame(title_frame)
        subtitle_frame.pack(side=LEFT)

        ttk.Label(
            subtitle_frame,
            text="Sistema Inteligente de Monitoramento Veicular",
            font=("Segoe UI", 14, "bold"),
            bootstyle="light"
        ).pack(anchor=W)

        ttk.Label(
            subtitle_frame,
            text="Controle de Trafego Urbano",
            font=("Segoe UI", 11),
            bootstyle="secondary"
        ).pack(anchor=W)

        # Info direita
        info_frame = ttk.Frame(header)
        info_frame.pack(side=RIGHT)

        ttk.Label(
            info_frame,
            text="Prefeitura Municipal",
            font=("Segoe UI", 12, "bold"),
            bootstyle="warning"
        ).pack(side=RIGHT, padx=20)

        # Relogio
        self.clock_label = ttk.Label(
            info_frame,
            text="",
            font=("Segoe UI", 14, "bold"),
            bootstyle="info"
        )
        self.clock_label.pack(side=RIGHT, padx=20)

    def _build_video_panel(self, parent):
        """Constroi o painel de video"""
        video_frame = ttk.Labelframe(parent, text=" Monitoramento em Tempo Real ", padding=15)
        video_frame.pack(fill=BOTH, expand=YES, pady=(0, 10))

        # Canvas para video - MAIOR
        self.video_canvas = ttk.Label(video_frame)
        self.video_canvas.pack(fill=BOTH, expand=YES)

        # Placeholder inicial
        self._show_placeholder()

    def _show_placeholder(self):
        """Mostra placeholder quando nao ha video"""
        placeholder = np.zeros((600, 1066, 3), dtype=np.uint8)
        placeholder[:] = (25, 25, 25)

        # Texto central
        cv2.putText(placeholder, "SIMV - Sistema de Monitoramento de Trafego",
                    (220, 280), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (80, 80, 80), 2)
        cv2.putText(placeholder, "Selecione um video para iniciar a analise",
                    (300, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 60, 60), 1)

        self._display_frame(placeholder)

    def _build_controls(self, parent):
        """Constroi os controles"""
        controls_frame = ttk.Labelframe(parent, text=" Painel de Controle ", padding=15)
        controls_frame.pack(fill=X, pady=5)

        # Linha 1: Selecao de arquivo
        file_frame = ttk.Frame(controls_frame)
        file_frame.pack(fill=X, pady=(0, 15))

        self.file_label = ttk.Label(
            file_frame,
            text="Nenhum arquivo selecionado",
            font=("Segoe UI", 11),
            bootstyle="secondary"
        )
        self.file_label.pack(side=LEFT, fill=X, expand=YES)

        ttk.Button(
            file_frame,
            text="Selecionar Video",
            bootstyle="info",
            command=self._select_video,
            width=18
        ).pack(side=RIGHT, padx=(15, 0))

        # Linha 2: Botoes de controle
        btn_frame = ttk.Frame(controls_frame)
        btn_frame.pack(fill=X, pady=(0, 15))

        self.start_btn = ttk.Button(
            btn_frame,
            text="INICIAR",
            bootstyle="success",
            command=self._start_processing,
            width=14
        )
        self.start_btn.pack(side=LEFT, padx=(0, 10))

        self.pause_btn = ttk.Button(
            btn_frame,
            text="PAUSAR",
            bootstyle="warning",
            command=self._toggle_pause,
            width=14,
            state=DISABLED
        )
        self.pause_btn.pack(side=LEFT, padx=10)

        self.stop_btn = ttk.Button(
            btn_frame,
            text="PARAR",
            bootstyle="danger",
            command=self._stop_processing,
            width=14,
            state=DISABLED
        )
        self.stop_btn.pack(side=LEFT, padx=10)

        ttk.Separator(btn_frame, orient=VERTICAL).pack(side=LEFT, fill=Y, padx=20)

        ttk.Button(
            btn_frame,
            text="Exportar Relatorio",
            bootstyle="info-outline",
            command=self._export_report,
            width=18
        ).pack(side=LEFT, padx=10)

        ttk.Button(
            btn_frame,
            text="Ajustes",
            bootstyle="secondary-outline",
            command=self._open_settings,
            width=10
        ).pack(side=LEFT, padx=10)

        # Linha 3: Controle da posicao da linha
        line_frame = ttk.Frame(controls_frame)
        line_frame.pack(fill=X, pady=(0, 10))

        ttk.Label(
            line_frame,
            text="Posicao da Linha de Contagem:",
            font=("Segoe UI", 10)
        ).pack(side=LEFT)

        self.line_position_var = ttk.DoubleVar(value=0.5)
        self.line_slider = ttk.Scale(
            line_frame,
            from_=0.2,
            to=0.8,
            variable=self.line_position_var,
            bootstyle="info",
            command=self._on_line_position_change,
            length=300
        )
        self.line_slider.pack(side=LEFT, padx=20)

        self.line_position_label = ttk.Label(
            line_frame,
            text="50%",
            font=("Segoe UI", 12, "bold"),
            bootstyle="info",
            width=8
        )
        self.line_position_label.pack(side=LEFT)

        # Linha 4: Barra de progresso
        progress_frame = ttk.Frame(controls_frame)
        progress_frame.pack(fill=X, pady=(10, 0))

        ttk.Label(progress_frame, text="Progresso:", font=("Segoe UI", 10)).pack(side=LEFT)

        self.progress_var = ttk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            bootstyle="success-striped",
            length=500
        )
        self.progress_bar.pack(side=LEFT, fill=X, expand=YES, padx=15)

        self.progress_label = ttk.Label(
            progress_frame,
            text="0%",
            font=("Segoe UI", 12, "bold"),
            width=8
        )
        self.progress_label.pack(side=RIGHT)

    def _build_stats_panel(self, parent):
        """Constroi o painel de estatisticas - NUMEROS GRANDES"""
        stats_frame = ttk.Labelframe(parent, text=" Estatisticas de Contagem ", padding=20)
        stats_frame.pack(fill=X, pady=(0, 10))

        # Grid de metricas - NUMEROS BEM MAIORES
        metrics_grid = ttk.Frame(stats_frame)
        metrics_grid.pack(fill=X)

        self.metric_widgets = {}

        # Linha 1: Entrada, Saida, Total
        row1 = ttk.Frame(metrics_grid)
        row1.pack(fill=X, pady=10)

        for key, label, style in [("total_entrada", "ENTRADA", "success"),
                                   ("total_saida", "SAIDA", "danger"),
                                   ("total_geral", "TOTAL", "info")]:
            frame = ttk.Frame(row1)
            frame.pack(side=LEFT, expand=YES, fill=X, padx=5)

            ttk.Label(
                frame,
                text=label,
                font=("Segoe UI", 11, "bold"),
                bootstyle="secondary"
            ).pack()

            value_label = ttk.Label(
                frame,
                text="0",
                font=("Segoe UI", 42, "bold"),
                bootstyle=style
            )
            value_label.pack(pady=5)
            self.metric_widgets[key] = value_label

        # Linha 2: Fluxo
        row2 = ttk.Frame(metrics_grid)
        row2.pack(fill=X, pady=15)

        flow_frame = ttk.Frame(row2)
        flow_frame.pack(expand=YES)

        ttk.Label(
            flow_frame,
            text="FLUXO DE VEICULOS",
            font=("Segoe UI", 11, "bold"),
            bootstyle="secondary"
        ).pack()

        flow_value_frame = ttk.Frame(flow_frame)
        flow_value_frame.pack()

        self.metric_widgets['fluxo_minuto'] = ttk.Label(
            flow_value_frame,
            text="0.0",
            font=("Segoe UI", 36, "bold"),
            bootstyle="warning"
        )
        self.metric_widgets['fluxo_minuto'].pack(side=LEFT)

        ttk.Label(
            flow_value_frame,
            text=" veic/min",
            font=("Segoe UI", 14),
            bootstyle="secondary"
        ).pack(side=LEFT, pady=(15, 0))

        # Indicador de nivel de trafego
        ttk.Separator(stats_frame, orient=HORIZONTAL).pack(fill=X, pady=15)

        density_frame = ttk.Frame(stats_frame)
        density_frame.pack(fill=X)

        ttk.Label(
            density_frame,
            text="NIVEL DE TRAFEGO:",
            font=("Segoe UI", 12, "bold")
        ).pack(side=LEFT)

        self.density_label = ttk.Label(
            density_frame,
            text="AGUARDANDO",
            font=("Segoe UI", 16, "bold"),
            bootstyle="secondary"
        )
        self.density_label.pack(side=LEFT, padx=20)

    def _build_charts_panel(self, parent):
        """Constroi o painel de graficos"""
        charts_frame = ttk.Labelframe(parent, text=" Analise Grafica ", padding=10)
        charts_frame.pack(fill=BOTH, expand=YES, pady=10)

        # Notebook para graficos
        self.charts_notebook = ttk.Notebook(charts_frame, bootstyle="dark")
        self.charts_notebook.pack(fill=BOTH, expand=YES)

        # Aba 1: Ultimos Veiculos (Log de Eventos)
        events_tab = ttk.Frame(self.charts_notebook)
        self.charts_notebook.add(events_tab, text=" Ultimos Veiculos ")

        # Treeview para mostrar os ultimos veiculos
        columns = ('id', 'direcao', 'cor', 'hora')
        self.events_tree = ttk.Treeview(
            events_tab,
            columns=columns,
            show='headings',
            bootstyle="dark",
            height=8
        )

        self.events_tree.heading('id', text='ID')
        self.events_tree.heading('direcao', text='Direcao')
        self.events_tree.heading('cor', text='Cor')
        self.events_tree.heading('hora', text='Horario')

        self.events_tree.column('id', width=60, anchor='center')
        self.events_tree.column('direcao', width=100, anchor='center')
        self.events_tree.column('cor', width=100, anchor='center')
        self.events_tree.column('hora', width=100, anchor='center')

        self.events_tree.pack(fill=BOTH, expand=YES, padx=5, pady=5)

        # Lista para armazenar eventos
        self.vehicle_events = []

        # Aba 2: Distribuicao por cor
        color_tab = ttk.Frame(self.charts_notebook)
        self.charts_notebook.add(color_tab, text=" Cores ")

        self.fig_colors = Figure(figsize=(6, 3.5), dpi=80, facecolor='#222222')
        self.fig_colors.subplots_adjust(left=0.05, right=0.95, top=0.88, bottom=0.05)
        self.ax_colors = self.fig_colors.add_subplot(111, facecolor='#222222')
        self.ax_colors.set_title("Distribuicao por Cor", color='white', fontsize=11, fontweight='bold')

        self.canvas_colors = FigureCanvasTkAgg(self.fig_colors, color_tab)
        self.canvas_colors.get_tk_widget().pack(fill=BOTH, expand=YES)

        # Aba 3: Entrada vs Saida (barras)
        compare_tab = ttk.Frame(self.charts_notebook)
        self.charts_notebook.add(compare_tab, text=" Comparativo ")

        self.fig_compare = Figure(figsize=(6, 3.5), dpi=80, facecolor='#222222')
        self.fig_compare.subplots_adjust(left=0.15, right=0.95, top=0.88, bottom=0.12)
        self.ax_compare = self.fig_compare.add_subplot(111, facecolor='#2d2d2d')
        self.ax_compare.set_title("Entrada vs Saida", color='white', fontsize=11, fontweight='bold')
        self.ax_compare.tick_params(colors='white', labelsize=9)
        for spine in self.ax_compare.spines.values():
            spine.set_color('#444444')

        self.canvas_compare = FigureCanvasTkAgg(self.fig_compare, compare_tab)
        self.canvas_compare.get_tk_widget().pack(fill=BOTH, expand=YES)

    def _build_alerts_panel(self, parent):
        """Constroi o painel de alertas"""
        alerts_frame = ttk.Labelframe(parent, text=" Log de Eventos ", padding=10)
        alerts_frame.pack(fill=X, pady=5)

        # Lista de alertas
        self.alerts_text = tk.Text(
            alerts_frame,
            height=5,
            bg='#1a1a1a',
            fg='#ffffff',
            font=("Consolas", 10),
            state=DISABLED,
            relief='flat'
        )
        self.alerts_text.pack(fill=X, expand=YES)

        # Tags de cores
        self.alerts_text.tag_config('warning', foreground='#f39c12')
        self.alerts_text.tag_config('danger', foreground='#e74c3c')
        self.alerts_text.tag_config('info', foreground='#3498db')
        self.alerts_text.tag_config('success', foreground='#27ae60')

    def _update_clock(self):
        """Atualiza o relogio"""
        now = datetime.now().strftime("%d/%m/%Y  %H:%M:%S")
        self.clock_label.config(text=now)
        self.root.after(1000, self._update_clock)

    def _on_line_position_change(self, value):
        """Atualiza quando a posicao da linha muda"""
        pos = float(value)
        self.line_position_label.config(text=f"{int(pos * 100)}%")

        if self.counter and hasattr(self.counter, 'frame_height'):
            self.counter.line_y = int(self.counter.frame_height * pos)
            self.counter.zone_margin = int(self.counter.frame_height * 0.05)

    def _on_video_scale_change(self, value):
        """Ajusta escala do video"""
        scale = float(value)
        if hasattr(self, 'video_scale_label'):
            self.video_scale_label.config(text=f"{int(scale * 100)}%")

    def _on_chart_scale_change(self, value):
        """Ajusta escala dos graficos"""
        scale = float(value)
        if hasattr(self, 'chart_scale_label'):
            self.chart_scale_label.config(text=f"{int(scale * 100)}%")

        # Atualizar DPI dos graficos
        new_dpi = int(80 * scale)
        self.fig_types.set_dpi(new_dpi)
        self.fig_colors.set_dpi(new_dpi)
        self.fig_compare.set_dpi(new_dpi)

        self.canvas_types.draw()
        self.canvas_colors.draw()
        self.canvas_compare.draw()

    def _on_panel_width_change(self, value):
        """Ajusta largura do painel direito"""
        width = int(float(value))
        if hasattr(self, 'panel_width_label'):
            self.panel_width_label.config(text=f"{width}px")
        self.right_panel.config(width=width)

    def _open_settings(self):
        """Abre janela de configuracoes"""
        settings_win = ttk.Toplevel(self.root)
        settings_win.title("Ajustes de Visualizacao")
        settings_win.geometry("400x300")
        settings_win.resizable(False, False)

        # Frame principal
        main_frame = ttk.Frame(settings_win, padding=20)
        main_frame.pack(fill=BOTH, expand=YES)

        ttk.Label(
            main_frame,
            text="Ajustes de Visualizacao",
            font=("Segoe UI", 14, "bold")
        ).pack(pady=(0, 20))

        # Tamanho do Video
        video_frame = ttk.Frame(main_frame)
        video_frame.pack(fill=X, pady=10)

        ttk.Label(video_frame, text="Tamanho do Video:", width=20).pack(side=LEFT)

        self.video_scale_label = ttk.Label(video_frame, text="100%", width=6, font=("Segoe UI", 10, "bold"))
        self.video_scale_label.pack(side=RIGHT)

        video_scale = ttk.Scale(
            video_frame,
            from_=0.5,
            to=1.5,
            variable=self.video_scale_var,
            bootstyle="info",
            command=self._on_video_scale_change,
            length=150
        )
        video_scale.pack(side=RIGHT, padx=10)

        # Tamanho dos Graficos
        chart_frame = ttk.Frame(main_frame)
        chart_frame.pack(fill=X, pady=10)

        ttk.Label(chart_frame, text="Tamanho dos Graficos:", width=20).pack(side=LEFT)

        self.chart_scale_label = ttk.Label(chart_frame, text="100%", width=6, font=("Segoe UI", 10, "bold"))
        self.chart_scale_label.pack(side=RIGHT)

        chart_scale = ttk.Scale(
            chart_frame,
            from_=0.5,
            to=1.5,
            variable=self.chart_scale_var,
            bootstyle="warning",
            command=self._on_chart_scale_change,
            length=150
        )
        chart_scale.pack(side=RIGHT, padx=10)

        # Largura do Painel
        panel_frame = ttk.Frame(main_frame)
        panel_frame.pack(fill=X, pady=10)

        ttk.Label(panel_frame, text="Largura Painel Direito:", width=20).pack(side=LEFT)

        self.panel_width_label = ttk.Label(panel_frame, text="550px", width=6, font=("Segoe UI", 10, "bold"))
        self.panel_width_label.pack(side=RIGHT)

        panel_scale = ttk.Scale(
            panel_frame,
            from_=400,
            to=700,
            variable=self.panel_width_var,
            bootstyle="success",
            command=self._on_panel_width_change,
            length=150
        )
        panel_scale.pack(side=RIGHT, padx=10)

        # Botao fechar
        ttk.Button(
            main_frame,
            text="Fechar",
            bootstyle="secondary",
            command=settings_win.destroy,
            width=15
        ).pack(pady=(30, 0))

    def _select_video(self):
        """Abre dialogo para selecionar video"""
        filepath = filedialog.askopenfilename(
            title="Selecionar Video",
            filetypes=[
                ("Videos", "*.mp4 *.avi *.mov *.mkv"),
                ("Todos os arquivos", "*.*")
            ]
        )
        if filepath:
            self.video_path = filepath
            self.file_label.config(text=Path(filepath).name, bootstyle="info")
            self._add_alert(f"Video selecionado: {Path(filepath).name}", "info")

    def _start_processing(self):
        """Inicia o processamento do video"""
        if not self.video_path:
            messagebox.showwarning("Aviso", "Selecione um video primeiro!")
            return

        self.is_running = True
        self.is_paused = False

        self.start_btn.config(state=DISABLED)
        self.pause_btn.config(state=NORMAL)
        self.stop_btn.config(state=NORMAL)

        self.processing_thread = threading.Thread(target=self._process_video, daemon=True)
        self.processing_thread.start()

        self._add_alert("Processamento iniciado", "success")

    def _toggle_pause(self):
        """Alterna pausa"""
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_btn.config(text="CONTINUAR", bootstyle="success")
            self._add_alert("Processamento pausado", "warning")
        else:
            self.pause_btn.config(text="PAUSAR", bootstyle="warning")
            self._add_alert("Processamento retomado", "info")

    def _stop_processing(self):
        """Para o processamento"""
        self.is_running = False
        self.is_paused = False

        self.start_btn.config(state=NORMAL)
        self.pause_btn.config(state=DISABLED, text="PAUSAR", bootstyle="warning")
        self.stop_btn.config(state=DISABLED)

        if self.cap:
            self.cap.release()
            self.cap = None

        # Limpar TreeView de eventos
        for item in self.events_tree.get_children():
            self.events_tree.delete(item)

        self._add_alert("Processamento finalizado", "info")

    def _process_video(self):
        """Processa o video em thread separada"""
        try:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                raise ValueError("Erro ao abrir video")

            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

            line_pos = self.line_position_var.get()
            self.detector = VehicleDetector(model_size='n', confidence=0.5)
            self.tracker = VehicleTracker()
            self.color_classifier = ColorClassifier()
            self.counter = VehicleCounter(frame_height=height, line_position=line_pos)
            self.analytics = AdvancedAnalytics(fps=fps)

            self._add_alert(f"Linha de contagem em {int(line_pos*100)}%", "info")

            self.vehicle_colors = {}
            frame_count = 0
            start_time = time.time()

            while self.is_running:
                if self.is_paused:
                    time.sleep(0.1)
                    continue

                ret, frame = self.cap.read()
                if not ret:
                    break

                frame_count += 1
                timestamp = frame_count / fps
                progress = (frame_count / total_frames) * 100

                detections = self.detector.detect(frame)
                tracked_vehicles = self.tracker.update(detections, frame)

                for vehicle in tracked_vehicles:
                    track_id = vehicle.get('track_id', -1)
                    if track_id >= 0:
                        color = self.color_classifier.classify_with_smoothing(
                            track_id, frame, vehicle['bbox']
                        )
                        self.vehicle_colors[track_id] = color

                newly_counted = self.counter.update(tracked_vehicles, self.vehicle_colors, timestamp)

                for track_id, direction in newly_counted:
                    color = self.vehicle_colors.get(track_id, 'indefinido')
                    self._add_alert(f"Veiculo #{track_id} - {direction.upper()} - Cor: {color}", "success" if direction == "entrada" else "danger")
                    # Adicionar ao log de eventos
                    self._add_vehicle_event(track_id, direction, color)

                frame = self._draw_visualizations(frame, tracked_vehicles)

                # Calcular fluxo
                elapsed = time.time() - start_time
                stats = self.counter.get_stats()
                flow_rate = (stats['total_geral'] / elapsed) * 60 if elapsed > 0 else 0

                self.root.after(0, lambda f=frame.copy(), p=progress, fr=flow_rate: self._update_ui(f, p, fr))

                time.sleep(1 / fps)

            self._stop_processing()

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Erro", str(e)))
            self._stop_processing()

    def _draw_visualizations(self, frame, tracked_vehicles):
        """Desenha visualizacoes no frame"""
        frame = self.counter.draw_counting_line(frame)

        for vehicle in tracked_vehicles:
            bbox = vehicle['bbox']
            track_id = vehicle.get('track_id', -1)
            x1, y1, x2, y2 = [int(c) for c in bbox]

            color = self.vehicle_colors.get(track_id, 'indefinido')
            box_color = self.color_classifier.get_color_bgr(color)

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            label = f"#{track_id} {color}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - 22), (x1 + w + 10, y1), box_color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        self._draw_stats_overlay(frame)

        return frame

    def _draw_stats_overlay(self, frame):
        """Desenha overlay de estatisticas no video"""
        h, w = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 130), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        cv2.putText(frame, "SIMV - Contagem", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        stats = self.counter.get_stats()
        cv2.putText(frame, f"Entrada: {stats['total_entrada']}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Saida: {stats['total_saida']}", (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f"Total: {stats['total_geral']}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    def _update_ui(self, frame, progress, flow_rate):
        """Atualiza a interface do usuario"""
        self._display_frame(frame)

        self.progress_var.set(progress)
        self.progress_label.config(text=f"{progress:.1f}%")

        if self.counter:
            stats = self.counter.get_stats()
            self.metric_widgets['total_entrada'].config(text=str(stats['total_entrada']))
            self.metric_widgets['total_saida'].config(text=str(stats['total_saida']))
            self.metric_widgets['total_geral'].config(text=str(stats['total_geral']))
            self.metric_widgets['fluxo_minuto'].config(text=f"{flow_rate:.1f}")

            # Classificar nivel de trafego baseado no FLUXO (nao em veiculos na cena)
            if flow_rate < 5:
                level, style = "BAIXO", "success"
            elif flow_rate < 15:
                level, style = "MODERADO", "warning"
            elif flow_rate < 30:
                level, style = "INTENSO", "info"
            else:
                level, style = "MUITO INTENSO", "danger"

            self.density_label.config(text=level, bootstyle=style)

            # Atualizar dados dos graficos
            self.flow_data.append(stats['total_geral'])
            self.time_labels.append(len(self.flow_data))

        # Atualizar graficos periodicamente
        if hasattr(self, '_chart_update_counter'):
            self._chart_update_counter += 1
        else:
            self._chart_update_counter = 0

        if self._chart_update_counter % 30 == 0:
            self._update_charts()

    def _display_frame(self, frame):
        """Exibe um frame no canvas"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]

        # Tamanho base ajustavel pelo slider
        video_scale = self.video_scale_var.get()
        base_width = 1066
        base_height = 600

        max_width = int(base_width * video_scale)
        max_height = int(base_height * video_scale)

        scale = min(max_width / w, max_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        frame_resized = cv2.resize(frame_rgb, (new_w, new_h))

        img = Image.fromarray(frame_resized)
        imgtk = ImageTk.PhotoImage(image=img)

        self.video_canvas.imgtk = imgtk
        self.video_canvas.config(image=imgtk)

    def _update_charts(self):
        """Atualiza os graficos"""
        # Grafico de cores (pizza)
        self.ax_colors.clear()
        self.ax_colors.set_facecolor('#222222')
        self.ax_colors.set_title("Distribuicao por Cor", color='white', fontsize=11, fontweight='bold', pad=10)
        if self.counter:
            color_dist = self.counter.get_color_distribution()
            if color_dist:
                # Ordenar por valor
                sorted_colors = sorted(color_dist.items(), key=lambda x: x[1], reverse=True)
                colors_list = [c[0] for c in sorted_colors]
                values = [c[1] for c in sorted_colors]

                color_map = {
                    'vermelho': '#e74c3c', 'azul': '#3498db', 'verde': '#27ae60',
                    'branco': '#ecf0f1', 'preto': '#34495e', 'prata': '#bdc3c7',
                    'cinza': '#7f8c8d', 'amarelo': '#f1c40f', 'laranja': '#e67e22',
                    'roxo': '#9b59b6', 'rosa': '#fd79a8', 'indefinido': '#636e72'
                }
                pie_colors = [color_map.get(c, '#95a5a6') for c in colors_list]

                wedges, texts, autotexts = self.ax_colors.pie(
                    values,
                    labels=colors_list,
                    colors=pie_colors,
                    autopct='%1.0f%%',
                    textprops={'color': 'white', 'fontsize': 10},
                    pctdistance=0.75,
                    labeldistance=1.1
                )
                for autotext in autotexts:
                    autotext.set_fontsize(9)
                    autotext.set_fontweight('bold')
                self.ax_colors.axis('equal')
        self.canvas_colors.draw()

        # Grafico de barras entrada/saida
        self.ax_compare.clear()
        self.ax_compare.set_facecolor('#2d2d2d')
        self.ax_compare.set_title("Entrada vs Saida", color='white', fontsize=11, fontweight='bold', pad=10)
        if self.counter:
            stats = self.counter.get_stats()
            categories = ['Entrada', 'Saida']
            values = [stats['total_entrada'], stats['total_saida']]
            colors = ['#27ae60', '#e74c3c']
            bars = self.ax_compare.bar(categories, values, color=colors, width=0.4)

            max_val = max(values) if values else 1
            for bar, val in zip(bars, values):
                self.ax_compare.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max_val * 0.02,
                    str(val),
                    ha='center',
                    color='white',
                    fontsize=16,
                    fontweight='bold'
                )

        self.ax_compare.tick_params(colors='white', labelsize=11)
        self.ax_compare.set_ylabel("Quantidade", color='#888888', fontsize=10)
        for spine in self.ax_compare.spines.values():
            spine.set_color('#444444')
        self.canvas_compare.draw()

    def _add_vehicle_event(self, track_id, direction, color):
        """Adiciona um evento de veiculo na TreeView"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        direcao_display = "ENTRADA" if direction == "entrada" else "SAIDA"

        # Inserir no inicio da TreeView
        self.events_tree.insert('', 0, values=(
            f"#{track_id}",
            direcao_display,
            color.upper(),
            timestamp
        ))

        # Manter apenas os ultimos 50 eventos
        children = self.events_tree.get_children()
        if len(children) > 50:
            self.events_tree.delete(children[-1])

    def _add_alert(self, message, severity='info'):
        """Adiciona um alerta ao painel"""
        self.alerts_text.config(state=NORMAL)
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.alerts_text.insert('1.0', f"[{timestamp}] {message}\n", severity)
        self.alerts_text.config(state=DISABLED)

    def _export_report(self):
        """Exporta relatorio completo"""
        if not self.counter:
            messagebox.showwarning("Aviso", "Nenhum dado para exportar!")
            return

        filepath = filedialog.asksaveasfilename(
            title="Salvar Relatorio",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("Todos", "*.*")]
        )

        if filepath:
            stats = self.counter.get_stats()

            report = {
                'data_geracao': datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                'video_analisado': str(self.video_path),
                'contagem': {
                    'total_entrada': stats['total_entrada'],
                    'total_saida': stats['total_saida'],
                    'total_geral': stats['total_geral']
                },
                'distribuicao_cores': self.counter.get_color_distribution()
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            messagebox.showinfo("Sucesso", f"Relatorio exportado com sucesso!\n\n{filepath}")
            self._add_alert(f"Relatorio exportado: {Path(filepath).name}", "success")

    def run(self):
        """Inicia a aplicacao"""
        self.root.mainloop()


def main():
    app = SIMVDashboard()
    app.run()


if __name__ == '__main__':
    main()
