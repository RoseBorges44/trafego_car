"""
SIMV Dashboard V2 - Design Profissional
Sistema Inteligente de Monitoramento Veicular

Layout baseado na apresentacao IBM Smart Cities
Design limpo e moderno para apresentacao executiva
Inclui sistema de alerta por cor e integracao com Telegram
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
import urllib.request
import urllib.parse
from pathlib import Path
from datetime import datetime


try:
    from fpdf import FPDF
    FPDF_DISPONIVEL = True
except ImportError:
    FPDF_DISPONIVEL = False
    print("Aviso: fpdf2 nao instalado. Execute: pip install fpdf2")

from src.detector import VehicleDetector
from src.tracker import VehicleTracker
from src.color_classifier import ColorClassifier
from src.counter import VehicleCounter
from src.analytics import AdvancedAnalytics


class TelegramBot:
    """Classe para enviar mensagens via Telegram"""

    def __init__(self, token=None, chat_id=None):
        self.token = token
        self.chat_id = chat_id

    def configurar(self, token, chat_id):
        """Configura o bot"""
        self.token = token
        self.chat_id = chat_id

    def enviar_mensagem(self, mensagem):
        """Envia mensagem para o Telegram"""
        if not self.token or not self.chat_id:
            print("Telegram nao configurado!")
            return False

        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            dados = {
                'chat_id': self.chat_id,
                'text': mensagem,
                'parse_mode': 'HTML'
            }
            dados_encoded = urllib.parse.urlencode(dados).encode('utf-8')
            req = urllib.request.Request(url, data=dados_encoded)
            urllib.request.urlopen(req, timeout=5)
            return True
        except Exception as e:
            print(f"Erro ao enviar Telegram: {e}")
            return False


class SIMVDashboardV2:
    """Dashboard V2 - Design Profissional baseado na apresentacao"""

    def __init__(self):
        # Janela principal
        self.root = ttk.Window(
            title="SIMV - Sistema Inteligente de Monitoramento Veicular",
            themename="superhero",
            size=(1920, 1000),
            minsize=(1600, 900)
        )
        self.root.place_window_center()

        # Cor de fundo escura
        self.bg_color = '#1a1a2e'
        self.card_color = '#16213e'
        self.accent_color = '#0f3460'

        # Configurar estilo
        self.style = ttk.Style()
        self.root.configure(bg=self.bg_color)

        # Variaveis de controle
        self.is_running = False
        self.is_paused = False
        self.video_path = None
        self.cap = None

        # Modulos de processamento
        self.detector = None
        self.tracker = None
        self.color_classifier = None
        self.counter = None
        self.analytics = None

        # Cores do veiculo
        self.vehicle_colors = {}

        # Thread de processamento
        self.processing_thread = None

        # Posicao da linha
        self.line_position = 0.5

        # Zoom do video
        self.zoom_level = 1.0

        # Sistema de Alerta
        self.cor_alerta = None
        self.alerta_ativo = False
        self.alerta_flash = False
        self.ultimo_alerta_enviado = 0  # Timestamp para evitar spam

        # Telegram Bot
        self.telegram = TelegramBot()

        # Controle de fluxo de trafego
        self.tempo_inicio_processamento = None
        self.contagem_ultimo_minuto = []  # Lista de timestamps de contagem

        # Construir interface
        self._build_ui()

    def _build_ui(self):
        """Constroi a interface do usuario"""
        # Container principal
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=BOTH, expand=YES, padx=10, pady=10)

        # Header
        self._build_header(main_frame)

        # Container de conteudo
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=BOTH, expand=YES, pady=(10, 0))

        # Painel esquerdo - Video
        left_panel = ttk.Frame(content_frame)
        left_panel.pack(side=LEFT, fill=BOTH, expand=YES, padx=(0, 10))
        self._build_video_panel(left_panel)

        # Painel direito - Stats, Graficos, Log
        right_panel = ttk.Frame(content_frame, width=380)
        right_panel.pack(side=RIGHT, fill=BOTH)
        right_panel.pack_propagate(False)

        self._build_stats_panel(right_panel)
        self._build_control_panel(right_panel)
        self._build_alert_panel(right_panel)
        self._build_charts_panel(right_panel)
        self._build_log_panel(right_panel)

    def _build_header(self, parent):
        """Header minimalista"""
        header = ttk.Frame(parent)
        header.pack(fill=X, pady=(0, 10))

        # Logo SIMV
        logo_frame = ttk.Frame(header)
        logo_frame.pack(side=LEFT)

        ttk.Label(
            logo_frame,
            text="SIMV",
            font=("Segoe UI", 28, "bold"),
            foreground="#4da6ff"
        ).pack(side=LEFT)

        ttk.Label(
            logo_frame,
            text="  Sistema Inteligente de Monitoramento Veicular",
            font=("Segoe UI", 12),
            foreground="#888888"
        ).pack(side=LEFT, pady=(10, 0))

        # Info direita
        info_frame = ttk.Frame(header)
        info_frame.pack(side=RIGHT)

        self.clock_label = ttk.Label(
            info_frame,
            text="",
            font=("Segoe UI", 11),
            foreground="#888888"
        )
        self.clock_label.pack(side=RIGHT)
        self._update_clock()

    def _build_video_panel(self, parent):
        """Painel de video - Live Feed"""
        video_frame = ttk.Labelframe(parent, text=" Live Feed ", padding=10)
        video_frame.pack(fill=BOTH, expand=YES)

        # Canvas para video
        self.video_canvas = ttk.Label(video_frame)
        self.video_canvas.pack(fill=BOTH, expand=YES)

        # Placeholder
        self._show_placeholder()

        # Controle de arquivo e acoes
        file_frame = ttk.Frame(video_frame)
        file_frame.pack(fill=X, pady=(10, 0))

        self.file_label = ttk.Label(
            file_frame,
            text="Nenhum video selecionado",
            font=("Segoe UI", 10),
            foreground="#666666"
        )
        self.file_label.pack(side=LEFT, fill=X, expand=YES)

        # Botoes de acao na mesma linha
        ttk.Button(
            file_frame,
            text="Selecionar Video",
            bootstyle="info-outline",
            command=self._select_video,
            width=14
        ).pack(side=LEFT, padx=(10, 5))

        ttk.Button(
            file_frame,
            text="Exportar PDF",
            bootstyle="success-outline",
            command=self._exportar_pdf,
            width=12
        ).pack(side=LEFT, padx=5)

        ttk.Button(
            file_frame,
            text="Telegram",
            bootstyle="warning-outline",
            command=self._abrir_config_telegram,
            width=10
        ).pack(side=LEFT, padx=5)

    def _show_placeholder(self):
        """Mostra placeholder quando nao ha video"""
        placeholder = np.zeros((720, 1280, 3), dtype=np.uint8)
        placeholder[:] = (30, 30, 45)

        # Texto central
        cv2.putText(placeholder, "SIMV - Live Feed",
                    (480, 340), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (80, 80, 100), 2)
        cv2.putText(placeholder, "Selecione um video para iniciar",
                    (450, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 60, 80), 1)

        self._display_frame(placeholder)

    def _build_stats_panel(self, parent):
        """Painel de estatisticas - ENTRADA, SAIDA, TOTAL"""
        stats_frame = ttk.Frame(parent)
        stats_frame.pack(fill=X, pady=(0, 10))

        # Grid horizontal
        for i, (key, label, color) in enumerate([
            ("entrada", "ENTRADA", "#27ae60"),
            ("saida", "SAIDA", "#e74c3c"),
            ("total", "TOTAL", "#3498db")
        ]):
            frame = ttk.Frame(stats_frame)
            frame.pack(side=LEFT, expand=YES, fill=X, padx=5)

            ttk.Label(
                frame,
                text=label,
                font=("Segoe UI", 10, "bold"),
                foreground="#888888"
            ).pack()

            lbl = ttk.Label(
                frame,
                text="0",
                font=("Segoe UI", 36, "bold"),
                foreground=color
            )
            lbl.pack()
            setattr(self, f"stat_{key}", lbl)

    def _build_control_panel(self, parent):
        """Painel de controle"""
        control_frame = ttk.Labelframe(parent, text=" Control ", padding=10)
        control_frame.pack(fill=X, pady=(0, 10))

        # Botoes em coluna
        self.start_btn = ttk.Button(
            control_frame,
            text="INICIAR",
            bootstyle="success",
            command=self._start_processing,
            width=20
        )
        self.start_btn.pack(fill=X, pady=2)

        self.pause_btn = ttk.Button(
            control_frame,
            text="PAUSAR",
            bootstyle="info",
            command=self._toggle_pause,
            width=20,
            state=DISABLED
        )
        self.pause_btn.pack(fill=X, pady=2)

        self.stop_btn = ttk.Button(
            control_frame,
            text="PARAR",
            bootstyle="danger",
            command=self._stop_processing,
            width=20,
            state=DISABLED
        )
        self.stop_btn.pack(fill=X, pady=2)

        # Separador
        ttk.Separator(control_frame).pack(fill=X, pady=10)

        # Linha de contagem
        line_frame = ttk.Frame(control_frame)
        line_frame.pack(fill=X)

        ttk.Label(line_frame, text="Linha:", font=("Segoe UI", 9)).pack(side=LEFT)

        self.line_var = ttk.DoubleVar(value=0.5)
        ttk.Scale(
            line_frame,
            from_=0.2,
            to=0.8,
            variable=self.line_var,
            bootstyle="info",
            command=self._on_line_change,
            length=120
        ).pack(side=LEFT, padx=5)

        self.line_label = ttk.Label(line_frame, text="50%", font=("Segoe UI", 9, "bold"))
        self.line_label.pack(side=LEFT)

        # Zoom do video
        zoom_frame = ttk.Frame(control_frame)
        zoom_frame.pack(fill=X, pady=(5, 0))

        ttk.Label(zoom_frame, text="Zoom:", font=("Segoe UI", 9)).pack(side=LEFT)

        self.zoom_var = ttk.DoubleVar(value=1.0)
        ttk.Scale(
            zoom_frame,
            from_=1.0,
            to=3.0,
            variable=self.zoom_var,
            bootstyle="warning",
            command=self._on_zoom_change,
            length=120
        ).pack(side=LEFT, padx=5)

        self.zoom_label = ttk.Label(zoom_frame, text="1.0x", font=("Segoe UI", 9, "bold"))
        self.zoom_label.pack(side=LEFT)

    def _build_alert_panel(self, parent):
        """Painel de configuracao de alerta"""
        alert_frame = ttk.Labelframe(parent, text=" Sistema de Alerta ", padding=10)
        alert_frame.pack(fill=X, pady=(0, 10))

        # Selecao de cor de alerta
        cor_frame = ttk.Frame(alert_frame)
        cor_frame.pack(fill=X, pady=(0, 5))

        ttk.Label(cor_frame, text="Cor Alerta:", font=("Segoe UI", 9)).pack(side=LEFT)

        self.cor_alerta_var = ttk.StringVar(value="Nenhuma")
        cores = ["Nenhuma", "vermelho", "azul", "preto", "branco", "prata", "cinza", "verde", "amarelo", "laranja"]

        self.cor_combo = ttk.Combobox(
            cor_frame,
            textvariable=self.cor_alerta_var,
            values=cores,
            state="readonly",
            width=12
        )
        self.cor_combo.pack(side=LEFT, padx=5)
        self.cor_combo.bind('<<ComboboxSelected>>', self._on_cor_alerta_change)

        # Indicador de alerta
        self.alerta_indicator = ttk.Label(
            cor_frame,
            text="",
            font=("Segoe UI", 9, "bold"),
            foreground="#888888"
        )
        self.alerta_indicator.pack(side=LEFT, padx=10)

    def _on_cor_alerta_change(self, event=None):
        """Callback quando cor de alerta muda"""
        cor = self.cor_alerta_var.get()
        if cor == "Nenhuma":
            self.cor_alerta = None
            self.alerta_indicator.config(text="", foreground="#888888")
        else:
            self.cor_alerta = cor.lower()
            self.alerta_indicator.config(text=f"MONITORANDO", foreground="#f39c12")
            self._add_log(f"Alerta ativado: {cor.upper()}")

    def _abrir_config_telegram(self):
        """Abre janela de configuracao do Telegram"""
        config_win = ttk.Toplevel(self.root)
        config_win.title("Configurar Telegram")
        config_win.geometry("550x450")
        config_win.resizable(True, True)
        config_win.minsize(500, 400)

        main = ttk.Frame(config_win, padding=20)
        main.pack(fill=BOTH, expand=YES)

        ttk.Label(
            main,
            text="Configuracao do Telegram",
            font=("Segoe UI", 14, "bold")
        ).pack(pady=(0, 10))

        ttk.Label(
            main,
            text="Para receber alertas no Telegram:",
            font=("Segoe UI", 10)
        ).pack(anchor=W)

        instrucoes = """
1. Abra o Telegram e procure @BotFather
2. Envie /newbot e siga as instrucoes
3. Copie o TOKEN do bot
4. Adicione o bot ao grupo da GCM
5. Obtenha o Chat ID do grupo
        """
        ttk.Label(main, text=instrucoes, font=("Segoe UI", 9), foreground="#888888").pack(anchor=W)

        # Token
        ttk.Label(main, text="Token do Bot:", font=("Segoe UI", 11, "bold")).pack(anchor=W, pady=(15, 5))
        self.token_entry = ttk.Entry(main, width=60, font=("Consolas", 11))
        self.token_entry.pack(fill=X, pady=(0, 15), ipady=8)
        if self.telegram.token:
            self.token_entry.insert(0, self.telegram.token)

        # Chat ID
        ttk.Label(main, text="Chat ID (grupo ou usuario):", font=("Segoe UI", 11, "bold")).pack(anchor=W, pady=(5, 5))
        self.chatid_entry = ttk.Entry(main, width=60, font=("Consolas", 11))
        self.chatid_entry.pack(fill=X, pady=(0, 15), ipady=8)
        if self.telegram.chat_id:
            self.chatid_entry.insert(0, self.telegram.chat_id)

        # Botoes
        btn_frame = ttk.Frame(main)
        btn_frame.pack(fill=X, pady=(20, 0))

        ttk.Button(
            btn_frame,
            text="Testar Envio",
            bootstyle="info",
            command=lambda: self._testar_telegram(config_win),
            width=15
        ).pack(side=LEFT, padx=(0, 10), ipady=5)

        ttk.Button(
            btn_frame,
            text="Salvar",
            bootstyle="success",
            command=lambda: self._salvar_telegram(config_win),
            width=15
        ).pack(side=LEFT, ipady=5)

        ttk.Button(
            btn_frame,
            text="Cancelar",
            bootstyle="secondary",
            command=config_win.destroy,
            width=15
        ).pack(side=RIGHT, ipady=5)

    def _testar_telegram(self, win):
        """Testa envio de mensagem"""
        token = self.token_entry.get().strip()
        chat_id = self.chatid_entry.get().strip()

        if not token or not chat_id:
            messagebox.showwarning("Aviso", "Preencha Token e Chat ID!")
            return

        self.telegram.configurar(token, chat_id)
        mensagem = "SIMV - Teste de conexao OK!\nSistema de alerta configurado com sucesso."

        if self.telegram.enviar_mensagem(mensagem):
            messagebox.showinfo("Sucesso", "Mensagem enviada com sucesso!")
        else:
            messagebox.showerror("Erro", "Falha ao enviar mensagem.\nVerifique Token e Chat ID.")

    def _salvar_telegram(self, win):
        """Salva configuracao do Telegram"""
        token = self.token_entry.get().strip()
        chat_id = self.chatid_entry.get().strip()

        self.telegram.configurar(token, chat_id)
        self._add_log("Telegram configurado!")
        win.destroy()

    def _exportar_pdf(self):
        """Exporta relatorio em PDF"""
        if not FPDF_DISPONIVEL:
            messagebox.showerror("Erro", "Biblioteca fpdf2 nao instalada!\nExecute: pip install fpdf2")
            return

        if not self.counter:
            messagebox.showwarning("Aviso", "Nenhum dado para exportar!\nProcesse um video primeiro.")
            return

        # Solicitar local para salvar
        filepath = filedialog.asksaveasfilename(
            title="Salvar Relatorio PDF",
            defaultextension=".pdf",
            initialfile=f"SIMV_Relatorio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            filetypes=[("PDF", "*.pdf")]
        )

        if not filepath:
            return

        try:
            self._gerar_pdf(filepath)
            messagebox.showinfo("Sucesso", f"Relatorio PDF gerado com sucesso!\n\n{filepath}")
            self._add_log(f"PDF exportado: {Path(filepath).name}")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao gerar PDF:\n{str(e)}")

    def _gerar_pdf(self, filepath):
        """Gera o arquivo PDF com o relatorio"""
        stats = self.counter.get_stats()
        color_dist = self.counter.get_color_distribution()

        # Criar PDF
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # Cabecalho
        pdf.set_fill_color(26, 26, 46)
        pdf.rect(0, 0, 210, 45, 'F')

        # Logo SIMV
        pdf.set_font('Helvetica', 'B', 28)
        pdf.set_text_color(77, 166, 255)
        pdf.set_xy(10, 12)
        pdf.cell(0, 10, 'SIMV', ln=False)

        pdf.set_font('Helvetica', '', 12)
        pdf.set_text_color(200, 200, 200)
        pdf.set_xy(10, 25)
        pdf.cell(0, 10, 'Sistema Inteligente de Monitoramento Veicular', ln=False)

        pdf.set_font('Helvetica', '', 10)
        pdf.set_xy(10, 35)
        pdf.cell(0, 5, 'Relatorio de Monitoramento de Trafego', ln=True)

        # Logo IBM (canto superior direito)
        pdf.set_font('Helvetica', 'B', 22)
        pdf.set_text_color(255, 255, 255)
        pdf.set_xy(160, 8)
        pdf.cell(40, 12, 'IBM', 0, 0, 'C')

        pdf.set_font('Helvetica', '', 8)
        pdf.set_text_color(150, 200, 255)
        pdf.set_xy(155, 20)
        pdf.cell(50, 5, 'Smart Cities', 0, 0, 'C')

        # Data e hora
        pdf.set_font('Helvetica', 'B', 10)
        pdf.set_text_color(200, 200, 200)
        pdf.set_xy(155, 30)
        pdf.cell(50, 5, datetime.now().strftime('%d/%m/%Y'), 0, 0, 'C')
        pdf.set_font('Helvetica', '', 9)
        pdf.set_xy(155, 36)
        pdf.cell(50, 5, datetime.now().strftime('%H:%M:%S'), 0, 0, 'C')

        pdf.ln(25)

        # Informacoes do Video
        pdf.set_font('Helvetica', 'B', 14)
        pdf.set_text_color(26, 26, 46)
        pdf.cell(0, 10, 'Informacoes da Analise', ln=True)

        pdf.set_font('Helvetica', '', 11)
        pdf.set_text_color(60, 60, 60)

        if self.video_path:
            pdf.cell(0, 7, f'Video analisado: {Path(self.video_path).name}', ln=True)
        pdf.cell(0, 7, f'Data da analise: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}', ln=True)
        pdf.cell(0, 7, f'Posicao da linha de contagem: {int(self.line_position * 100)}%', ln=True)

        pdf.ln(5)

        # Estatisticas de Contagem
        pdf.set_font('Helvetica', 'B', 14)
        pdf.set_text_color(26, 26, 46)
        pdf.cell(0, 10, 'Estatisticas de Contagem', ln=True)

        # Tabela de estatisticas
        pdf.set_font('Helvetica', 'B', 11)
        pdf.set_fill_color(240, 240, 240)

        col_width = 60
        pdf.cell(col_width, 10, 'Metrica', 1, 0, 'C', True)
        pdf.cell(col_width, 10, 'Quantidade', 1, 0, 'C', True)
        pdf.cell(col_width, 10, 'Percentual', 1, 1, 'C', True)

        pdf.set_font('Helvetica', '', 11)

        total = stats['total_geral'] or 1

        # Entrada
        pdf.set_text_color(39, 174, 96)
        pdf.cell(col_width, 10, 'Entrada', 1, 0, 'C')
        pdf.cell(col_width, 10, str(stats['total_entrada']), 1, 0, 'C')
        pdf.cell(col_width, 10, f"{(stats['total_entrada']/total)*100:.1f}%", 1, 1, 'C')

        # Saida
        pdf.set_text_color(231, 76, 60)
        pdf.cell(col_width, 10, 'Saida', 1, 0, 'C')
        pdf.cell(col_width, 10, str(stats['total_saida']), 1, 0, 'C')
        pdf.cell(col_width, 10, f"{(stats['total_saida']/total)*100:.1f}%", 1, 1, 'C')

        # Total
        pdf.set_text_color(52, 152, 219)
        pdf.set_font('Helvetica', 'B', 11)
        pdf.cell(col_width, 10, 'TOTAL', 1, 0, 'C')
        pdf.cell(col_width, 10, str(stats['total_geral']), 1, 0, 'C')
        pdf.cell(col_width, 10, '100%', 1, 1, 'C')

        pdf.ln(10)

        # Distribuicao por Cor
        pdf.set_font('Helvetica', 'B', 14)
        pdf.set_text_color(26, 26, 46)
        pdf.cell(0, 10, 'Distribuicao por Cor de Veiculo', ln=True)

        if color_dist:
            pdf.set_font('Helvetica', 'B', 11)
            pdf.set_fill_color(240, 240, 240)
            pdf.set_text_color(0, 0, 0)

            pdf.cell(90, 10, 'Cor', 1, 0, 'C', True)
            pdf.cell(45, 10, 'Quantidade', 1, 0, 'C', True)
            pdf.cell(45, 10, 'Percentual', 1, 1, 'C', True)

            pdf.set_font('Helvetica', '', 11)

            # Ordenar por quantidade
            sorted_colors = sorted(color_dist.items(), key=lambda x: x[1], reverse=True)
            total_cores = sum(color_dist.values()) or 1

            cores_rgb = {
                'vermelho': (231, 76, 60), 'azul': (52, 152, 219),
                'verde': (39, 174, 96), 'branco': (100, 100, 100),
                'preto': (52, 73, 94), 'prata': (149, 165, 166),
                'cinza': (127, 140, 141), 'amarelo': (241, 196, 15),
                'laranja': (230, 126, 34), 'roxo': (155, 89, 182),
                'rosa': (253, 121, 168), 'indefinido': (99, 110, 114)
            }

            for cor, qtd in sorted_colors:
                rgb = cores_rgb.get(cor, (0, 0, 0))
                pdf.set_text_color(*rgb)
                pdf.cell(90, 8, cor.upper(), 1, 0, 'C')
                pdf.cell(45, 8, str(qtd), 1, 0, 'C')
                pdf.cell(45, 8, f"{(qtd/total_cores)*100:.1f}%", 1, 1, 'C')

        pdf.ln(10)

        # Log de Alertas
        pdf.set_font('Helvetica', 'B', 14)
        pdf.set_text_color(26, 26, 46)
        pdf.cell(0, 10, 'Sistema de Alerta', ln=True)

        pdf.set_font('Helvetica', '', 11)
        pdf.set_text_color(60, 60, 60)

        if self.cor_alerta:
            pdf.cell(0, 7, f'Cor monitorada: {self.cor_alerta.upper()}', ln=True)
        else:
            pdf.cell(0, 7, 'Nenhuma cor de alerta configurada', ln=True)

        if self.telegram.token and self.telegram.chat_id:
            pdf.cell(0, 7, 'Telegram: Configurado', ln=True)
        else:
            pdf.cell(0, 7, 'Telegram: Nao configurado', ln=True)

        pdf.ln(10)

        # Rodape
        pdf.set_y(-30)
        pdf.set_font('Helvetica', 'I', 9)
        pdf.set_text_color(150, 150, 150)
        pdf.cell(0, 5, 'SIMV - Sistema Inteligente de Monitoramento Veicular', 0, 1, 'C')
        pdf.cell(0, 5, 'Projeto IBM Smart Cities - Prefeitura Municipal', 0, 1, 'C')
        pdf.cell(0, 5, f'Relatorio gerado em {datetime.now().strftime("%d/%m/%Y as %H:%M:%S")}', 0, 1, 'C')

        # Salvar PDF
        pdf.output(filepath)

    def _disparar_alerta(self, track_id, cor, direcao):
        """Dispara alerta visual e envia Telegram"""
        agora = time.time()

        # Evitar spam (minimo 10 segundos entre alertas)
        if agora - self.ultimo_alerta_enviado < 10:
            return

        self.ultimo_alerta_enviado = agora
        self.alerta_ativo = True

        # Log
        self._add_log(f"ALERTA! ID{track_id} {cor.upper()} - {direcao}")

        # Enviar Telegram
        mensagem = f"""
<b>ALERTA SIMV - GCM</b>

Veiculo suspeito detectado!

<b>ID:</b> {track_id}
<b>Cor:</b> {cor.upper()}
<b>Direcao:</b> {direcao.upper()}
<b>Horario:</b> {datetime.now().strftime('%H:%M:%S')}
<b>Data:</b> {datetime.now().strftime('%d/%m/%Y')}

Equipe de monitoramento acionada.
        """

        # Enviar em thread separada para nao travar
        threading.Thread(target=self.telegram.enviar_mensagem, args=(mensagem,), daemon=True).start()

        # Piscar tela por 3 segundos
        self._iniciar_flash_alerta()

    def _iniciar_flash_alerta(self):
        """Inicia efeito de piscar"""
        self.alerta_flash = True
        self._flash_count = 0
        self._executar_flash()

    def _executar_flash(self):
        """Executa o efeito de flash"""
        if self._flash_count >= 6:  # 3 segundos (6 x 500ms)
            self.alerta_flash = False
            self.alerta_ativo = False
            return

        self._flash_count += 1
        self.root.after(500, self._executar_flash)

    def _build_charts_panel(self, parent):
        """Painel de informacoes de trafego"""
        traffic_frame = ttk.Labelframe(parent, text=" Analise de Trafego ", padding=10)
        traffic_frame.pack(fill=X, pady=(0, 10))

        # Tipo de Transito
        tipo_frame = ttk.Frame(traffic_frame)
        tipo_frame.pack(fill=X, pady=(0, 10))

        ttk.Label(
            tipo_frame,
            text="TIPO DE TRANSITO",
            font=("Segoe UI", 9),
            foreground="#888888"
        ).pack()

        self.tipo_transito_label = ttk.Label(
            tipo_frame,
            text="AGUARDANDO",
            font=("Segoe UI", 18, "bold"),
            foreground="#888888"
        )
        self.tipo_transito_label.pack()

        # Separador
        ttk.Separator(traffic_frame).pack(fill=X, pady=5)

        # Fluxo por minuto
        fluxo_frame = ttk.Frame(traffic_frame)
        fluxo_frame.pack(fill=X, pady=(5, 0))

        ttk.Label(
            fluxo_frame,
            text="FLUXO DE VEICULOS",
            font=("Segoe UI", 9),
            foreground="#888888"
        ).pack()

        fluxo_valor_frame = ttk.Frame(fluxo_frame)
        fluxo_valor_frame.pack()

        self.fluxo_label = ttk.Label(
            fluxo_valor_frame,
            text="0",
            font=("Segoe UI", 28, "bold"),
            foreground="#3498db"
        )
        self.fluxo_label.pack(side=LEFT)

        ttk.Label(
            fluxo_valor_frame,
            text=" veic/min",
            font=("Segoe UI", 12),
            foreground="#888888"
        ).pack(side=LEFT, pady=(10, 0))

    def _build_log_panel(self, parent):
        """Painel de log de eventos"""
        log_frame = ttk.Labelframe(parent, text=" Event Log ", padding=5)
        log_frame.pack(fill=BOTH, expand=YES)

        # Treeview para eventos
        columns = ('hora', 'evento')
        self.log_tree = ttk.Treeview(
            log_frame,
            columns=columns,
            show='headings',
            height=12,
            bootstyle="dark"
        )

        self.log_tree.heading('hora', text='Hora')
        self.log_tree.heading('evento', text='Evento')
        self.log_tree.column('hora', width=70, anchor='center')
        self.log_tree.column('evento', width=300)

        # Scrollbar
        scrollbar = ttk.Scrollbar(log_frame, orient=VERTICAL, command=self.log_tree.yview)
        self.log_tree.configure(yscrollcommand=scrollbar.set)

        self.log_tree.pack(side=LEFT, fill=BOTH, expand=YES)
        scrollbar.pack(side=RIGHT, fill=Y)

    def _update_clock(self):
        """Atualiza o relogio"""
        now = datetime.now().strftime("%d/%m/%Y  %H:%M:%S")
        self.clock_label.config(text=now)
        self.root.after(1000, self._update_clock)

    def _on_line_change(self, value):
        """Callback para mudanca da linha"""
        self.line_position = float(value)
        self.line_label.config(text=f"{int(self.line_position * 100)}%")

        if self.counter:
            self.counter.line_y = int(self.counter.frame_height * self.line_position)

    def _on_zoom_change(self, value):
        """Callback para mudanca do zoom"""
        self.zoom_level = float(value)
        self.zoom_label.config(text=f"{self.zoom_level:.1f}x")

    def _apply_zoom(self, frame, zoom_level):
        """Aplica zoom no frame (crop central e redimensiona)"""
        h, w = frame.shape[:2]

        # Calcular tamanho da regiao de crop
        crop_w = int(w / zoom_level)
        crop_h = int(h / zoom_level)

        # Calcular coordenadas do centro
        x1 = (w - crop_w) // 2
        y1 = (h - crop_h) // 2
        x2 = x1 + crop_w
        y2 = y1 + crop_h

        # Crop da regiao central
        cropped = frame[y1:y2, x1:x2]

        # Redimensionar de volta ao tamanho original
        zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

        return zoomed

    def _select_video(self):
        """Seleciona video"""
        filepath = filedialog.askopenfilename(
            title="Selecionar Video",
            filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv"), ("Todos", "*.*")]
        )
        if filepath:
            self.video_path = filepath
            self.file_label.config(text=Path(filepath).name)
            self._add_log(f"Video: {Path(filepath).name}")

    def _start_processing(self):
        """Inicia processamento"""
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

        self._add_log("Processamento iniciado")

    def _toggle_pause(self):
        """Alterna pausa"""
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_btn.config(text="CONTINUAR", bootstyle="success")
            self._add_log("Pausado")
        else:
            self.pause_btn.config(text="PAUSAR", bootstyle="info")
            self._add_log("Retomado")

    def _stop_processing(self):
        """Para processamento"""
        self.is_running = False
        self.is_paused = False

        self.start_btn.config(state=NORMAL)
        self.pause_btn.config(state=DISABLED, text="PAUSAR", bootstyle="info")
        self.stop_btn.config(state=DISABLED)

        if self.cap:
            self.cap.release()
            self.cap = None

        self._add_log("Processamento finalizado")

    def _process_video(self):
        """Processa o video"""
        try:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                raise ValueError("Erro ao abrir video")

            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30

            self.detector = VehicleDetector(model_size='n', confidence=0.5)
            self.tracker = VehicleTracker()
            self.color_classifier = ColorClassifier()
            self.counter = VehicleCounter(frame_height=height, line_position=self.line_position)
            self.analytics = AdvancedAnalytics(fps=fps)

            self.vehicle_colors = {}
            frame_count = 0
            start_time = time.time()
            self.tempo_inicio_processamento = start_time
            self.contagem_ultimo_minuto = []

            while self.is_running:
                if self.is_paused:
                    time.sleep(0.1)
                    continue

                ret, frame = self.cap.read()
                if not ret:
                    break

                frame_count += 1
                timestamp = frame_count / fps

                # Aplicar zoom (crop central)
                if self.zoom_level > 1.0:
                    frame = self._apply_zoom(frame, self.zoom_level)

                # Deteccao e tracking
                detections = self.detector.detect(frame)
                tracked_vehicles = self.tracker.update(detections, frame)

                # Classificacao de cores
                for vehicle in tracked_vehicles:
                    track_id = vehicle.get('track_id', -1)
                    if track_id >= 0:
                        color = self.color_classifier.classify_with_smoothing(
                            track_id, frame, vehicle['bbox']
                        )
                        self.vehicle_colors[track_id] = color

                # Contagem
                newly_counted = self.counter.update(tracked_vehicles, self.vehicle_colors, timestamp)

                # Log de eventos e verificar alertas
                for track_id, direction in newly_counted:
                    color = self.vehicle_colors.get(track_id, 'indefinido')
                    dir_text = "entrada" if direction == "entrada" else "saida"
                    self._add_log(f"ID{track_id} ({color}) - {dir_text}")

                    # Registrar timestamp da contagem para calculo de fluxo
                    self.contagem_ultimo_minuto.append(time.time())

                    # Verificar se cor corresponde ao alerta
                    if self.cor_alerta and color.lower() == self.cor_alerta.lower():
                        self.root.after(0, lambda tid=track_id, c=color, d=dir_text: self._disparar_alerta(tid, c, d))

                # Desenhar visualizacoes
                frame = self._draw_frame(frame, tracked_vehicles)

                # Atualizar UI
                self.root.after(0, lambda f=frame.copy(): self._update_ui(f))

                time.sleep(1 / fps)

            self._stop_processing()

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Erro", str(e)))
            self._stop_processing()

    def _draw_frame(self, frame, tracked_vehicles):
        """Desenha visualizacoes no frame"""
        # Linha de contagem
        frame = self.counter.draw_counting_line(frame)

        # Bounding boxes
        for vehicle in tracked_vehicles:
            bbox = vehicle['bbox']
            track_id = vehicle.get('track_id', -1)
            x1, y1, x2, y2 = [int(c) for c in bbox]

            color = self.vehicle_colors.get(track_id, 'indefinido')
            box_color = self.color_classifier.get_color_bgr(color)

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            # Label
            label = f"ID{track_id}"
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + 55, y1), box_color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Overlay de ALERTA se ativo
        if self.alerta_flash and self._flash_count % 2 == 1:
            h, w = frame.shape[:2]

            # Overlay vermelho semi-transparente
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

            # Borda vermelha grossa
            cv2.rectangle(frame, (5, 5), (w-5, h-5), (0, 0, 255), 10)

            # Texto ALERTA grande
            texto = "ALERTA"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 4
            thickness = 8

            (tw, th), _ = cv2.getTextSize(texto, font, font_scale, thickness)
            x = (w - tw) // 2
            y = (h + th) // 2

            # Sombra
            cv2.putText(frame, texto, (x+4, y+4), font, font_scale, (0, 0, 0), thickness+4)
            # Texto principal
            cv2.putText(frame, texto, (x, y), font, font_scale, (255, 255, 255), thickness)

        return frame

    def _update_ui(self, frame):
        """Atualiza interface"""
        self._display_frame(frame)

        if self.counter:
            stats = self.counter.get_stats()
            self.stat_entrada.config(text=str(stats['total_entrada']))
            self.stat_saida.config(text=str(stats['total_saida']))
            self.stat_total.config(text=str(stats['total_geral']))

            # Atualizar fluxo e tipo de transito
            self._update_traffic_info()

    def _display_frame(self, frame):
        """Exibe frame"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]

        # Redimensionar mantendo proporcao - VIDEO GRANDE
        max_w, max_h = 1280, 720
        scale = min(max_w / w, max_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        frame_resized = cv2.resize(frame_rgb, (new_w, new_h))

        img = Image.fromarray(frame_resized)
        imgtk = ImageTk.PhotoImage(image=img)

        self.video_canvas.imgtk = imgtk
        self.video_canvas.config(image=imgtk)

    def _update_traffic_info(self):
        """Atualiza informacoes de fluxo e tipo de transito"""
        agora = time.time()

        # Remover contagens mais antigas que 60 segundos
        self.contagem_ultimo_minuto = [
            t for t in self.contagem_ultimo_minuto
            if agora - t <= 60
        ]

        # Calcular fluxo por minuto
        fluxo = len(self.contagem_ultimo_minuto)
        self.fluxo_label.config(text=str(fluxo))

        # Determinar tipo de transito baseado no fluxo
        if fluxo == 0:
            tipo = "LIVRE"
            cor = "#27ae60"  # Verde
        elif fluxo <= 5:
            tipo = "LEVE"
            cor = "#2ecc71"  # Verde claro
        elif fluxo <= 15:
            tipo = "MODERADO"
            cor = "#f39c12"  # Amarelo/Laranja
        elif fluxo <= 30:
            tipo = "INTENSO"
            cor = "#e67e22"  # Laranja
        else:
            tipo = "CONGESTIONADO"
            cor = "#e74c3c"  # Vermelho

        self.tipo_transito_label.config(text=tipo, foreground=cor)

    def _add_log(self, message):
        """Adiciona mensagem ao log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_tree.insert('', 0, values=(timestamp, message))

        # Manter apenas ultimos 50
        children = self.log_tree.get_children()
        if len(children) > 50:
            self.log_tree.delete(children[-1])

    def run(self):
        """Inicia aplicacao"""
        self.root.mainloop()


def main():
    app = SIMVDashboardV2()
    app.run()


if __name__ == '__main__':
    main()
