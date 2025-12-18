"""
SIMV - Script de Validacao de Acuracia
Sistema Inteligente de Monitoramento Veicular

Este script permite validar a acuracia do sistema comparando
a contagem automatica com a contagem manual (ground truth).
"""

import cv2
import time
import json
from pathlib import Path
from datetime import datetime
from src.detector import VehicleDetector
from src.tracker import VehicleTracker
from src.color_classifier import ColorClassifier
from src.counter import VehicleCounter


def processar_video(video_path, line_position=0.5, mostrar_video=True):
    """
    Processa o video e retorna a contagem automatica.

    Args:
        video_path: Caminho do video
        line_position: Posicao da linha de contagem (0-1)
        mostrar_video: Se True, mostra o video durante processamento

    Returns:
        dict com estatisticas de contagem
    """
    print(f"\n{'='*60}")
    print("PROCESSANDO VIDEO COM SIMV")
    print(f"{'='*60}")
    print(f"Video: {video_path}")
    print(f"Linha de contagem: {int(line_position*100)}%")
    print(f"{'='*60}\n")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Erro ao abrir video")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Resolucao: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Total de frames: {total_frames}")
    print(f"Duracao: {total_frames/fps:.1f} segundos\n")

    # Inicializar modulos
    detector = VehicleDetector(model_size='n', confidence=0.5)
    tracker = VehicleTracker()
    color_classifier = ColorClassifier()
    counter = VehicleCounter(frame_height=height, line_position=line_position)

    vehicle_colors = {}
    frame_count = 0
    start_time = time.time()

    print("Processando... (Pressione 'q' para cancelar)\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        timestamp = frame_count / fps
        progress = (frame_count / total_frames) * 100

        # Deteccao e tracking
        detections = detector.detect(frame)
        tracked_vehicles = tracker.update(detections, frame)

        # Classificacao de cores
        for vehicle in tracked_vehicles:
            track_id = vehicle.get('track_id', -1)
            if track_id >= 0:
                color = color_classifier.classify_with_smoothing(
                    track_id, frame, vehicle['bbox']
                )
                vehicle_colors[track_id] = color

        # Contagem
        newly_counted = counter.update(tracked_vehicles, vehicle_colors, timestamp)

        # Mostrar progresso
        if frame_count % 30 == 0:
            stats = counter.get_stats()
            print(f"\rProgresso: {progress:5.1f}% | Entrada: {stats['total_entrada']} | Saida: {stats['total_saida']} | Total: {stats['total_geral']}", end="")

        # Mostrar video (opcional)
        if mostrar_video:
            frame = counter.draw_counting_line(frame)

            for vehicle in tracked_vehicles:
                bbox = vehicle['bbox']
                track_id = vehicle.get('track_id', -1)
                x1, y1, x2, y2 = [int(c) for c in bbox]

                color = vehicle_colors.get(track_id, 'indefinido')
                box_color = color_classifier.get_color_bgr(color)

                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                label = f"ID{track_id}"
                cv2.putText(frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

            # Redimensionar para exibicao
            scale = 0.6
            frame_show = cv2.resize(frame, (int(width*scale), int(height*scale)))
            cv2.imshow("SIMV - Validacao (Q para sair)", frame_show)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n\nProcessamento cancelado pelo usuario!")
                break

    cap.release()
    cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    stats = counter.get_stats()

    print(f"\n\n{'='*60}")
    print("PROCESSAMENTO CONCLUIDO")
    print(f"{'='*60}")
    print(f"Tempo de processamento: {elapsed:.1f} segundos")
    print(f"Velocidade: {frame_count/elapsed:.1f} FPS")

    return stats


def calcular_metricas(contagem_sistema, contagem_manual_entrada, contagem_manual_saida):
    """
    Calcula metricas de acuracia.

    Args:
        contagem_sistema: dict com estatisticas do sistema
        contagem_manual_entrada: Contagem manual de entrada
        contagem_manual_saida: Contagem manual de saida

    Returns:
        dict com metricas calculadas
    """
    sistema_entrada = contagem_sistema['total_entrada']
    sistema_saida = contagem_sistema['total_saida']
    sistema_total = contagem_sistema['total_geral']

    manual_total = contagem_manual_entrada + contagem_manual_saida

    # Acuracia por direcao
    if contagem_manual_entrada > 0:
        acuracia_entrada = min(sistema_entrada / contagem_manual_entrada, 1.0) * 100
        erro_entrada = abs(sistema_entrada - contagem_manual_entrada)
    else:
        acuracia_entrada = 100 if sistema_entrada == 0 else 0
        erro_entrada = sistema_entrada

    if contagem_manual_saida > 0:
        acuracia_saida = min(sistema_saida / contagem_manual_saida, 1.0) * 100
        erro_saida = abs(sistema_saida - contagem_manual_saida)
    else:
        acuracia_saida = 100 if sistema_saida == 0 else 0
        erro_saida = sistema_saida

    # Acuracia total
    if manual_total > 0:
        # Calculo considerando erros (tanto para mais quanto para menos)
        erro_total = abs(sistema_total - manual_total)
        acuracia_total = max(0, (1 - erro_total / manual_total)) * 100
    else:
        acuracia_total = 100 if sistema_total == 0 else 0

    # Metricas detalhadas
    metricas = {
        'contagem_manual': {
            'entrada': contagem_manual_entrada,
            'saida': contagem_manual_saida,
            'total': manual_total
        },
        'contagem_sistema': {
            'entrada': sistema_entrada,
            'saida': sistema_saida,
            'total': sistema_total
        },
        'diferenca': {
            'entrada': sistema_entrada - contagem_manual_entrada,
            'saida': sistema_saida - contagem_manual_saida,
            'total': sistema_total - manual_total
        },
        'acuracia': {
            'entrada': round(acuracia_entrada, 1),
            'saida': round(acuracia_saida, 1),
            'total': round(acuracia_total, 1)
        },
        'erro_absoluto': {
            'entrada': erro_entrada,
            'saida': erro_saida,
            'total': abs(sistema_total - manual_total)
        }
    }

    return metricas


def exibir_relatorio(metricas, video_path):
    """Exibe relatorio formatado"""
    print(f"\n{'='*60}")
    print("RELATORIO DE VALIDACAO DE ACURACIA - SIMV")
    print(f"{'='*60}")
    print(f"Video: {Path(video_path).name}")
    print(f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"{'='*60}\n")

    print("COMPARATIVO DE CONTAGEM:")
    print("-" * 40)
    print(f"{'Metrica':<15} {'Manual':>10} {'Sistema':>10} {'Diff':>10}")
    print("-" * 40)

    m = metricas
    print(f"{'Entrada':<15} {m['contagem_manual']['entrada']:>10} {m['contagem_sistema']['entrada']:>10} {m['diferenca']['entrada']:>+10}")
    print(f"{'Saida':<15} {m['contagem_manual']['saida']:>10} {m['contagem_sistema']['saida']:>10} {m['diferenca']['saida']:>+10}")
    print(f"{'TOTAL':<15} {m['contagem_manual']['total']:>10} {m['contagem_sistema']['total']:>10} {m['diferenca']['total']:>+10}")
    print("-" * 40)

    print(f"\nACURACIA:")
    print("-" * 40)

    # Cores ANSI para terminal
    def cor_acuracia(valor):
        if valor >= 95:
            return f"\033[92m{valor}%\033[0m"  # Verde
        elif valor >= 85:
            return f"\033[93m{valor}%\033[0m"  # Amarelo
        else:
            return f"\033[91m{valor}%\033[0m"  # Vermelho

    print(f"Entrada:  {cor_acuracia(m['acuracia']['entrada'])}")
    print(f"Saida:    {cor_acuracia(m['acuracia']['saida'])}")
    print(f"\n>>> ACURACIA TOTAL: {cor_acuracia(m['acuracia']['total'])} <<<")
    print("-" * 40)

    # Avaliacao
    acuracia = m['acuracia']['total']
    print(f"\nAVALIACAO:")
    if acuracia >= 95:
        print("EXCELENTE - Sistema com alta precisao!")
    elif acuracia >= 90:
        print("BOM - Sistema com boa precisao.")
    elif acuracia >= 80:
        print("REGULAR - Considere ajustar parametros.")
    else:
        print("NECESSITA AJUSTES - Verifique posicao da linha e qualidade do video.")

    print(f"\n{'='*60}\n")

    return metricas


def salvar_relatorio(metricas, video_path, output_path=None):
    """Salva relatorio em JSON"""
    if output_path is None:
        output_path = f"validacao_{Path(video_path).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    relatorio = {
        'video': str(video_path),
        'data': datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
        'metricas': metricas
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(relatorio, f, indent=2, ensure_ascii=False)

    print(f"Relatorio salvo em: {output_path}")
    return output_path


def modo_contagem_manual(video_path):
    """
    Modo para ajudar na contagem manual.
    Permite pausar, avançar frame a frame e contar.
    """
    print(f"\n{'='*60}")
    print("MODO CONTAGEM MANUAL")
    print(f"{'='*60}")
    print("Controles:")
    print("  ESPACO  - Pausar/Continuar")
    print("  D       - Avancar 1 frame")
    print("  A       - Voltar 1 frame")
    print("  E       - Marcar ENTRADA (+1)")
    print("  S       - Marcar SAIDA (+1)")
    print("  R       - Reset contagem")
    print("  Q       - Finalizar e salvar")
    print(f"{'='*60}\n")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Erro ao abrir video")

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    contagem_entrada = 0
    contagem_saida = 0
    pausado = False
    frame_atual = 0

    while True:
        if not pausado:
            ret, frame = cap.read()
            if not ret:
                break
            frame_atual = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        # Overlay com informacoes
        display = frame.copy()
        h, w = display.shape[:2]

        # Fundo para texto
        cv2.rectangle(display, (10, 10), (350, 140), (0, 0, 0), -1)
        cv2.rectangle(display, (10, 10), (350, 140), (255, 255, 255), 2)

        # Textos
        cv2.putText(display, "CONTAGEM MANUAL", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, f"Frame: {frame_atual}/{total_frames}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(display, f"ENTRADA (E): {contagem_entrada}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, f"SAIDA (S):   {contagem_saida}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if pausado:
            cv2.putText(display, "PAUSADO", (w-150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Linha de referencia no centro
        cv2.line(display, (0, h//2), (w, h//2), (0, 255, 255), 2)

        # Redimensionar
        scale = 0.7
        display = cv2.resize(display, (int(w*scale), int(h*scale)))
        cv2.imshow("Contagem Manual - SIMV", display)

        key = cv2.waitKey(30 if not pausado else 0) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            pausado = not pausado
        elif key == ord('e'):
            contagem_entrada += 1
            print(f"ENTRADA: {contagem_entrada}")
        elif key == ord('s'):
            contagem_saida += 1
            print(f"SAIDA: {contagem_saida}")
        elif key == ord('r'):
            contagem_entrada = 0
            contagem_saida = 0
            print("Contagem resetada!")
        elif key == ord('d') and pausado:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
        elif key == ord('a') and pausado:
            pos = max(0, int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n{'='*60}")
    print("CONTAGEM MANUAL FINALIZADA")
    print(f"{'='*60}")
    print(f"Entrada: {contagem_entrada}")
    print(f"Saida:   {contagem_saida}")
    print(f"Total:   {contagem_entrada + contagem_saida}")
    print(f"{'='*60}\n")

    return contagem_entrada, contagem_saida


def main():
    """Funcao principal"""
    print("\n" + "="*60)
    print("SIMV - VALIDACAO DE ACURACIA")
    print("Sistema Inteligente de Monitoramento Veicular")
    print("="*60 + "\n")

    # Solicitar video
    video_path = input("Caminho do video (ou Enter para 'highway_mini.mp4'): ").strip()
    if not video_path:
        video_path = "highway_mini.mp4"

    if not Path(video_path).exists():
        print(f"Erro: Video '{video_path}' nao encontrado!")
        return

    # Menu
    print("\nOpcoes:")
    print("1. Fazer contagem manual primeiro (recomendado)")
    print("2. Ja tenho a contagem manual")
    print("3. Apenas processar video (sem validacao)")

    opcao = input("\nEscolha (1/2/3): ").strip()

    if opcao == "1":
        # Modo contagem manual
        entrada_manual, saida_manual = modo_contagem_manual(video_path)

        # Processar com sistema
        print("\nAgora vamos processar com o sistema automatico...")
        input("Pressione Enter para continuar...")

        line_pos = input("Posicao da linha (0.2-0.8, padrao 0.5): ").strip()
        line_pos = float(line_pos) if line_pos else 0.5

        stats = processar_video(video_path, line_position=line_pos, mostrar_video=True)

        # Calcular e exibir metricas
        metricas = calcular_metricas(stats, entrada_manual, saida_manual)
        exibir_relatorio(metricas, video_path)

        # Salvar
        if input("Salvar relatorio? (s/n): ").strip().lower() == 's':
            salvar_relatorio(metricas, video_path)

    elif opcao == "2":
        # Usuario ja tem contagem manual
        print("\nInforme a contagem manual:")
        entrada_manual = int(input("Veiculos de ENTRADA: ").strip() or "0")
        saida_manual = int(input("Veiculos de SAIDA: ").strip() or "0")

        line_pos = input("Posicao da linha (0.2-0.8, padrao 0.5): ").strip()
        line_pos = float(line_pos) if line_pos else 0.5

        mostrar = input("Mostrar video durante processamento? (s/n): ").strip().lower() == 's'

        stats = processar_video(video_path, line_position=line_pos, mostrar_video=mostrar)

        # Calcular e exibir metricas
        metricas = calcular_metricas(stats, entrada_manual, saida_manual)
        exibir_relatorio(metricas, video_path)

        # Salvar
        if input("Salvar relatorio? (s/n): ").strip().lower() == 's':
            salvar_relatorio(metricas, video_path)

    elif opcao == "3":
        # Apenas processar
        line_pos = input("Posicao da linha (0.2-0.8, padrao 0.5): ").strip()
        line_pos = float(line_pos) if line_pos else 0.5

        stats = processar_video(video_path, line_position=line_pos, mostrar_video=True)

        print("\nRESULTADO:")
        print(f"Entrada: {stats['total_entrada']}")
        print(f"Saida: {stats['total_saida']}")
        print(f"Total: {stats['total_geral']}")

    else:
        print("Opcao invalida!")

    print("\nValidacao finalizada!")


if __name__ == "__main__":
    main()
