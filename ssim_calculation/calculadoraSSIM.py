import subprocess

# Caminho dos vídeos
input_video1 = 'BigBuckBunny_4726737bps.mp4'
input_video2 = [
    'BigBuckBunny_46980bps.mp4', 'BigBuckBunny_91917bps.mp4', 'BigBuckBunny_135410bps.mp4',
    'BigBuckBunny_182366bps.mp4', 'BigBuckBunny_226106bps.mp4', 'BigBuckBunny_270316bps.mp4',
    'BigBuckBunny_352546bps.mp4', 'BigBuckBunny_424520bps.mp4', 'BigBuckBunny_537825bps.mp4',
    'BigBuckBunny_620705bps.mp4', 'BigBuckBunny_808057bps.mp4', 'BigBuckBunny_1071529bps.mp4',
    'BigBuckBunny_1312787bps.mp4', 'BigBuckBunny_1662809bps.mp4', 'BigBuckBunny_2234145bps.mp4',
    'BigBuckBunny_3305118bps.mp4', 'BigBuckBunny_3841983bps.mp4', 'BigBuckBunny_4242923bps.mp4'
]
output_ssim_file = 'ssim_result.txt'

# Função para calcular o SSIM entre dois vídeos
def calculate_ssim(video1, video2):
    # Comando FFmpeg para redimensionar o vídeo 2 e calcular o SSIM
    command = [
        'ffmpeg',
        '-i', video1,
        '-i', video2,
        '-filter_complex', '[1:v]scale=1920:1080[scaled];[0:v][scaled]ssim=stats_file=ssim_log.txt',
        '-f', 'null',
        '-'
    ]

    # Executa o comando FFmpeg
    print(f"Executando comando FFmpeg para vídeo {video2}")
    result = subprocess.run(command, stderr=subprocess.PIPE, text=True)

    # Exibe a saída do FFmpeg para depuração
    print("Saída do FFmpeg (stderr):")
    print(result.stderr)

    # Processa o arquivo de log do SSIM para extrair os valores
    ssim_values = []
    try:
        with open('ssim_log.txt', 'r') as log_file:
            for line in log_file:
                if 'All:' in line:
                    # Extrai o valor SSIM da linha
                    ssim_value = float(line.split('All:')[1].split()[0])
                    ssim_values.append(ssim_value)
                    print(f"Valor SSIM encontrado: {ssim_value}")
    except FileNotFoundError:
        print("Arquivo de log SSIM não encontrado.")
        return None

    # Calcula a média SSIM
    if len(ssim_values) > 0:
        average_ssim = sum(ssim_values) / len(ssim_values)
        print(f"Média SSIM calculada: {average_ssim}")
        return average_ssim
    else:
        print("Nenhum valor SSIM encontrado no arquivo de log.")
        return None

# Itera sobre os vídeos e calcula o SSIM
for video in input_video2:
    average_ssim = calculate_ssim(input_video1, video)
    if average_ssim is not None:
        # Grava a média SSIM no arquivo de saída
        with open(output_ssim_file, 'a') as ssim_file:
            ssim_file.write(f"({video},{average_ssim})\n")