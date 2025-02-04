import ffmpeg
import subprocess

input_video1 = 'BigBuckBunny_4219897bps.mp4'
input_video2 = ['BigBuckBunny_45652bps.mp4', 'BigBuckBunny_89283bps.mp4', 'BigBuckBunny_131087bps.mp4', 
                'BigBuckBunny_178351bps.mp4', 'BigBuckBunny_221600bps.mp4', 'BigBuckBunny_262537bps.mp4',
                'BigBuckBunny_334349bps.mp4', 'BigBuckBunny_522286bps.mp4', 'BigBuckBunny_595491bps.mp4',
                'BigBuckBunny_791182bps.mp4', 'BigBuckBunny_1032682bps.mp4', 'BigBuckBunny_1244778bps.mp4',
                'BigBuckBunny_1546902bps.mp4', 'BigBuckBunny_2133691bps.mp4', 'BigBuckBunny_2484135bps.mp4',
                'BigBuckBunny_3526922bps.mp4']
output_ssim_file = 'ssim_result.txt'

# Comando ffmpeg para redimensionar o vídeo 2 e calcular o SSIM
for video in input_video2:
    command = [
        'ffmpeg',
        '-i', input_video1,
        '-i', video,
        '-filter_complex', '[1:v]scale=1920x1080[scaled];[0:v][scaled]ssim',  # Redimensiona e calcula SSIM
        '-f', 'null',
        '-'
    ]

    # Executa o comando e captura a saída
    print(f"Executando comando FFmpeg para vídeo {video}")
    result = subprocess.run(command, stderr=subprocess.PIPE, text=True)

    # Exibe a saída do FFmpeg para depuração
    print("Saída do FFmpeg (stderr):")
    print(result.stderr)

    # Processa a saída para extrair os valores SSIM
    ssim_values = []
    for line in result.stderr.splitlines():
        if "All:" in line:
            try:
                # Extrai o valor SSIM da linha
                ssim_value = float(line.split('All:')[1].split()[0])
                ssim_values.append(ssim_value)
                print(f"Valor SSIM encontrado: {ssim_value}")
            except ValueError:
                print(f"Erro ao tentar converter o valor SSIM na linha: {line}")

    # Calcula e grava a média SSIM no arquivo
    if len(ssim_values) > 0:
        average_ssim = sum(ssim_values) / len(ssim_values)
        print(f"Média SSIM calculada: {average_ssim}")

        # Grava a média no arquivo de saída
        with open(output_ssim_file, 'a') as ssim_file:
            ssim_file.write(f"({video},{average_ssim})\n")
    else:
        print("Nenhum valor SSIM encontrado na saída.")