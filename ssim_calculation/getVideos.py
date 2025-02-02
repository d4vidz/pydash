import ffmpeg


input_video1 = 'BigBuckBunny_4219897bps.mp4'
input_video2 = 'BigBuckBunny_22160bps.mp4'
output_ssim_file = 'ssim_result.txt'


ffmpeg.input(input_video1).input(input_video2).output(
    'pipe:1',  
    filter_complex="[1:v]scale=1920:1080[v2];[0:v][v2]ssim=" + output_ssim_file,  
    v='verbose', 
    f='null'  
).run()

with open('ssim_result.txt', 'r') as file:
    ssim_values = []
    for line in file:
        if "All:" in line:
            try:
                ssim_value = float(line.split('All:')[1].split()[0]) 
                ssim_values.append(ssim_value)
            except ValueError:
                print(f"Erro ao tentar converter o valor SSIM na linha: {line}")
                
if len(ssim_values) > 0:
    average_ssim = sum(ssim_values) / len(ssim_values)
    print(f"Média SSIM: {average_ssim}")
else:
    print("Nenhum valor SSIM encontrado no arquivo.")


