import numpy as np
import matplotlib.pyplot as plt

bitrates = []
ssim_values = []
bit_ssim = [(46980,0.8221028093205515),
            (91917,0.8476104038905605),
            (135410,0.8627666132530963),
            (182366,0.872579079140718),
            (226106,0.8829015420194528),
            (270316,0.8910175927506823),
            (352546,0.9026483485410398),
            (424520,0.9101606996011475),
            (537825,0.922603339654328),
            (620705,0.9293193729620041),
            (808057,0.9344805092015954),
            (1071529,0.9473637510321181),
            (1312787,0.955839637953957),
            (1662809,0.9641462656916941),
            (2234145,0.9685280322580646),
            (3305118,0.9805357287803513),
            (3841983,0.9843367343782801),
            (4242923,0.9871164390175635)]

for tupla in bit_ssim:
    bitrates.append(tupla[0])
    ssim_values.append(tupla[1])

bitrates = np.array(bitrates)
ssim_values = np.array(ssim_values)

# Normalizar os bitrates
R1 = 4726737
normalized_bitrates = bitrates / R1
log_normalized_bitrates = np.log(normalized_bitrates)

# Ajuste polinomial de ordem 4
coeffs = np.polyfit(log_normalized_bitrates, ssim_values, 4)
reversed_coeffs = coeffs[::-1]
# Salvar os coeficientes no arquivo
with open("ssim_calculation/ssim_qualities.txt", "w") as ssim_file:
    ssim_file.write("Coeficientes do ajuste polinomial:\n")
    ssim_file.write("\n".join(map(str, reversed_coeffs[1::])))

print("Coeficientes salvos com sucesso.")

#Plotagem
plt.scatter(log_normalized_bitrates, ssim_values, label='Dados Originais')
x_vals = np.linspace(log_normalized_bitrates.min(), log_normalized_bitrates.max(), 200)
y_vals = np.polyval(coeffs, x_vals)
plt.plot(x_vals, y_vals, color='red', label='Ajuste Polinomial (ordem 4)')
plt.legend()
plt.xlabel('Logaritmo da Taxa Normalizada')
plt.ylabel('SSIM')
plt.title('Ajuste Polinomial de SSIM')
plt.show()
