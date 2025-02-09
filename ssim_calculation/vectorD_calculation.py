import numpy as np
import matplotlib.pyplot as plt

bitrates = []
ssim_values = []
bit_ssim = [(45652, 0.828158),
            (89283, 0.852399),
            (131087, 0.865724),
            (178351, 0.876839),
            (221600, 0.88609),
            (262537, 0.892973),
            (334349, 0.902203),
            (522286, 0.925122),
            (595491, 0.930934),
            (791182, 0.93882),
            (1032682, 0.950304),
            (1244778, 0.956959),
            (1546902, 0.964165),
            (2133691, 0.971367),
            (2484135, 0.97593),
            (3526922, 0.984788)]

for tupla in bit_ssim:
    bitrates.append(tupla[0])
    ssim_values.append(tupla[1])

bitrates = np.array(bitrates)
ssim_values = np.array(ssim_values)

# Normalizar os bitrates
R1 = 4219897
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
