import numpy as np
bitrates = []
ssim_values = []
bit_ssim = [(45652,0.828158),
(89283,0.852399),
(131087,0.865724),
(178351,0.876839),
(221600,0.88609),
(262537,0.892973),
(334349,0.902203),
(522286,0.925122),
(595491,0.930934),
(791182,0.93882),
(1032682,0.950304),
(1244778,0.956959),
(1546902,0.964165),
(2133691,0.971367),
(2484135,0.97593),
(3526922,0.984788),
]
# Dados de bitrate e SSIM calculados
for tupla in bit_ssim:
    bitrates.append(tupla[0])
    ssim_values.append(tupla[1])
bitrates = np.array(bitrates)  # bitrates
ssim_values = np.array(ssim_values)  # SSIM médio para cada bitrate

# Normalizar os bitrates (assumindo que R1 é o maior bitrate observado)
R1 = 4219897
normalized_bitrates = bitrates / R1

# Aplicar o logaritmo à taxa normalizada
log_normalized_bitrates = np.log(normalized_bitrates)

# Ajuste polinomial de 4ª ordem aos dados logaritmicamente transformados
coeffs = np.polyfit(log_normalized_bitrates, ssim_values, 19)

# Os coeficientes (d1, d2, d3, d4)
d = coeffs
print("Vetor d:", d)
