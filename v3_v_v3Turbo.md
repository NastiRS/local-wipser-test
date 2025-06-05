| Característica                | large-v3                           | large-v3-turbo                    |
|---------------------------    |----------------------------------  |-----------------------------------|
| **Parámetros**                | 1 550 M                            | 809 M                             |
| **Capas de decodificación**   | 32                                 | 4                                 |
| **VRAM requerida**            | ≈ 10 GB (fp16)                     | ≈ 4 GB (int8)                     |
| **Velocidad vs large-v3**     | 1× (baseline)                      | 3.5×–6× más rápido                |
| **Precisión (1–10)**          | **10 / 10**                        | **7.5 / 10**                        |
| **Uso óptimo**                | Exactitud es esencial              | Bajo presupuesto y baja latencia  |


Para usar la GPU es necesario instalar CUDA Toolkit , cuDNN Library , Pytorch y ffmpeg
