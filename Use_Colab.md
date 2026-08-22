# CONFIGURACIÓN EN COLAB

Guía para usar `acv_bolivia` en Google Colab. La estrategia es instalar el
paquete (y sus dependencias) en una carpeta local del SSD de Colab, comprimirla
y guardarla en Google Drive, para cargarla en sesiones futuras sin reinstalar.

> **Compatibilidad de binarios:** el zip contiene paquetes con extensiones
> compiladas (numpy, scipy, brightway2, xgboost, shap, etc.) para el runtime de
> Colab de la sesión en que se creó. Si Colab cambia la versión de Python de sus
> runtime, el zip puede dejar de funcionar. En ese caso, regenéralo desde la
> sección 1. Colab usa Python 3.10/3.11, compatible con el proyecto
> (`requires-python >= 3.10`, recomendado 3.11).

## 1. Crear el entorno (ejecutar solo la primera vez)

Se puede ejecutar sobre una sesión que ya tenga la dependencia de Brightway2.

```python
import os
from google.colab import drive

# 1. Montar Drive
drive.mount('/content/drive')

# 2. Definir rutas: carpeta local en el SSD de Colab y zip en Google Drive
LOCAL_ENV = '/content/env_acv_bolivia'
ZIP_DRIVE_PATH = '/content/drive/MyDrive/acv_Bolivia/env_acv_bolivia.zip'

# 3. Instalar el paquete (y sus dependencias) en el directorio local
!python -m pip install --target={LOCAL_ENV} git+https://github.com/csJorge/acv-bolivia

# 4. Comprimir la carpeta y guardarla en Google Drive
print("Comprimiendo entorno hacia Google Drive...")
!zip -r -q {ZIP_DRIVE_PATH} {LOCAL_ENV}
print("Entorno guardado exitosamente en Google Drive.")
```

Nota: `zip -r {ZIP_DRIVE_PATH} en/absolute/path` guarda en el interior del zip la
carpeta con su **último nombre** (`env_acv_bolivia`). Por eso, al descomprimir,
se debe extraer dentro de `/content` para que la ruta resultante
(`/content/env_acv_bolivia`) coincida con `LOCAL_ENV` (ver siguiente sección).

## 2. Cargar el entorno desde Drive (sesiones posteriores)

```python
import os
import sys
from google.colab import drive

# 1. Montar Drive
drive.mount('/content/drive')

# 2. Definir rutas
ZIP_DRIVE_PATH = '/content/drive/MyDrive/acv_Bolivia/env_acv_bolivia.zip'
LOCAL_ENV = '/content/env_acv_bolivia'

# 3. Copiar el zip de Drive al SSD local y descomprimirlo
if os.path.exists(ZIP_DRIVE_PATH):
    # IMPORTANTE: descomprimir dentro de /content (no en /) para que la
    # carpeta quede en /content/env_acv_bolivia y coincida con LOCAL_ENV.
    !unzip -q {ZIP_DRIVE_PATH} -d /content

    # 4. Verificar que la carpeta quedó donde se espera y activarla
    if os.path.isdir(LOCAL_ENV):
        if LOCAL_ENV not in sys.path:
            sys.path.insert(0, LOCAL_ENV)
        print("Entorno local activado y listo para usar.")
        print(f"Paquete en: {LOCAL_ENV}")
    else:
        print(f"No se encontró la carpeta del entorno en {LOCAL_ENV}.")
else:
    print("No se encontró el archivo de entorno en Google Drive.")
```

## 3. Usar el entorno directamente desde Drive (sin comprimir)

También se puede usar el entorno instalado directamente en Drive, sin pasar por
el zip. Es más lento de importar pero no requiere descomprimir ni regenerar.

```python
import os
import sys
from google.colab import drive

drive.mount('/content/drive')

ENV_DRIVE_PATH = '/content/drive/MyDrive/acv_Bolivia/env_acv_bolivia'

if os.path.isdir(ENV_DRIVE_PATH):
    if ENV_DRIVE_PATH not in sys.path:
        sys.path.insert(0, ENV_DRIVE_PATH)
    print("Entorno activado directamente desde Google Drive.")
else:
    print(f"No se encontró el entorno en {ENV_DRIVE_PATH}.")
```

## 4. Notas

- **Duplicación de paquetes:** `pip install --target` instala las dependencias
  en la carpeta local, que queda con **prioridad** en `sys.path` sobre las de
  Colab. Si el entorno se creó con el mismo runtime de Colab, no debería haber
  conflictos de versiones.
- Después de activar el entorno, el uso es el de siempre:
  ```python
  from acv_bolivia import ACVEngine, AppConfig
  ```
- Recuerda el **parche de SciPy** descrito en `docs/MANUAL_ACVENGINE.md`
  (sección 3) antes de usar `ACVEngine`.