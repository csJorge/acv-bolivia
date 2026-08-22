# CONFIGURACIÓN E INSTALACIÓN - USANDO GIT Y PYTHON

Guía para instalar y utilizar el paquete `acv_bolivia`, tanto para usarlo como
librería (`import acv_bolivia`) como para desarrollo o modificación.

> El módulo importable es **`acv_bolivia`** (con guion bajo). El repositorio en
> GitHub se llama `acv-bolivia` (con guion), pero en Python siempre se usa
> `import acv_bolivia`.

## Requisitos previos

- **Python >= 3.10** (recomendado: 3.11).
- **Git**.
- Un entorno virtual activado: Conda o `venv`.

## Opción 1: Uso directo como librería

Ideal si solo necesitas importar el módulo en scripts de Python o Jupyter
Notebooks.

### Pasos previos

1. Abre tu terminal de preferencia (**PowerShell**, **Símbolo del sistema /
   CMD**, o **Anaconda Prompt / Conda**).
2. Asegúrate de tener el entorno virtual activado donde deseas trabajar:

```bash
# En Conda:
conda activate nombre_de_tu_entorno

# En venv (Windows):
.\nombre_entorno\Scripts\activate
```

### Comandos de instalación

**Instalación estándar** (desde el repositorio Git):

```bash
python -m pip install git+https://github.com/csJorge/acv-bolivia.git
```

**Instalación desde la rama principal** (archivo zip):

```bash
python -m pip install https://github.com/csJorge/acv-bolivia/archive/refs/heads/main.zip
```

**Forzar reinstalación limpia** — si hubo cambios recientes o errores de caché:

```bash
python -m pip install --force-reinstall --no-cache-dir https://github.com/csJorge/acv-bolivia/archive/refs/heads/main.zip
```

Los tres comandos resuelven automáticamente las dependencias declaradas en el
`pyproject.toml` del paquete (Brightway2, numpy, scipy, pandas, matplotlib,
SALib, shap, xgboost, etc.).

### Verificar la instalación

```python
import acv_bolivia
from acv_bolivia import ACVEngine, AppConfig

eng = ACVEngine.from_json("config/settings.json")
print(eng)
```

## Opción 2: Modo desarrollador (editable / código fuente)

Si deseas explorar el código, realizar modificaciones locales o utilizar las
estructuras internas con importaciones del tipo `from acv_bolivia.* import ...`.

1. **Abrir la terminal y navegar a la carpeta de proyectos**

```bash
# Ejemplo de navegación en Windows:
cd C:\Ruta\A\Tu\Carpeta_de_Proyectos
```

2. **Clonar el repositorio**

```bash
git clone https://github.com/csJorge/acv-bolivia.git
```

3. **Entrar al directorio del proyecto**

```bash
cd acv-bolivia
```

4. **Activar el entorno virtual**

```bash
conda activate tu_entorno
# o, con venv (Windows):
# .\tu_entorno\Scripts\activate
```

5. **Instalación en modo editable (`-e`)**

```bash
python -m pip install -e .
```

Para incluir las herramientas de desarrollo (pytest, black, flake8, mypy):

```bash
python -m pip install -e ".[dev]"
```

> **Nota:** el parámetro `-e` permite que cualquier cambio que realices en el
> código fuente de la carpeta se refleje inmediatamente en el `import acv_bolivia`
> o `from acv_bolivia import ...`, sin necesidad de reinstalar el paquete.

## Verificación de calidad (modo desarrollador)

```bash
python -m pytest                              # suite de tests
python -m black --check acv_bolivia           # formato
python -m flake8 acv_bolivia                  # lint
python -m mypy acv_bolivia                    # tipos estáticos
```

## Configuración del proyecto

Después de instalar, copia la plantilla de configuración y complétala con tus
rutas reales:

```bash
# Windows (PowerShell):
Copy-Item configuracion/settings.example.json config/settings.json
# macOS / Linux (o Git Bash):
# cp configuracion/settings.example.json config/settings.json
```

Consulta `docs/MANUAL_ACVENGINE.md` para el uso completo de `ACVEngine`.