# lab-cv-amta

Laboratorio de computer vision de AMTA.

**Norte:** dado un prompt en lenguaje natural sobre una cámara ("personas que entran a la
tienda A"), detectar y seguir esos objetos y convertir el seguimiento en **eventos y series
temporales** (entradas, salidas, ocupación, permanencia por zona y ventana de tiempo) sobre
las que se pueda hacer EDA y entrenar modelos de ML. El producto no son las máscaras: son
los datos que salen de ellas. Documento rector: [`docs/norte.md`](docs/norte.md) (contrato
de datos L1/L2/L3 y estado contra el norte).

Piezas: la plataforma de detección + conteo + API en `v-jepa-2/` (cascada de vigilancia),
los bancos de prueba de modelos candidatos (Grounding DINO, SAM2, Florence-2, YOLO-seg,
YOLO-World, DINOv3) y la línea de caracterización de **froth de flotación**, hoy **en pausa**.

## Estructura

| Carpeta | Qué hay |
|---|---|
| `dino_v3/`, `sam2_ov_florence_2/`, `sam2_ov_grounded/`, `yolo_seg/`, `yolo_world/` | Una carpeta por pipeline: notebook de prueba + repo upstream en `third_party/` (gitignoreado). Prefijos de autoría: `dnl_` = Daniel López, `drn_` = Dorian |
| `docs/` | `norte.md` (documento rector), `roadmap/` (roadmaps estilo roadmap.sh: el del proyecto, `roadmap.html`, y la inducción para quien se suma, `onboarding_tabata.html`; se edita el `.json` y se regeneran con `python docs/roadmap/build_roadmap.py --all`), `prior_art_norte.md` + documentación técnica de los modelos (DINOv3, Florence-2, SAM2, Grounding DINO, YOLO) |
| `froth_gate/` | **En pausa.** Experimento GATE C1/C2/C3 sobre el dataset IEEE (ver su `README.md`) + pipelines A/B de Dorian |
| `v-jepa-2/` | Plataforma: cascada de vigilancia (movimiento → detección por prompt → ByteTrack → ReID → conteo), evaluación, medición de costo, API FastAPI y tests (ver `v-jepa-2/README_cascade.md`). Aloja también el scaffold del lazo de auto-corrección y el smoke test de V-JEPA 2.1 |
| `data_froth/`, `local_docs/`, `handoffs/` | Datos y documentación de trabajo — **no versionados** (ver `.gitignore`) |

## Topología: dónde corre cada cosa

- **Local**: código liviano y análisis sobre CPU. No hay GPU de entrenamiento.
- **Colab**: todo el trabajo pesado (encoders, segmentadores, VLM, fine-tuning). T4 para el
  día a día; L4/A100 solo con código ya depurado.
- **Drive `MyDrive/Amta_lab`**: única persistencia entre sesiones de Colab (`models/`,
  `datasets/`, `outputs/`). La VM es efímera: lo que quede solo en `/content` se pierde. El
  trabajo pesado se hace sobre el disco de la VM, nunca sobre el mount FUSE, que es lento.

Los notebooks se editan localmente y se ejecutan en runtimes Colab vía la extensión oficial
de VS Code (kernel picker → Colab → **New Colab Server** para poder elegir la GPU). Al
terminar, **Remove Server**: las unidades se consumen por hora conectada, incluso idle.

## Datos

Los datasets no se versionan (varios GB). Fuentes públicas usadas, con licencia verificada
y `manifest.psv` por dataset: videos CC de froth minero real y de espumas de laboratorio,
IEEE DataPort ("Flotation Froth Sequence Images", 2.386 secuencias etiquetadas en 4
condiciones operativas; "Froth Image Pairs for Burst Bubble Recognition"), Kaggle
(`obobojk/froth-bubbles`, con máscaras) y Roboflow Universe (CC BY 4.0). Hay además
material propietario de planta, no redistribuible.

Las credenciales van en un `.env` local (gitignoreado); la plantilla versionada es
`.env.example`.

---

# Gotchas verificados

Cada uno se pagó con horas de debugging. Antes de pelear con algo de esta lista, leerla.

## V-JEPA 2 / 2.1

**1. `torch.hub.load(..., pretrained=True)` está roto upstream.** Tanto la copia local como
`facebookresearch/vjepa2@main` tienen `VJEPA_BASE_URL = "http://localhost:8300"`, que quedó
de una sesión de testing. Workaround: instanciar con `pretrained=False` y bajar el
checkpoint a mano de `https://dl.fbaipublicfiles.com/vjepa2/<archivo>.pt`. Las keys del
`state_dict` cambian según la versión: 2.1 ViT-B/L → `ema_encoder` (`strict=True`);
2.1 ViT-g y toda la serie 2.0 → `target_encoder` (2.0 necesita `strict=False`).

**2. El RoPE de V-JEPA 2.1 rompe los backends rápidos de SDPA.**
`rotate_queries_or_keys` computa seno/coseno con posiciones en fp32 y **devuelve q/k en
fp32 aun bajo autocast fp16**. Con q/k en fp32 y v en fp16, PyTorch descarta los backends
flash y memory-efficient y cae al backend *math*, que materializa la matriz de atención
completa: 64 frames a 384² son 18.432 tokens → 18.432² × 12 heads × 4 B ≈ **15 GB** → OOM
en una T4. Fix: monkeypatch que computa el RoPE en fp32 y castea la salida al dtype de
entrada, para que q/k/v queden uniformes.

**3. `torch.cuda.is_bf16_supported()` devuelve `True` en una T4 — y es una trampa.** El
soporte es emulado (sm75 no tiene bf16 nativo), pero el SDPA memory-efficient **no lo
acepta** en sm75, así que la atención vuelve al backend *math* pese al parche anterior. Hay
que gatear por `get_device_capability()[0] >= 8`, no por `is_bf16_supported()`. Impacto
medido sobre el mismo forward de ViT-B con 32 frames:

| autocast | tiempo | VRAM pico |
|---|---|---|
| bf16 (emulado → backend math) | 3150 ms | 9.9 GB |
| fp16 (→ memory-efficient) | **410 ms** | **0.7 GB** |

7,7× más rápido y 14× menos memoria, con el mismo resultado.

## Hugging Face / modelos gated

**4. DINOv3 es un repo gated, y el código de error importa.** `401` = no hay token en la
sesión (falta autenticar). `403` = el token es válido pero **la cuenta no está en la lista
de autorizados**: hay que pedir acceso en la página del modelo y esperar la aprobación de
Meta, que no es instantánea. Dos problemas distintos, con arreglos distintos.

**5. `model_info()` NO sirve para verificar acceso a un repo gated.** Devuelve `200` aunque
no tengas acceso, porque los **metadatos** del repo son públicos; el gate solo protege
`/resolve/` (los archivos). Un chequeo temprano basado en `model_info()` da falso verde y
el error aparece recién a mitad de la extracción. Para validar de verdad hay que pedir un
archivo: `hf_hub_download(repo, "config.json")` (~1 KB), capturando `GatedRepoError`.

**6. El vault de secrets de Colab no existe fuera de la UI web.** Al ejecutar desde VS Code
contra un Colab server, `huggingface_hub` intenta leer el secret del vault y falla con
*"Secrets can only be fetched when running from the Colab UI"*. Solución: cascada de
fuentes de token — variable de entorno → token ya persistido en la VM → archivo en el Drive
privado → vault de Colab → `getpass` interactivo. Guardarlo en Drive lo hace sobrevivir a
la VM.

## Colab / Drive

**7. El consentimiento de `drive.mount` es por VM y no se puede hacer persistente.** Es una
frontera de seguridad deliberada de Google: cada VM nueva vuelve a pedir autorización. Para
minimizar prompts: datos públicos/CC por link compartido + `gdown --folder` (sin mount);
outputs privados sí requieren mount (dos clics por sesión).

**8. La web de Drive puede mostrar una carpeta vacía por caché.** Es un bug visual:
verificar con `ls` desde la VM, nunca por la vista web.

**9. Los datasets en Drive tienen nombres de carpeta con `: "`.** Los notebooks deben
localizarlos con glob, no con rutas literales escritas a mano.

**10. Los Colab servers son efímeros por diseño.** Crear uno nuevo en cada sesión es el
flujo normal, no un error.

## Datos

**11. Las secuencias del dataset IEEE no sirven para tracking.** Los 12 frames de cada
secuencia sí son consecutivos (0,4 s de intervalo, 2,5 fps), pero a ese intervalo el froth
**se renueva por completo entre frames**: la correlación entre frames consecutivos es ≈ 0
en tres de las cuatro clases (medido). Son inútiles para SAM2, optical flow o cualquier
modelo que asuma continuidad temporal. Sirven como frames sueltos (conteo, BSD, radio) o
para clasificación de condición operativa. Cualquier conclusión "temporal" sobre este
dataset mide textura a 0,4 s, no movimiento — de ahí que el GATE incluya un control de
permutación obligatorio.

**12. El dataset IEEE tiene 13 secuencias con frames completamente negros**, todas de la
clase Ⅳ: `1352`–`1359`, `1368`, `1379`–`1382`. Tienen `std = 0` (un único valor de píxel en
toda la imagen). Dos consecuencias: (a) rompen cualquier feature que divida por la
desviación estándar — asimetría, curtosis y correlación entre frames quedan indefinidas
(`NaN`); (b) más grave, como están concentradas en una sola clase, **un frame negro es un
predictor casi perfecto de esa clase**. Los modelos profundos no producen `NaN` pero sí un
embedding muy distintivo para una imagen negra, así que el artefacto contamina por igual a
cualquier condición que se evalúe. Hay que excluirlas explícitamente de todas las
condiciones, no imputarlas.

**13. Los overlays PNG de Roboflow son solo visuales** (escala thumbnail): no usarlos como
anotaciones. Los labels del export COCO vienen en coordenadas 512×512 — re-escalarlos si se
aplican sobre las imágenes originales de alta resolución.

## Entorno local (Windows)

**14. El repo upstream de V-JEPA 2 tiene rutas que colisionan por mayúsculas/minúsculas**
(`vitG-384` vs `vitg-384`). El clone en Windows falla en esos archivos, pero no afecta el
uso vía PyTorch Hub.

**15. Los "imports no resueltos" de los notebooks son esperados.** Los notebooks corren en
Colab: `torch`, `torchvision`, `transformers`, `google.colab`, `timm` y `einops` no existen
en el Python local y Pylance los marca. Están bajados a severidad "information" en
`.vscode/settings.json` — no a "none", porque los scripts que **sí** corren en local
necesitan ese diagnóstico. Un subrayado en el editor sin haber ejecutado la celda es
siempre análisis estático, no un fallo de ejecución: un fallo real aparece como traceback
en la salida de la celda.

---

# Pendientes

Orden de prioridad según el norte (`docs/norte.md` §4). La percepción está resuelta y
medida; lo que falta es la mitad de datos.

## 1. De detecciones a datos (prioridad)

- [ ] **L1 enriquecida**: agregar `camara_id`, `ts` absoluto (UTC), `prompt` y `modelo` a
      cada detección; coordenadas normalizadas [0,1].
- [ ] **Zonas con nombre**: polígonos y líneas declarados por cámara (`zona_id`), no solo la
      línea de conteo por defecto.
- [ ] **L2 eventos** con timestamp: `entrada`, `salida`, `cruce`, `permanencia`, `ocupacion`
      por `track_id` y zona — hoy el conteo solo sale agregado por clip.
- [ ] **L3 agregados** por ventana configurable (1 min / 15 min / 1 h / 1 día).
- [ ] **Persistencia en Parquet** particionado por cámara y fecha; exponer L1/L2/L3 en la API.
- [ ] **Notebook de EDA** sobre L3 (distribuciones, picos, estacionalidad) como primer
      consumidor del contrato, y un baseline de ML (p. ej. pronóstico de afluencia).
- [ ] **Prompt de punta a punta**: que el prompt de la petición gobierne la detección (hoy
      solo YOLO-World lo usa; el default YOLO11 tiene clases COCO fijas).

## 2. Percepción (mejoras)

- [ ] Evaluar como backends de N2 los modelos de los bancos de prueba (Grounding DINO,
      SAM2 con máscaras, Florence-2), midiendo su efecto sobre la precisión del conteo (L2).
- [ ] Portar a `core/perception/encoder.py` el patrón validado del smoke test de V-JEPA 2.1
      (carga manual + parche RoPE + fp16 en T4), si se decide usarlo para eventos ricos.
- [ ] Lazo de auto-corrección (VLM verificador + pseudo-labels): largo plazo.

## 3. Froth — en pausa

- [ ] GATE froth: C2 (DINOv3, bloqueado por acceso gated), C3 (V-JEPA 2.1) y veredicto con
      `gate_analysis.py`. Sin trabajo activo hasta nuevo aviso. C1 ya medido: F1 macro
      91,9; permutando frames 91,7; split por bloques 84,3 (leakage temporal suave).

## Repo

- `main` tiene branch protection: solo se actualiza mergeando un PR desde `dev/*`
  (sin push directo, ni siquiera de admins; sin force push ni borrado).
