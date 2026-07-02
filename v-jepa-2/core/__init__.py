"""amta v-jepa-2: detector de objetos intuitivo con lazo de auto-correccion.

Paquete de integracion que orquesta V-JEPA 2 + segmentacion (SAM2) +
deteccion open-vocabulary + un VLM verificador, dentro de un lazo que
genera pseudo-labels y reentrena un modelo destilado.

Ver CLAUDE.md para la arquitectura completa y la topologia local/Colab.
"""

from core.config import Config, Runtime, Tier, load_config

__all__ = ["Config", "Runtime", "Tier", "load_config"]
