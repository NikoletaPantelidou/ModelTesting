"""Test script para verificar que los logs salen en el color correcto"""
import sys
import logging

# Configurar logger con stdout
file_handler = logging.FileHandler('test.log', encoding='utf-8')
console_handler = logging.StreamHandler(sys.stdout)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Test de mensajes
print("=== PRUEBA DE COLORES EN TERMINAL ===\n")
logger.info("[INFO] Este mensaje deberia salir en blanco/normal")
logger.info("[OK] Este mensaje tambien deberia salir en blanco/normal")
logger.warning("[WARNING] Este mensaje sale en amarillo (es correcto)")
logger.error("[ERROR] Este mensaje sale en rojo (es correcto para errores)")
print("\n=== FIN DE PRUEBA ===")

