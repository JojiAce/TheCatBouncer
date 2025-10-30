import requests
import logging
import json

# Disable SSL warnings for local bridge communication
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

class HueController:
    """
    Steuert Philips Hue Lichter direkt über die lokale API.
    Ist ausfallsicher konzipiert und führt bei Fehlern nicht zum Absturz.
    """
    def __init__(self, bridge_ip: str, app_key: str):
        if not bridge_ip or not app_key:
            logger.error("[Hue] Bridge-IP oder App-Key fehlen. Hue-Steuerung ist deaktiviert.")
            self.base_url = None
            self.is_active = False
            return

        self.base_url = f"https://{bridge_ip}/clip/v2/resource/light"
        self.headers = {
            "hue-application-key": app_key,
        }
        self.is_active = self._test_connection()

    def _test_connection(self) -> bool:
        """Prüft beim Start, ob die Bridge erreichbar ist."""
        logger.info(f"[Hue] Teste Verbindung zur Bridge unter {self.base_url}...")
        try:
            # Wir fragen den Status eines beliebigen Lichts ab (hier ID 1 als Test)
            response = requests.get(f"{self.base_url}/1", headers=self.headers, verify=False, timeout=5)
            if response.status_code == 200:
                logger.info("[Hue] Verbindung zur Bridge erfolgreich hergestellt.")
                return True
            else:
                logger.error(f"[Hue] Verbindungstest fehlgeschlagen. Status: {response.status_code}, Antwort: {response.text}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"[Hue] Verbindung zur Bridge konnte nicht hergestellt werden: {e}")
            return False

    def _send_put_request(self, light_id: str, payload: dict):
        """Sendet einen PUT-Request an ein spezifisches Licht."""
        if not self.is_active:
            return

        url = f"{self.base_url}/{light_id}"
        try:
            response = requests.put(url, headers=self.headers, data=json.dumps(payload), verify=False, timeout=5)
            response.raise_for_status() # Löst einen Fehler bei 4xx/5xx Antworten aus
            logger.debug(f"[Hue] Befehl erfolgreich an Licht {light_id} gesendet. Payload: {payload}")
        except requests.exceptions.RequestException as e:
            logger.error(f"[Hue] Fehler beim Senden des Befehls an Licht {light_id}: {e}")
            # Optional: Deaktiviere weitere Versuche nach einem Fehler
            # self.is_active = False

    def set_lights_on(self, light_ids: list, brightness: int, saturation: int, hue: int):
        """Schaltet die definierten Lichter mit den gewünschten Einstellungen an."""
        logger.info(f"[Hue] Schalte Lichter an: {light_ids}")
        payload = {
            "on": {"on": True},
            "dimming": {"brightness": float(brightness) / 254 * 100}, # API v2 will Helligkeit in %
            "color": {
                "xy": self._convert_hsv_to_xy(hue, saturation) # API v2 bevorzugt xy
            }
        }
        for light_id in light_ids:
            self._send_put_request(light_id.strip(), payload)

    def set_lights_off(self, light_ids: list):
        """Schaltet die definierten Lichter aus."""
        logger.info(f"[Hue] Schalte Lichter aus: {light_ids}")
        payload = {"on": {"on": False}}
        for light_id in light_ids:
            self._send_put_request(light_id.strip(), payload)
            
    def _convert_hsv_to_xy(self, h, s):
        # Vereinfachte Konvertierung von Hue/Sat zu XY für die Hue API
        # Dies ist eine Annäherung und kann für präzise Farben verbessert werden.
        # Für weißes Licht (s=0) ist dies ausreichend.
        if s == 0: return {"x": 0.3127, "y": 0.3290} # Standard-Weißpunkt
        
        # Implementierung einer echten HSV->XY Konvertierung wäre hier notwendig
        # Fürs Erste verwenden wir einen Standardwert für farbiges Licht
        logger.warning("[Hue] Präzise HSV-zu-XY-Farbkonvertierung nicht implementiert. Verwende Standardwerte.")
        return {"x": 0.4573, "y": 0.4100}  # Beispielwert

