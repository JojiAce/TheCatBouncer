import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import requests
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

class Notifier:
    """
    Verwaltet den Versand von Benachrichtigungen über verschiedene Dienste.
    """
    def __init__(self, config: dict):
        self.config = config
        self.is_active = config.get('enabled', 'false').lower() == 'true'
        self.service = config.get('service', 'email').lower()
        
        if self.is_active:
            logger.info(f"Benachrichtigungen sind aktiviert. Dienst: '{self.service}'")
        else:
            logger.info("Benachrichtigungen sind deaktiviert.")

    def send(self, message: str, image_path: str = None):
        """Sendet eine Benachrichtigung über den konfigurierten Dienst."""
        if not self.is_active:
            return

        logger.info(f"Sende Benachrichtigung: '{message}'")
        try:
            if self.service == 'email':
                self._send_email(message, image_path)
            elif self.service == 'telegram':
                self._send_telegram(message, image_path)
            else:
                logger.error(f"Unbekannter Benachrichtigungsdienst: '{self.service}'")
        except Exception as e:
            logger.error(f"Fehler beim Senden der Benachrichtigung: {e}")

    def _send_email(self, message: str, image_path: str = None):
        """Sendet eine E-Mail-Benachrichtigung."""
        msg = MIMEMultipart()
        msg['Subject'] = self.config.get('email_subject', 'Intruder Alert!')
        msg['From'] = self.config.get('email_from')
        msg['To'] = self.config.get('email_to')
        msg.attach(MIMEText(message, 'plain'))

        if image_path and Path(image_path).exists():
            with open(image_path, 'rb') as f:
                img = MIMEImage(f.read())
                img.add_header('Content-Disposition', 'attachment', filename=Path(image_path).name)
                msg.attach(img)
        
        with smtplib.SMTP(self.config['smtp_server'], int(self.config['smtp_port'])) as server:
            server.starttls()
            server.login(self.config['smtp_user'], self.config['smtp_password'])
            server.send_message(msg)
        logger.info("E-Mail-Benachrichtigung erfolgreich gesendet.")

    def _send_telegram(self, message: str, image_path: str = None):
        """Sendet eine Telegram-Benachrichtigung."""
        token = self.config.get('telegram_bot_token')
        chat_id = self.config.get('telegram_chat_id')
        
        if image_path and Path(image_path).exists():
            url = f"https://api.telegram.org/bot{token}/sendPhoto"
            with open(image_path, 'rb') as f:
                files = {'photo': f}
                data = {'chat_id': chat_id, 'caption': message}
                response = requests.post(url, files=files, data=data, timeout=10)
        else:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            data = {'chat_id': chat_id, 'text': message}
            response = requests.post(url, data=data, timeout=10)
        
        response.raise_for_status()
        logger.info("Telegram-Benachrichtigung erfolgreich gesendet.")

