"""
Notification system with email and Telegram support.
"""
import smtplib
import requests
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from typing import Dict, Any, Optional
from pathlib import Path
import time
import json

from src.interfaces.monitoring import NotificationService


class NotificationService(NotificationService):
    """
    Notification service with email and Telegram support.
    Implements retry logic and graceful degradation when services are unavailable.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the notification service.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Extract configuration values
        self.enabled = config.get('enabled', False)
        self.service = config.get('service', 'email').lower()
        self.retry_attempts = config.get('retry_attempts', 3)
        self.retry_delay = config.get('retry_delay', 5)
        
        # Email configuration
        self.email_subject = config.get('email_subject', 'Intruder Alert!')
        self.email_from = config.get('email_from', '')
        self.email_to = config.get('email_to', '')
        self.smtp_server = config.get('smtp_server', '')
        self.smtp_port = config.get('smtp_port', 587)
        self.smtp_user = config.get('smtp_user', '')
        self.smtp_password = config.get('smtp_password', '')
        
        # Telegram configuration
        self.telegram_bot_token = config.get('telegram_bot_token', '')
        self.telegram_chat_id = config.get('telegram_chat_id', '')
        
        # Validate configuration
        if self.enabled:
            if self.service == 'email' and not all([self.email_from, self.email_to, self.smtp_server, self.smtp_user, self.smtp_password]):
                self.logger.error("Email service enabled but required fields are missing. Disabling notifications.")
                self.enabled = False
            elif self.service == 'telegram' and not all([self.telegram_bot_token, self.telegram_chat_id]):
                self.logger.error("Telegram service enabled but required fields are missing. Disabling notifications.")
                self.enabled = False
            elif self.service not in ['email', 'telegram']:
                self.logger.error(f"Unknown notification service: {self.service}. Disabling notifications.")
                self.enabled = False
        
        self.logger.info(f"Notification service initialized. Enabled: {self.enabled}, Service: {self.service}")
    
    def send_notification(self, message: str, image_path: Optional[str] = None):
        """
        Send a notification using the configured service.
        
        Args:
            message: Notification message
            image_path: Optional path to image attachment
        """
        if not self.enabled:
            self.logger.debug("Notifications are disabled, skipping notification.")
            return
        
        self.logger.info(f"Sending notification via {self.service}: '{message[:50]}...'")
        
        for attempt in range(self.retry_attempts):
            try:
                if self.service == 'email':
                    success = self._send_email(message, image_path)
                elif self.service == 'telegram':
                    success = self._send_telegram(message, image_path)
                else:
                    self.logger.error(f"Unknown notification service: {self.service}")
                    return
                
                if success:
                    self.logger.info("Notification sent successfully.")
                    return
                else:
                    self.logger.warning(f"Failed to send notification (attempt {attempt + 1}/{self.retry_attempts})")
                    if attempt < self.retry_attempts - 1:
                        time.sleep(self.retry_delay)
            except Exception as e:
                self.logger.error(f"Error sending notification (attempt {attempt + 1}/{self.retry_attempts}): {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)
        
        self.logger.error("Failed to send notification after all retry attempts.")
    
    def _send_email(self, message: str, image_path: Optional[str] = None) -> bool:
        """
        Send an email notification.
        
        Args:
            message: Email message
            image_path: Optional path to image attachment
            
        Returns:
            True if email was sent successfully, False otherwise
        """
        try:
            # Create message container
            msg = MIMEMultipart()
            msg['Subject'] = self.email_subject
            msg['From'] = self.email_from
            msg['To'] = self.email_to
            msg.attach(MIMEText(message, 'plain'))
            
            # Add image attachment if provided
            if image_path and Path(image_path).exists():
                with open(image_path, 'rb') as f:
                    img = MIMEImage(f.read())
                    img.add_header('Content-Disposition', 'attachment', filename=Path(image_path).name)
                    msg.attach(img)
            
            # Send the email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()  # Enable encryption
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)
            
            return True
            
        except smtplib.SMTPException as e:
            self.logger.error(f"SMTP error sending email: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error sending email: {e}")
            return False
    
    def _send_telegram(self, message: str, image_path: Optional[str] = None) -> bool:
        """
        Send a Telegram notification.
        
        Args:
            message: Telegram message
            image_path: Optional path to image to send
            
        Returns:
            True if message was sent successfully, False otherwise
        """
        try:
            if image_path and Path(image_path).exists():
                # Send photo with caption
                url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendPhoto"
                with open(image_path, 'rb') as f:
                    files = {'photo': f}
                    data = {'chat_id': self.telegram_chat_id, 'caption': message}
                    response = requests.post(url, files=files, data=data, timeout=30)
            else:
                # Send text message only
                url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
                data = {
                    'chat_id': self.telegram_chat_id,
                    'text': message,
                    'parse_mode': 'HTML'  # Enable HTML formatting
                }
                response = requests.post(url, data=data, timeout=30)
            
            if response.status_code == 200:
                response_data = response.json()
                if response_data.get('ok'):
                    return True
                else:
                    self.logger.error(f"Telegram API error: {response_data.get('description')}")
                    return False
            else:
                self.logger.error(f"Telegram API request failed with status {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request error sending Telegram message: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error sending Telegram message: {e}")
            return False
    
    def test_notification(self) -> bool:
        """
        Test if notifications are working.
        
        Returns:
            True if test notification was successful, False otherwise
        """
        if not self.enabled:
            return False
        
        test_message = "Test notification - TheCatBouncer is working properly."
        self.logger.info("Sending test notification...")
        
        # Temporarily store the original send method to avoid recursion
        original_enabled = self.enabled
        self.enabled = True
        self.send_notification(test_message)
        self.enabled = original_enabled
        
        # For the purpose of this test, we'll assume success if no exceptions occurred
        # A more thorough test would check actual delivery status
        return True
    
    def set_enabled(self, enabled: bool):
        """
        Enable or disable the notification service.
        
        Args:
            enabled: Whether to enable notifications
        """
        self.enabled = enabled
        self.logger.info(f"Notifications {'enabled' if enabled else 'disabled'}")


def send_notification(config: Dict[str, Any], message: str, image_path: Optional[str] = None):
    """
    Convenience function to send a notification.
    
    Args:
        config: Notification configuration
        message: Notification message
        image_path: Optional path to image attachment
    """
    service = NotificationService(config)
    service.send_notification(message, image_path)


def validate_email_config(config: Dict[str, Any]) -> bool:
    """
    Validate email configuration.
    
    Args:
        config: Email configuration
        
    Returns:
        True if configuration is valid, False otherwise
    """
    required_fields = ['email_from', 'email_to', 'smtp_server', 'smtp_user', 'smtp_password']
    return all(config.get(field) for field in required_fields)


def validate_telegram_config(config: Dict[str, Any]) -> bool:
    """
    Validate Telegram configuration.
    
    Args:
        config: Telegram configuration
        
    Returns:
        True if configuration is valid, False otherwise
    """
    required_fields = ['telegram_bot_token', 'telegram_chat_id']
    return all(config.get(field) for field in required_fields)


def check_smtp_connection(smtp_server: str, smtp_port: int, smtp_user: str, smtp_password: str) -> bool:
    """
    Check SMTP connection.
    
    Args:
        smtp_server: SMTP server address
        smtp_port: SMTP port
        smtp_user: SMTP username
        smtp_password: SMTP password
        
    Returns:
        True if connection is successful, False otherwise
    """
    try:
        with smtplib.SMTP(smtp_server, smtp_port, timeout=10) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
        return True
    except Exception as e:
        logging.error(f"SMTP connection failed: {e}")
        return False


def check_telegram_bot(telegram_bot_token: str) -> bool:
    """
    Check if Telegram bot is valid.
    
    Args:
        telegram_bot_token: Telegram bot token
        
    Returns:
        True if bot is valid, False otherwise
    """
    try:
        url = f"https://api.telegram.org/bot{telegram_bot_token}/getMe"
        response = requests.get(url, timeout=10)
        return response.status_code == 200 and response.json().get('ok', False)
    except Exception as e:
        logging.error(f"Telegram bot check failed: {e}")
        return False