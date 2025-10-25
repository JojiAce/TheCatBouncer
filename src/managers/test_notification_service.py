"""
Unit tests for notification system.
"""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.managers.notification_service import (
    NotificationService, 
    send_notification, 
    validate_email_config,
    validate_telegram_config,
    check_smtp_connection,
    check_telegram_bot
)


def test_notification_service_initialization_email():
    """Test NotificationService initialization with email configuration."""
    config = {
        'enabled': True,
        'service': 'email',
        'email_subject': 'Test Subject',
        'email_from': 'from@example.com',
        'email_to': 'to@example.com',
        'smtp_server': 'smtp.example.com',
        'smtp_port': 587,
        'smtp_user': 'user',
        'smtp_password': 'password'
    }
    
    service = NotificationService(config)
    
    assert service.enabled == True
    assert service.service == 'email'
    assert service.email_subject == 'Test Subject'


def test_notification_service_initialization_telegram():
    """Test NotificationService initialization with Telegram configuration."""
    config = {
        'enabled': True,
        'service': 'telegram',
        'telegram_bot_token': 'test_token',
        'telegram_chat_id': 'test_chat_id'
    }
    
    service = NotificationService(config)
    
    assert service.enabled == True
    assert service.service == 'telegram'
    assert service.telegram_bot_token == 'test_token'


def test_notification_service_initialization_missing_email_fields():
    """Test NotificationService initialization with missing email fields."""
    config = {
        'enabled': True,
        'service': 'email',
        # Missing required email fields
    }
    
    service = NotificationService(config)
    
    assert service.enabled == False  # Should be disabled due to missing config


def test_notification_service_initialization_missing_telegram_fields():
    """Test NotificationService initialization with missing Telegram fields."""
    config = {
        'enabled': True,
        'service': 'telegram',
        # Missing required telegram fields
    }
    
    service = NotificationService(config)
    
    assert service.enabled == False  # Should be disabled due to missing config


def test_notification_service_initialization_unknown_service():
    """Test NotificationService initialization with unknown service."""
    config = {
        'enabled': True,
        'service': 'unknown_service',
        'email_from': 'from@example.com',
        'email_to': 'to@example.com',
        'smtp_server': 'smtp.example.com',
        'smtp_user': 'user',
        'smtp_password': 'password'
    }
    
    service = NotificationService(config)
    
    assert service.enabled == False  # Should be disabled due to unknown service


@patch('smtplib.SMTP')
def test_send_email_success(mock_smtp):
    """Test successful email sending."""
    # Mock SMTP connection
    mock_server = Mock()
    mock_smtp.return_value.__enter__.return_value = mock_server
    
    config = {
        'enabled': True,
        'service': 'email',
        'email_subject': 'Test Subject',
        'email_from': 'from@example.com',
        'email_to': 'to@example.com',
        'smtp_server': 'smtp.example.com',
        'smtp_port': 587,
        'smtp_user': 'user',
        'smtp_password': 'password'
    }
    
    service = NotificationService(config)
    
    # Create a temporary image file for testing
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        success = service._send_email("Test message", tmp_path)
        assert success == True
    finally:
        Path(tmp_path).unlink()


@patch('requests.post')
def test_send_telegram_success(mock_post):
    """Test successful Telegram message sending."""
    # Mock successful response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {'ok': True}
    mock_post.return_value = mock_response
    
    config = {
        'enabled': True,
        'service': 'telegram',
        'telegram_bot_token': 'test_token',
        'telegram_chat_id': 'test_chat_id'
    }
    
    service = NotificationService(config)
    
    # Create a temporary image file for testing
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        success = service._send_telegram("Test message", tmp_path)
        assert success == True
    finally:
        Path(tmp_path).unlink()


@patch('requests.post')
def test_send_telegram_text_only_success(mock_post):
    """Test successful Telegram text-only message sending."""
    # Mock successful response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {'ok': True}
    mock_post.return_value = mock_response
    
    config = {
        'enabled': True,
        'service': 'telegram',
        'telegram_bot_token': 'test_token',
        'telegram_chat_id': 'test_chat_id'
    }
    
    service = NotificationService(config)
    
    success = service._send_telegram("Test message")
    assert success == True


def test_send_notification_disabled():
    """Test that send_notification does nothing when disabled."""
    config = {
        'enabled': False,
        'service': 'email',
        'email_subject': 'Test Subject',
        'email_from': 'from@example.com',
        'email_to': 'to@example.com',
        'smtp_server': 'smtp.example.com',
        'smtp_port': 587,
        'smtp_user': 'user',
        'smtp_password': 'password'
    }
    
    service = NotificationService(config)
    service.send_notification("Test message")  # Should not do anything


def test_validate_email_config():
    """Test email configuration validation."""
    config = {
        'email_from': 'from@example.com',
        'email_to': 'to@example.com',
        'smtp_server': 'smtp.example.com',
        'smtp_user': 'user',
        'smtp_password': 'password'
    }
    
    assert validate_email_config(config) == True
    
    # Test with missing field
    incomplete_config = {
        'email_from': 'from@example.com',
        'email_to': 'to@example.com',
        'smtp_server': 'smtp.example.com'
        # Missing smtp_user and smtp_password
    }
    
    assert validate_email_config(incomplete_config) == False


def test_validate_telegram_config():
    """Test Telegram configuration validation."""
    config = {
        'telegram_bot_token': 'test_token',
        'telegram_chat_id': 'test_chat_id'
    }
    
    assert validate_telegram_config(config) == True
    
    # Test with missing field
    incomplete_config = {
        'telegram_bot_token': 'test_token'
        # Missing telegram_chat_id
    }
    
    assert validate_telegram_config(incomplete_config) == False


@patch('smtplib.SMTP')
def test_check_smtp_connection_success(mock_smtp):
    """Test successful SMTP connection check."""
    # Mock successful SMTP connection
    mock_server = Mock()
    mock_smtp.return_value.__enter__.return_value = mock_server
    
    result = check_smtp_connection('smtp.example.com', 587, 'user', 'password')
    assert result == True


@patch('requests.get')
def test_check_telegram_bot_success(mock_get):
    """Test successful Telegram bot check."""
    # Mock successful response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {'ok': True}
    mock_get.return_value = mock_response
    
    result = check_telegram_bot('test_token')
    assert result == True


@patch('requests.get')
def test_check_telegram_bot_failure(mock_get):
    """Test failed Telegram bot check."""
    # Mock failed response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {'ok': False}
    mock_get.return_value = mock_response
    
    result = check_telegram_bot('test_token')
    assert result == False


def test_set_enabled():
    """Test enabling/disabling notifications."""
    config = {
        'enabled': True,
        'service': 'email',
        'email_subject': 'Test Subject',
        'email_from': 'from@example.com',
        'email_to': 'to@example.com',
        'smtp_server': 'smtp.example.com',
        'smtp_port': 587,
        'smtp_user': 'user',
        'smtp_password': 'password'
    }
    
    service = NotificationService(config)
    assert service.enabled == True
    
    service.set_enabled(False)
    assert service.enabled == False
    
    service.set_enabled(True)
    assert service.enabled == True


def test_send_notification_convenience_function():
    """Test the send_notification convenience function."""
    config = {
        'enabled': False,  # Disable to avoid actual sending
        'service': 'email',
        'email_subject': 'Test Subject',
        'email_from': 'from@example.com',
        'email_to': 'to@example.com',
        'smtp_server': 'smtp.example.com',
        'smtp_port': 587,
        'smtp_user': 'user',
        'smtp_password': 'password'
    }
    
    # This should not raise an exception
    send_notification(config, "Test message")


if __name__ == "__main__":
    test_notification_service_initialization_email()
    test_notification_service_initialization_telegram()
    test_notification_service_initialization_missing_email_fields()
    test_notification_service_initialization_missing_telegram_fields()
    test_notification_service_initialization_unknown_service()
    test_send_email_success()
    test_send_telegram_success()
    test_send_telegram_text_only_success()
    test_send_notification_disabled()
    test_validate_email_config()
    test_validate_telegram_config()
    test_check_smtp_connection_success()
    test_check_telegram_bot_success()
    test_check_telegram_bot_failure()
    test_set_enabled()
    test_send_notification_convenience_function()
    print("All notification system tests passed!")