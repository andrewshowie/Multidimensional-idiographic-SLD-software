# logger_config.py
import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict
import json
import sys
from functools import wraps
import traceback
import threading
from queue import Queue

class LoggerFormatter(logging.Formatter):
    """Custom formatter with extra fields and color support"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def __init__(self, use_colors: bool = True):
        super().__init__(
            fmt='%(asctime)s | %(levelname)-8s | %(threadName)s | %(name)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with optional color"""
        # Add thread ID if not main thread
        if not hasattr(record, 'threadName'):
            record.threadName = threading.current_thread().name

        # Format the message
        message = super().format(record)
        
        if self.use_colors and sys.stdout.isatty():
            color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
            reset = self.COLORS['RESET']
            return f"{color}{message}{reset}"
        
        return message

class AsyncHandler(logging.Handler):
    """Asynchronous logging handler using a queue"""
    
    def __init__(self, handler: logging.Handler):
        super().__init__()
        self.handler = handler
        self.queue = Queue()
        self.thread = threading.Thread(target=self._process_queue, daemon=True)
        self.thread.start()

    def emit(self, record: logging.LogRecord):
        """Add log record to queue"""
        self.queue.put(record)

    def _process_queue(self):
        """Process log records from queue"""
        while True:
            try:
                record = self.queue.get()
                self.handler.emit(record)
                self.queue.task_done()
            except Exception:
                continue

class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'thread': record.threadName,
            'line': record.lineno,
            'message': record.getMessage()
        }
        
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
            
        if hasattr(record, 'stack_info') and record.stack_info:
            log_data['stack_info'] = self.formatStack(record.stack_info)
            
        return json.dumps(log_data)

class LogManager:
    """Central logging manager"""
    
    def __init__(self, 
                 log_dir: str = "logs",
                 app_name: str = "chat_app",
                 log_level: int = logging.DEBUG,
                 use_json: bool = False,
                 use_async: bool = True,
                 max_bytes: int = 10_000_000,  # 10MB
                 backup_count: int = 5):
        
        self.log_dir = Path(log_dir)
        self.app_name = app_name
        self.log_level = log_level
        self.use_json = use_json
        self.use_async = use_async
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        
        # Create log directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        self.setup_logging()

    def setup_logging(self):
        """Configure logging with multiple handlers"""
        # Clear any existing handlers
        logging.getLogger().handlers.clear()
        
        # Create root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
        
        # Add handlers
        handlers = []
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(LoggerFormatter(use_colors=True))
        handlers.append(console_handler)
        
        # File handlers
        log_file = self.log_dir / f"{self.app_name}_{datetime.now():%Y%m%d_%H%M%S}.log"
        
        # Regular log file
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=self.max_bytes,
            backupCount=self.backup_count
        )
        file_handler.setFormatter(LoggerFormatter(use_colors=False))
        handlers.append(file_handler)
        
        # JSON log file (if enabled)
        if self.use_json:
            json_file = self.log_dir / f"{self.app_name}_{datetime.now():%Y%m%d_%H%M%S}.json"
            json_handler = logging.handlers.RotatingFileHandler(
                json_file,
                maxBytes=self.max_bytes,
                backupCount=self.backup_count
            )
            json_handler.setFormatter(JsonFormatter())
            handlers.append(json_handler)
        
        # Make handlers asynchronous if enabled
        if self.use_async:
            handlers = [AsyncHandler(handler) for handler in handlers]
        
        # Add all handlers to root logger
        for handler in handlers:
            root_logger.addHandler(handler)

def log_exceptions(logger: Optional[logging.Logger] = None):
    """Decorator to log exceptions"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = logging.getLogger(func.__module__)
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    f"Exception in {func.__name__}: {str(e)}\n"
                    f"Args: {args}, Kwargs: {kwargs}\n"
                    f"Traceback:\n{traceback.format_exc()}"
                )
                raise
        return wrapper
    return decorator

# Create global log manager instance
log_manager = LogManager()

# Create logger for this module
logger = logging.getLogger(__name__)

# Example usage functions
def test_logging():
    """Test logging functionality"""
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    try:
        raise ValueError("Test exception")
    except Exception as e:
        logger.exception("Caught an exception")

@log_exceptions()
def test_exception_logging():
    """Test exception logging decorator"""
    raise ValueError("Test decorated exception")

if __name__ == "__main__":
    # Test logging system
    test_logging()
    try:
        test_exception_logging()
    except Exception:
        pass