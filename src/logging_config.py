# Logging configuration for ScientificLLM-Forge
import logging
import logging.handlers
from pathlib import Path

# Create logs directory if it doesn't exist
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console handler
        logging.handlers.RotatingFileHandler(
            logs_dir / "scientific_llm_forge.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
    ]
)

# Configure specific loggers
loggers = {
    'data': logging.getLogger('data'),
    'models': logging.getLogger('models'),
    'serving': logging.getLogger('serving'),
    'utils': logging.getLogger('utils'),
}

for logger_name, logger in loggers.items():
    logger.setLevel(logging.INFO)
