import logging

def setup_logger():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('volatility_analysis.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)
