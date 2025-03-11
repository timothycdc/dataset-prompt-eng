import re
from typing import Optional, List, Dict, Any, Union

# Normalize and clean up special characters or punctuation
def normalize_text(text: str) -> str:
    """
    Normalize and clean up text by removing special characters and punctuation.
    
    Args:
        text: The text to normalize
        
    Returns:
        The normalized text
    """
    # Remove special characters and punctuation
    text = re.sub(r'[^\w\s]', '', text)  # Removes all characters except word characters and whitespace
    return text.strip().lower()

