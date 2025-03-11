import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Literal, Set, List, Union
from pydantic import BaseModel


# Load languages from language_data.json
data_dir = Path(__file__).parent.parent / "data"
language_data_path = data_dir / "language_data.json"

try:
    with open(language_data_path, "r", encoding="utf-8") as f:
        language_data = json.load(f)
    LANGUAGES = set(language_data.keys())
except (FileNotFoundError, json.JSONDecodeError):
    print(f"Error loading language data from {language_data_path}. Language set will be empty.")
    LANGUAGES = set()  # Empty set if file not found or invalid


class Preamble(BaseModel):
    """Pydantic class for a preamble template. Main function is `format` which returns the preamble template with the variables substituted."""
    preamble_name: str
    preamble_type: Literal["Zero-Shot", "Few-Shot"]
    preamble_template: str = ""
    
    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self.load_template()
    
    def load_template(self) -> str:
        """Load the preamble template from the corresponding file."""
        # Determine the directory based on preamble type
        type_dir = self.preamble_type.lower()  # Convert to lowercase
        
        template_path = Path(__file__).parent.parent / "data_preamble" / type_dir / f"{self.preamble_name}.txt"
        
        if not template_path.exists():
            raise FileNotFoundError(f"Preamble template not found: {template_path}")
        
        with open(template_path, "r", encoding="utf-8") as f:
            self.preamble_template = f.read()
        
        return self.preamble_template
    
    def format(self, language: str, variables: Optional[Dict[str, Any]]=None) -> str:
        """
        Format the preamble template with language data, few-shot examples (if preamble_type is "Few-Shot") and additional variables.
        
        Args:
            language: The target language for translation
            examples: Optional dictionary of examples to override the ones from language_data.json
            variables: Optional dictionary of additional variables to substitute
            
        Returns:
            The formatted preamble text
        """
        # Check if language is supported
        if language not in LANGUAGES:
            available_languages = ", ".join(sorted(LANGUAGES))
            raise ValueError(f"Language '{language}' not found in language data. Available languages: {available_languages}")
        
        # Load language data from JSON
        data_dir = Path(__file__).parent.parent / "data"
        language_data_path = data_dir / "language_data.json"
        
        with open(language_data_path, "r", encoding="utf-8") as f:
            all_language_data = json.load(f)
        
        # Get language-specific examples
        language_examples = all_language_data.get(language, {})
        
        # Create the format variables dictionary
        format_vars = {"language": language}
        format_vars.update(language_examples)
        if variables:
            format_vars.update(variables)
        
        # Format the template
        try:
            return self.preamble_template.format(**format_vars)
        except KeyError as e:
            raise KeyError(f"Missing variable in preamble template: {e}")
        

def load_all_preambles() -> List[Preamble]:
    """
    Load all preambles from the few-shot and zero-shot directories.
    
    Returns:
        A list of Preamble objects
    """
    preambles: List[Preamble] = []
    base_dir = Path(__file__).parent.parent / "data_preamble"
    
    # Load zero-shot preambles
    zero_shot_dir = base_dir / "zero-shot"
    if zero_shot_dir.exists():
        for file_path in zero_shot_dir.glob("*.txt"):
            preamble_name = file_path.stem
            preambles.append(Preamble(
                preamble_name=preamble_name,
                preamble_type="Zero-Shot"
            ))
    
    # Load few-shot preambles
    few_shot_dir = base_dir / "few-shot"
    if few_shot_dir.exists():
        for file_path in few_shot_dir.glob("*.txt"):
            preamble_name = file_path.stem
            preambles.append(Preamble(
                preamble_name=preamble_name,
                preamble_type="Few-Shot"
            ))
    
    return preambles

# Create a list of all available preambles
ALL_PREAMBLES: List[Preamble] = load_all_preambles()

