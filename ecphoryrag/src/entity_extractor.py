# entity_extractor.py

import json
import re
import time
from typing import Callable, Dict, List, Optional, Any, Union, Tuple

# Predefined entity types for standardization
ENTITY_TYPES = [
    "PERSON",      # People, individuals
    "ORG",         # Organizations, companies, institutions
    "LOC",         # Locations, places, geographical entities
    "CONCEPT",     # Abstract ideas, theories, methodologies
    "DATE",        # Dates, time periods
    "EVENT",       # Events, happenings, occurrences
    "PRODUCT",     # Products, services, commercial offerings
    "WORK",        # Creative works (books, movies, etc.)
    "QUANTITY",    # Measurements, numbers with units
    "MISC"         # Miscellaneous entities that don't fit other categories
]

def extract_entities_llm(
    text: str, 
    ollama_completion_func: Callable[[str, str], Tuple[str, Dict[str, Any]]], 
    extraction_llm_model: str = "phi3:mini"
) -> List[Dict[str, Any]]:
    """
    Extract rich entities from text using an Ollama LLM.
    
    Args:
        text: The input text to extract entities from
        ollama_completion_func: Function to call Ollama completion API
        extraction_llm_model: Name of the LLM model to use for extraction
        
    Returns:
        List of extracted entity dictionaries, each containing:
        - text: The entity text
        - type: The entity type (e.g., PERSON, LOCATION, etc.)
        - description: Optional description or context for the entity
        - importance_score: Optional importance score from 1-5
    """
    if not text or not text.strip():
        return []
    
    # Track extraction time
    start_time = time.time()
    
    # Truncate text if it's too long
    max_length = 4000
    if len(text) > max_length:
        truncated_text = text[:max_length]
        # print(f"Warning: Text truncated from {len(text)} to {max_length} characters for entity extraction.")
    else:
        truncated_text = text
    
    # Construct entity extraction prompt
    prompt = f"""
Extract key entities and concepts from the following text. For each entity, provide its type, a brief description or contextual information, and an importance score (1-5, where 5 is most important). Focus on entities that are crucial for understanding the context.

Text to analyze:
```
{truncated_text}
```

Respond in the following JSON format:
[
  {{
    "text": "entity1",
    "type": "PERSON/LOCATION/ORGANIZATION/DATE/CONCEPT/EVENT/...",
    "description": "brief contextual information",
    "importance_score": 1-5
  }},
  ...
]

Only return the JSON array, nothing else.
"""
    
    try:
        # Call LLM
        response, token_info = ollama_completion_func(prompt, extraction_llm_model)
        
        # Try to parse the response as JSON
        if not response:
            # print("Error: Empty response from entity extraction LLM")
            return []
        
        # Clean up the response - try to extract JSON array
        cleaned_response = response.strip()
        
        # Check if the response starts and ends with square brackets
        if not (cleaned_response.startswith('[') and cleaned_response.endswith(']')):
            # Try to find the JSON array in the response
            start_idx = cleaned_response.find('[')
            end_idx = cleaned_response.rfind(']')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                cleaned_response = cleaned_response[start_idx:end_idx+1]
            else:
                # print(f"Error: Could not find JSON array in entity extraction LLM response: {cleaned_response[:100]}...")
                return []
        
        # Parse JSON
        entities = json.loads(cleaned_response)
        
        # Filter and clean entities
        valid_entities = []
        for entity in entities:
            if not isinstance(entity, dict):
                continue
                
            # Ensure required fields
            if 'text' not in entity or not entity['text']:
                continue
                
            # Ensure entity has a type
            if 'type' not in entity or not entity['type']:
                entity['type'] = 'MISC'
                
            # Add default description if missing
            if 'description' not in entity or not entity['description']:
                entity['description'] = ''
                
            # Add default importance score if missing or invalid
            if 'importance_score' not in entity:
                entity['importance_score'] = 3
            elif not isinstance(entity['importance_score'], (int, float)) or entity['importance_score'] < 1 or entity['importance_score'] > 5:
                entity['importance_score'] = 3
                
            valid_entities.append(entity)
            
        extraction_time = time.time() - start_time
        # print(f"Entity extraction completed in {extraction_time:.2f} seconds, found {len(valid_entities)} entities")
        
        return valid_entities
        
    except json.JSONDecodeError as e:
        # print(f"Error parsing entity extraction LLM response: {e}")
        # print(f"Raw response: {response[:100]}...")
        return []
    except Exception as e:
        # print(f"Error during entity extraction: {e}")
        return []

def _parse_enriched_json_response(response: str) -> Optional[List[Dict[str, Any]]]:
    """
    Parse the JSON response from the LLM for enriched entity extraction, dealing with common formatting issues.
    
    Args:
        response: The string response from the LLM.
        
    Returns:
        A list of entity dictionaries if parsing succeeds, None otherwise.
    """
    # Print the raw response for debugging
    print("Raw LLM response:")
    print(response[:200] + "..." if len(response) > 200 else response)
    
    # Try to find a JSON array in the response
    json_pattern = r'\[.*?\]'
    json_matches = re.findall(json_pattern, response, re.DOTALL)
    
    # Try different parsing strategies
    candidates = [
        response,  # Try the full response first
        *json_matches  # Then try any JSON-like substrings
    ]
    
    for candidate in candidates:
        try:
            # Clean the candidate string - remove any markdown code block markers
            cleaned_candidate = re.sub(r'```json|```|\n', '', candidate)
            parsed = json.loads(cleaned_candidate)
            # Ensure we have a list
            if isinstance(parsed, list):
                # Check if the list contains dictionaries with the expected structure
                if all(isinstance(item, dict) and "text" in item for item in parsed):
                    return parsed
                # If we have a list of strings, convert them to the new format
                elif all(isinstance(item, str) for item in parsed):
                    return [{"text": item, "type": "MISC", "description": "No description provided.", "importance_score": 3} for item in parsed]
        except json.JSONDecodeError:
            continue
    
    # If no parsing worked, try manual extraction for simpler formats
    # This is a fallback for when the LLM doesn't return proper JSON
    try:
        # Check for a pattern that might indicate entity objects in text form
        entity_pattern = r'\{\s*"text"\s*:\s*"([^"]+)"[^\}]*\}'
        entity_matches = re.findall(entity_pattern, response, re.DOTALL)
        if entity_matches:
            return [{"text": match, "type": "MISC", "description": "No description provided.", "importance_score": 3} for match in entity_matches]
    except Exception:
        pass
    
    # As a last resort, try to extract simple strings and convert them to the new format
    try:
        # Look for quoted strings
        quoted_strings = re.findall(r'"([^"]*)"', response)
        if quoted_strings:
            return [{"text": item, "type": "MISC", "description": "No description provided.", "importance_score": 3} for item in quoted_strings]
    except Exception:
        pass
    
    # Try markdown list format as a final fallback
    try:
        if '- ' in response:  # Markdown list format
            lines = [line.strip() for line in response.split('\n') if line.strip().startswith('- ')]
            if lines:
                return [{"text": line[2:], "type": "MISC", "description": "No description provided.", "importance_score": 3} for line in lines]  # Remove the '- ' prefix
    except Exception:
        pass
    
    return None


def _validate_entity_type(entity_type: str) -> str:
    """
    Validates that the entity type is from the predefined list.
    If not, returns "MISC" as a fallback.
    
    Args:
        entity_type: The entity type to validate.
        
    Returns:
        A valid entity type from the ENTITY_TYPES list.
    """
    # Normalize to uppercase for comparison
    normalized_type = entity_type.upper() if entity_type else ""
    
    # Check if the normalized type is in our predefined list
    if normalized_type in ENTITY_TYPES:
        return normalized_type
    
    # Try to map common variations to standard types
    type_mapping = {
        "PERSON": ["PEOPLE", "INDIVIDUAL", "HUMAN", "PERSONAL"],
        "ORG": ["ORGANIZATION", "COMPANY", "INSTITUTION", "CORPORATION", "AGENCY", "GROUP"],
        "LOC": ["LOCATION", "PLACE", "GEOGRAPHICAL", "GEOGRAPHY", "AREA", "REGION", "COUNTRY", "CITY"],
        "CONCEPT": ["IDEA", "THEORY", "PRINCIPLE", "METHODOLOGY", "METHOD", "TECHNIQUE", "APPROACH"],
        "DATE": ["TIME", "PERIOD", "YEAR", "MONTH", "DAY", "DATETIME", "TEMPORAL"],
        "EVENT": ["HAPPENING", "OCCURRENCE", "INCIDENT", "OCCASION"],
        "PRODUCT": ["SERVICE", "OFFERING", "ITEM", "GOOD", "MERCHANDISE"],
        "WORK": ["BOOK", "MOVIE", "FILM", "PUBLICATION", "ARTWORK", "CREATION"],
        "QUANTITY": ["MEASUREMENT", "NUMBER", "AMOUNT", "VALUE", "METRIC"],
        "MISC": ["MISCELLANEOUS", "OTHER", "UNKNOWN", "VARIOUS"]
    }
    
    for standard_type, variations in type_mapping.items():
        if any(variation in normalized_type for variation in variations):
            return standard_type
    
    # Default to MISC if no match is found
    return "MISC"


def _normalize_importance_score(score: Any) -> Union[int, str]:
    """
    Normalizes the importance score to either an integer (1-5) or a string (high/medium/low).
    
    Args:
        score: The importance score to normalize, which could be an int, string, or other value.
        
    Returns:
        A normalized importance score as either an int (1-5) or string (high/medium/low).
    """
    # If score is None or empty, return a default value
    if score is None or (isinstance(score, str) and not score.strip()):
        return 3  # Default to medium importance
    
    # If score is already an integer in the range 1-5, return it
    if isinstance(score, int) and 1 <= score <= 5:
        return score
    
    # Try to convert string to integer
    if isinstance(score, str):
        # Check for textual importance levels
        score_lower = score.lower().strip()
        if score_lower in ["high", "important", "critical", "key", "essential"]:
            return 5
        elif score_lower in ["medium", "moderate", "average"]:
            return 3
        elif score_lower in ["low", "minor", "minimal"]:
            return 1
        
        # Try to extract a number from the string
        try:
            # Extract digits from the string
            digits = re.findall(r'\d+', score_lower)
            if digits:
                num = int(digits[0])
                if 1 <= num <= 5:
                    return num
                elif num > 5:
                    return 5  # Cap at 5
                else:
                    return 1  # Minimum is 1
        except ValueError:
            pass
    
    # Default to medium importance (3) if we couldn't parse the score
    return 3

if __name__ == "__main__":
    # Example usage of the entity extractor
    from ollama_clients import get_ollama_completion
    import sys
    
    # Set up basic logging
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    sample_text = """
    OpenAI was founded in December 2015 by Sam Altman, Elon Musk, Greg Brockman, 
    Ilya Sutskever, John Schulman, and Wojciech Zaremba. Microsoft invested $1 billion in OpenAI in 2019. 
    The company is headquartered in San Francisco and is developing artificial general intelligence (AGI).
    In November 2022, OpenAI released ChatGPT, a large language model trained on GPT-3.5. 
    The system was further improved with GPT-4, released in March 2023.
    """
    
    print("Sample text for entity extraction:")
    print(sample_text)
    print("\nExtracting enriched entities...")
    
    try:
        entities = extract_entities_llm(sample_text, get_ollama_completion)
        
        print("\nExtracted entities:")
        for i, entity in enumerate(entities, 1):
            print(f"{i}. {entity['text']} (Type: {entity['type']})")
            print(f"   Description: {entity['description']}")
            print(f"   Importance: {entity['importance_score']}")
            print()
        
        print(f"Total entities extracted: {len(entities)}")
    except Exception as e:
        print(f"Error in entity extraction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 