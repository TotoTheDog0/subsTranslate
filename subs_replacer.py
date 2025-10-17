#!/usr/bin/env python3
import json
import re
import os
import srt
from typing import Dict, List, Optional

def load_replacement_list(replacement_list_path: Optional[str] = None) -> Dict[str, str]:
    """
    Load the replacement list from JSON file.
    
    Args:
        replacement_list_path (Optional[str]): Path to replacement list JSON file. 
            If None, will look in same directory as script.
            
    Returns:
        Dict[str, str]: Dictionary mapping patterns to replacements
    """
    if replacement_list_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        replacement_list_path = os.path.join(script_dir, 'subs_replace.json')
    
    try:
        with open(replacement_list_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load replacement list from {replacement_list_path}: {e}")

def compile_replacement_patterns(replacement_dict: Dict[str, str]) -> List[tuple]:
    """
    Compile replacement patterns from the dictionary into regex patterns.
    
    Args:
        replacement_dict (Dict[str, str]): Raw replacement dictionary from JSON
        
    Returns:
        List[tuple]: List of (compiled_regex, replacement_string) tuples
    """
    compiled_patterns = []
    
    for pattern_key, replacement in replacement_dict.items():
        # Handle alternation patterns (e.g., "ah|ahh|ahhh")
        alternatives = [alt.strip() for alt in pattern_key.split('|')]
        
        # Escape special regex characters in each alternative
        escaped_alternatives = [re.escape(alt) for alt in alternatives]
        
        # Create regex pattern with word boundaries to avoid partial matches
        regex_pattern = r'\b(?:' + '|'.join(escaped_alternatives) + r')\b'
        
        try:
            compiled_regex = re.compile(regex_pattern, re.IGNORECASE)
            compiled_patterns.append((compiled_regex, replacement))
        except re.error as e:
            print(f"Warning: Failed to compile regex for pattern '{pattern_key}': {e}")
            continue
    
    return compiled_patterns

def apply_replacements(text: str, compiled_patterns: List[tuple]) -> str:
    """
    Apply all replacement patterns to the given text.
    
    Args:
        text (str): Input text to process
        compiled_patterns (List[tuple]): List of (compiled_regex, replacement) tuples
        
    Returns:
        str: Text with replacements applied
    """
    result = text
    
    for regex, replacement in compiled_patterns:
        result = regex.sub(replacement, result)
    
    # Clean up extra whitespace that might result from replacements
    result = re.sub(r'\s+', ' ', result).strip()
    
    return result

def replace_subtitles(srt_content: str, replacement_dict: Optional[Dict[str, str]] = None) -> str:
    """
    Apply string replacements to subtitle content.
    
    Args:
        srt_content (str): Raw SRT file content
        replacement_dict (Optional[Dict[str, str]]): Replacement dictionary. 
            If None, will load from default location.
        
    Returns:
        str: SRT content with replacements applied
    """
    if replacement_dict is None:
        replacement_dict = load_replacement_list()
    
    # Compile replacement patterns
    compiled_patterns = compile_replacement_patterns(replacement_dict)
    
    # Parse the SRT content
    try:
        subs = list(srt.parse(srt_content))
    except Exception as e:
        raise RuntimeError(f"Failed to parse SRT content: {e}")
    
    # Apply replacements to each subtitle
    for sub in subs:
        sub.content = apply_replacements(sub.content, compiled_patterns)
    
    # Remove empty subtitles that might result from replacements
    subs_filtered = [sub for sub in subs if sub.content.strip()]
    
    # Renumber subtitles
    for i, sub in enumerate(subs_filtered):
        sub.index = i + 1
    
    # Compose back to SRT format
    return srt.compose(subs_filtered)

def replace_srt_file(input_file: str, output_file: str, replacement_dict: Optional[Dict[str, str]] = None) -> None:
    """
    Apply replacements to an SRT file and save the results.
    
    Args:
        input_file (str): Path to input SRT file
        output_file (str): Path to save processed SRT file
        replacement_dict (Optional[Dict[str, str]]): Replacement dictionary. 
            If None, will load from default location.
    """
    if replacement_dict is None:
        replacement_dict = load_replacement_list()
        
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            srt_content = f.read()
    except Exception as e:
        raise RuntimeError(f"Failed to read input file {input_file}: {e}")
        
    replaced_content = replace_subtitles(srt_content, replacement_dict)
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(replaced_content)
    except Exception as e:
        raise RuntimeError(f"Failed to write output file {output_file}: {e}")

# Example usage and testing
if __name__ == "__main__":
    # Test the replacement functionality
    test_srt = """1
00:00:01,000 --> 00:00:03,000
ah this is a test

2
00:00:04,000 --> 00:00:06,000
umm yeah this works

3
00:00:07,000 --> 00:00:09,000
haha that's awesome
"""
    
    print("Original SRT:")
    print(test_srt)
    print("\nAfter replacements:")
    print(replace_subtitles(test_srt))
