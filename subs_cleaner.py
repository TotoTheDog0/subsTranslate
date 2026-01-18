#!/usr/bin/env python3
import json
import re
import os
import srt
from datetime import timedelta
from typing import List, Optional

def load_garbage_list(garbage_list_path: Optional[str] = None) -> List[str]:
    """
    Load the garbage list from JSON file.
    
    Args:
        garbage_list_path (Optional[str]): Path to garbage list JSON file. 
            If None, will look in same directory as script.
            
    Returns:
        List[str]: List of garbage patterns to filter out
    """
    if garbage_list_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        garbage_list_path = os.path.join(script_dir, 'garbage_list.json')
    
    try:
        with open(garbage_list_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load garbage list from {garbage_list_path}: {e}")

def clean_subtitles(srt_content: str, garbage_list: List[str]) -> str:
    """
    Clean subtitle content by removing hallucinations, repetitions and garbage text.
    
    Args:
        srt_content (str): Raw SRT file content
        garbage_list (List[str]): List of garbage patterns to filter out
        
    Returns:
        str: Cleaned SRT content
    """
    # Parse the SRT content
    try:
        subs = list(srt.parse(srt_content))
    except Exception as e:
        raise RuntimeError(f"Failed to parse SRT content: {e}")

    i = 0
    while i < len(subs):
        # Merge consecutive identical lines
        if i < len(subs) - 1 and subs[i].content == subs[i+1].content and (subs[i+1].start - subs[i].end).seconds < 0.4:
            subs[i].end = subs[i+1].end
            del subs[i+1]
            continue

        text = subs[i].content

        # Split the text into phrases and remove repetitions
        phrases = text.split('、')
        non_repeating_phrases = []
        has_repetitions = False

        for j in range(len(phrases)):
            if j == 0 or phrases[j].strip() != phrases[j-1].strip():
                non_repeating_phrases.append(phrases[j])
            else:
                has_repetitions = True

        text_without_repetitions = '、'.join(non_repeating_phrases)

        # If there were repetitions, adjust duration
        if has_repetitions:
            subs[i].end = subs[i].start + timedelta(milliseconds=100)

        # Remove repeated words/phrases
        pattern = r"([^\s]+)(\s*\1){3,}"
        while re.search(pattern, text_without_repetitions):
            text_without_repetitions = re.sub(pattern, r"\1", text_without_repetitions)

        if text_without_repetitions != subs[i].content:
            # Calculate new duration based on text length
            duration = len(text_without_repetitions) * 0.040 * 1000  # milliseconds
            start_time = subs[i].start
            end_time = start_time + timedelta(milliseconds=duration)
            
            new_sub = srt.Subtitle(i+1, start_time, end_time, text_without_repetitions)
            subs[i] = new_sub

        i += 1

    # Define patterns to remove
    patterns = [r'★.*?★', r'「.*?」', r'【.*?】', '^「', '^★']

    # Remove subtitles matching patterns or garbage list
    subs = [sub for sub in subs if not any(re.search(pattern, sub.content) for pattern in patterns)]
    subs = [sub for sub in subs if re.sub(r'\W+', '', sub.content).strip() not in 
            map(lambda s: re.sub(r'\W+', '', s).strip(), garbage_list)]

    # Remove empty subtitles and renumber
    subs_cleaned = [s for s in subs if s.content.strip() != '']
    for i, sub in enumerate(subs_cleaned):
        sub.index = i + 1

    # Compose back to SRT format
    return srt.compose(subs_cleaned)

def clean_srt_file(input_file: str, output_file: str, garbage_list: Optional[List[str]] = None) -> None:
    """
    Clean an SRT file and save the results.

    Args:
        input_file (str): Path to input SRT file
        output_file (str): Path to save cleaned SRT file
        garbage_list (Optional[List[str]]): List of garbage patterns. If None, will load from default location.
    """
    if garbage_list is None:
        garbage_list = load_garbage_list()

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            srt_content = f.read()
    except Exception as e:
        raise RuntimeError(f"Failed to read input file {input_file}: {e}")

    cleaned_content = clean_subtitles(srt_content, garbage_list)

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
    except Exception as e:
        raise RuntimeError(f"Failed to write output file {output_file}: {e}")


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Clean SRT subtitle files by removing hallucinations, repetitions, and garbage text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python subs_cleaner.py input.srt output.srt
  python subs_cleaner.py input.srt output.srt --garbage-list custom_garbage.json
        """
    )

    parser.add_argument("input_file", help="Path to input SRT file")
    parser.add_argument("output_file", help="Path to output SRT file")
    parser.add_argument(
        "--garbage-list",
        help="Path to custom garbage list JSON file (optional, defaults to garbage_list.json in script directory)"
    )

    args = parser.parse_args()

    try:
        garbage_list = None
        if args.garbage_list:
            garbage_list = load_garbage_list(args.garbage_list)

        clean_srt_file(args.input_file, args.output_file, garbage_list)
        print(f"Successfully cleaned subtitles: {args.input_file} -> {args.output_file}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)