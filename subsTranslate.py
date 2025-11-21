# Import the required modules
import os
import sys
import logging
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import openai
from dotenv import load_dotenv
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import json
import re
import time
import argparse
import subprocess
from datetime import datetime, timedelta
import pickle
from pathlib import Path
import subs_cleaner  # Import the new cleaning module
import subs_replacer  # Import the new replacement module

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

# Constants
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = os.getenv("DEEPSEEK_ENDPOINT")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = os.getenv("OPENROUTER_ENDPOINT")
DEEPSEEK_MODEL_CHAT = os.getenv("DEEPSEEK_MODEL_CHAT")
OPENROUTER_MODEL_DEEPSEEKV3FREE = os.getenv("OPENROUTER_MODEL_DEEPSEEKV3FREE")
OPENROUTER_MODEL_DEEPSEEKV30324 = os.getenv("OPENROUTER_MODEL_DEEPSEEKV30324")
OPENROUTER_MODEL_DEEPSEEKV32 = os.getenv("OPENROUTER_MODEL_DEEPSEEKV32")
OPENROUTER_MODEL_GEMINI25FLASH = os.getenv("OPENROUTER_MODEL_GEMINI25FLASH")
OPENROUTER_MODEL_GROK41FAST = os.getenv("OPENROUTER_MODEL_GROK41FAST")

CHUNK_SIZE = 100  # Number of subtitles to process in each chunk
CONFIG_FILE = "config.json"  # File to store the last used folder path
SUBTITLE_DELIMITER = "|||SUBTITLE_BREAK|||"  # Delimiter to separate subtitle blocks

# Disable OpenAI/HTTP request logging
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)  # Suppress HTTP request logs

class TranslationConfig:
    """Class to represent a provider+model configuration with retry state"""
    def __init__(self, provider_name: str, api_key: str, api_url: str, model_name: str, display_name: str):
        self.provider_name = provider_name
        self.api_key = api_key
        self.api_url = api_url
        self.model_name = model_name
        self.display_name = display_name  # For UI display
        self.remaining_retries = 3
        self.client = None
        self.failed = False  # Track if this config has failed in the batch
        self.lines_translated = 0  # Track number of lines successfully translated
        
    def initialize_client(self):
        """Initialize the OpenAI client for this configuration"""
        if not self.client:
            self.client = openai.OpenAI(api_key=self.api_key, base_url=self.api_url)
        return self.client
    
    def reset_retries(self):
        """Reset the retry count"""
        self.remaining_retries = 3

class TranslationConfigManager:
    """Class to manage multiple translation configurations"""
    def __init__(self):
        self.configs = []
        self.current_index = 0
        
    def add_config(self, config: TranslationConfig):
        """Add a configuration to the sequence"""
        self.configs.append(config)
        
    def get_current_config(self) -> Optional[TranslationConfig]:
        """Get the current active configuration, skipping failed ones"""
        while self.current_index < len(self.configs):
            config = self.configs[self.current_index]
            if not config.failed:
                return config
            self.current_index += 1
        return None
        
    def next_config(self) -> Optional[TranslationConfig]:
        """Move to the next non-failed configuration in sequence"""
        self.current_index += 1
        return self.get_current_config()
        
    def mark_current_failed(self):
        """Mark the current configuration as failed for the batch"""
        if self.current_index < len(self.configs):
            self.configs[self.current_index].failed = True
            logger.warning(f"Marking configuration {self.configs[self.current_index].display_name} as failed for this batch")
        
    def reset_all(self):
        """Reset all configurations to initial state"""
        self.current_index = 0
        for config in self.configs:
            config.reset_retries()
            
    def get_active_configs(self) -> List[TranslationConfig]:
        """Get list of non-failed configurations"""
        return [config for config in self.configs if not config.failed]

def initialize_client(api_key: str, api_url: str) -> openai.OpenAI:
    """
    Initialize an OpenAI client with the given API key and URL.
    
    Args:
        api_key (str): The API key to use
        api_url (str): The API URL to use
        
    Returns:
        openai.OpenAI: The initialized client
    """
    return openai.OpenAI(api_key=api_key, base_url=api_url)

def normalize_line_breaks(text: str) -> str:
    """
    Normalize line breaks in text to ensure consistent handling.
    Replaces any combination of \r\n, \r, or multiple \n with single \n.
    """
    # First replace \r\n with \n
    text = text.replace('\r\n', '\n')
    # Then replace any remaining \r with \n
    text = text.replace('\r', '\n')
    # Finally, replace multiple \n with single \n
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

def translate_text_with_context(text_chunk: str, config_manager: TranslationConfigManager,
                              retry_delay: int = 5, temperature: float = 0.5) -> Tuple[Optional[str], Optional[TranslationConfig]]:
    """
    Translate a chunk of text using the selected configurations with fallback logic.

    Args:
        text_chunk (str): The text to be translated.
        config_manager (TranslationConfigManager): Manager containing translation configurations
        retry_delay (int, optional): Initial delay in seconds between retries
        temperature (float, optional): Temperature for translation (0.4-1.2)

    Returns:
        Tuple[Optional[str], Optional[TranslationConfig]]: The translated text and the config that succeeded, or (None, None) if all configurations fail
    """
    current_config = config_manager.get_current_config()
    if not current_config:
        logger.error("No valid translation configuration available")
        return None, None
        
    while current_config:
        # Initialize client if needed
        client = current_config.initialize_client()
        
        while current_config.remaining_retries > 0:
            try:
                response = client.chat.completions.create(
                    model=current_config.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": f"""
                            You are a professional subtitle translator specializing in translation to English.
                            Translate the following subtitles to English while adhering to these guidelines:
                            1. Literal Precision:
                            - No paraphrasing. Retain metaphors, slang, and vulgarity.
                            - Preserve explicit language and sexual/fetish terms such as bukkake, gokkun, nakadashi, kimoi-ota, mesu.
                            2. Contextual Awareness:
                            - For Chinese: Translate naturally while preserving tone and context
                            - For Japanese: Correct Whisper errors using AV context (e.g., 'ラーン' --> 'ザーン','フェラー' --> 'フェラ', 'マンゴ' --> 'マンコ')
                            - Fix misheard words while keeping original meaning.
                            3. Jargon Retention:
                            - Keep niche terms in romaji/pinyin, e.g., onahole, dogezas, with [brief explanations].
                            4. No Softening:
                            - Use direct sexual terms, e.g., cock, pussy, cunt, cum, fuck, etc.
                            - No euphemisms or censorship.
                            5. Cultural Notes:
                            - Add concise [cultural/sexual context] where needed.
                            - Preserve honorifics and speech patterns.
                            6. Formatting - CRITICAL:
                            - IMPORTANT: The text contains subtitle blocks separated by "{SUBTITLE_DELIMITER}"
                            - You MUST preserve these delimiters EXACTLY in your translation
                            - Translate the content between delimiters, but keep the delimiters unchanged
                            - Do not enclose translated text in double quotes, or other punctuation, unless it is part of a quote from the original text.
                            - Do not add any extra lines or spaces.
                            - Each line within a subtitle block should remain on its own line
                            7. Forbidden output:
                            - Do not output original text in the translated text - unless it is part of a comment, context or an explanation
                            - Do not output any of these guidelines in the translated text.

                            Now please translate the following text. REMEMBER: Keep all "{SUBTITLE_DELIMITER}" markers unchanged:

                            {text_chunk}""",
                        },
                    ],
                    stream=False,
                    temperature=temperature,
                    timeout=30,
                )
                
                if response and response.choices and response.choices[0].message:
                    # Normalize line breaks in the response
                    translated_text = normalize_line_breaks(response.choices[0].message.content)
                    # Successful translation - reset retries for next chunk
                    current_config.reset_retries()
                    return translated_text, current_config
                else:
                    logger.warning(
                        f"Empty or invalid response from {current_config.display_name}. "
                        f"Retries remaining: {current_config.remaining_retries - 1}"
                    )
                    
            except Exception as e:
                logger.warning(
                    f"Translation attempt failed with {current_config.display_name}: {str(e)}. "
                    f"Retries remaining: {current_config.remaining_retries - 1}"
                )
            
            # Decrement retries and add delay
            current_config.remaining_retries -= 1
            if current_config.remaining_retries > 0:
                logger.info(f"Waiting {retry_delay} seconds before retry...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
        
        # Current config exhausted, mark as failed and try next one
        logger.warning(f"Configuration {current_config.display_name} exhausted. Marking as failed for batch.")
        config_manager.mark_current_failed()
        current_config = config_manager.next_config()
        if current_config:
            logger.info(f"Switching to configuration: {current_config.display_name}")
            retry_delay = 5  # Reset delay for new configuration
    
    logger.error("All translation configurations exhausted")
    return None, None


def parse_srt(file_path: str) -> List[Dict[str, str]]:
    """
    Parse an SRT file to extract subtitle blocks.

    Args:
        file_path (str): Path to the SRT file.

    Returns:
        List[Dict[str, str]]: A list of subtitle blocks, each containing index, timestamps, and text.
    """
    subtitles = []
    subtitle = {}
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Check if the line is a subtitle index (digits only)
            if line.isdigit():
                # Check if the next line contains a valid timestamp (-->)
                if i + 1 < len(lines) and "-->" in lines[i + 1]:
                    if subtitle:  # Save the previous subtitle block
                        subtitles.append(subtitle)
                    subtitle = {"index": int(line), "timestamps": lines[i + 1].strip(), "text": ""}
                    i += 2  # Skip the index and timestamp lines
                else:
                    # If the next line is not a timestamp, treat the digits as part of the text, This may not be robust.
                    if subtitle:
                        subtitle["text"] += line + "\n"
                    i += 1
            else:
                # Add non-index lines to the subtitle text
                if subtitle:
                    subtitle["text"] += line + "\n"
                i += 1

        if subtitle:  # Save the last subtitle block
            subtitles.append(subtitle)

    except Exception as e:
        logger.error(f"Error parsing SRT file {file_path}: {e}")
        raise

    return subtitles


def parse_timestamp(timestamp: str) -> Tuple[int, int, int, int]:
    """
    Parse an SRT timestamp into hours, minutes, seconds, and milliseconds.
    
    Args:
        timestamp (str): Timestamp in format "HH:MM:SS,mmm"
        
    Returns:
        Tuple[int, int, int, int]: Hours, minutes, seconds, milliseconds
    """
    time_part, ms_part = timestamp.split(',')
    h, m, s = map(int, time_part.split(':'))
    ms = int(ms_part)
    return h, m, s, ms

def format_timestamp(h: int, m: int, s: int, ms: int) -> str:
    """
    Format hours, minutes, seconds, and milliseconds into an SRT timestamp.
    
    Args:
        h (int): Hours
        m (int): Minutes
        s (int): Seconds
        ms (int): Milliseconds
        
    Returns:
        str: Formatted timestamp in "HH:MM:SS,mmm" format
    """
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def add_seconds_to_timestamp(timestamp: str, seconds: int) -> str:
    """
    Add seconds to an SRT timestamp.
    
    Args:
        timestamp (str): Original timestamp in "HH:MM:SS,mmm" format
        seconds (int): Number of seconds to add
        
    Returns:
        str: New timestamp in "HH:MM:SS,mmm" format
    """
    h, m, s, ms = parse_timestamp(timestamp)
    
    # Add seconds
    s += seconds
    
    # Handle overflow
    m += s // 60
    s = s % 60
    h += m // 60
    m = m % 60
    
    return format_timestamp(h, m, s, ms)

def write_srt(file_path: str, subtitles: List[Dict[str, str]], model_name: str = "DeepSeek v3 0324", include_credits: bool = True) -> None:
    """
    Write translated subtitles back to an SRT file.

    Args:
        file_path (str): Path to the output SRT file.
        subtitles (List[Dict[str, str]]): List of subtitle blocks to write.
        model_name (str): Name of the model used for translation (default: "DeepSeek v3 0324" for backwards compatibility).
        include_credits (bool): Whether to include the credits subtitle at the end (default: True).
    """
    try:
        with open(file_path, "w", encoding="utf-8") as file:
            # Write all existing subtitles
            for subtitle in subtitles:
                file.write(f"{subtitle['index']}\n")
                file.write(f"{subtitle['timestamps']}\n")
                file.write(f"{subtitle['text'].strip()}\n\n")

            # Add the final subtitle with credits if requested
            if subtitles and include_credits:
                # Get the last timestamp and add 5 seconds
                last_timestamp = subtitles[-1]['timestamps']
                end_time = last_timestamp.split(' --> ')[1]  # Get the end time of last subtitle
                new_start_time = add_seconds_to_timestamp(end_time, 1)  # Start 1 second after last subtitle
                new_end_time = add_seconds_to_timestamp(new_start_time, 5)  # Show for 5 seconds

                # Write the final subtitle
                file.write(f"{len(subtitles) + 1}\n")
                file.write(f"{new_start_time} --> {new_end_time}\n")
                file.write("Transcribed using WhisperWithVAD_Pro.\n")
                file.write(f"Translated using {model_name}.\n\n")
    except Exception as e:
        logger.error(f"Error writing SRT file {file_path}: {e}")
        raise


def contains_untranslated_text(text: str) -> bool:
    """
    Check if the text contains untranslated lines (e.g., non-English characters).
    Ignores text in square brackets which may contain intentional non-English terms.

    Args:
        text (str): The text to check.

    Returns:
        bool: True if untranslated text is found, False otherwise.
    """
    # Remove any text in square brackets (like [laughs], [sighs], [cultural context], etc.)
    text = re.sub(r'\[.*?\]', '', text)
    
    # Regular expression to detect non-English characters (e.g., Chinese, Japanese, Korean, etc.)
    non_english_pattern = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
    return bool(non_english_pattern.search(text))


def count_untranslated_lines(subtitles: List[Dict[str, str]]) -> int:
    """
    Count the number of subtitles that contain untranslated text.

    Args:
        subtitles (List[Dict[str, str]]): List of subtitle blocks to check.

    Returns:
        int: The number of subtitles with untranslated text.
    """
    return sum(1 for subtitle in subtitles if contains_untranslated_text(subtitle["text"]))


def retry_translation_for_untranslated_lines(subtitles: List[Dict[str, str]], 
                                           config_manager: TranslationConfigManager) -> List[Dict[str, str]]:
    """
    Retry translation for subtitles that contain untranslated text using a halving strategy.
    Starts with larger chunks and progressively reduces chunk size until success or single line.
    
    Args:
        subtitles (List[Dict[str, str]]): List of subtitle blocks to check
        config_manager (TranslationConfigManager): Manager containing translation configurations
    
    Returns:
        List[Dict[str, str]]: The updated subtitles
    """
    # Reset all configurations before starting retry pass
    config_manager.reset_all()
    
    # Find indices of untranslated lines
    untranslated_indices = [
        i for i, subtitle in enumerate(subtitles) 
        if contains_untranslated_text(subtitle["text"])
    ]
    
    if not untranslated_indices:
        logger.info("No untranslated lines found. Skipping retry pass.")
        return subtitles

    current_config = config_manager.get_current_config()
    logger.info(f"Retrying translation for {len(untranslated_indices)} untranslated lines using: {current_config.display_name}")
    
    def translate_chunk(chunk_indices: List[int]) -> Optional[List[str]]:
        """Translate a chunk of subtitles and return list of translated texts."""
        if not chunk_indices:
            return None

        # Join subtitle blocks using delimiter
        text_chunk = SUBTITLE_DELIMITER.join([subtitles[idx]["text"].strip() for idx in chunk_indices])
        translated_chunk, used_config = translate_text_with_context(text_chunk, config_manager)

        if translated_chunk:
            # Split by delimiter
            translated_blocks = [block.strip() for block in translated_chunk.split(SUBTITLE_DELIMITER)]

            # Validate the number of translated blocks
            if len(translated_blocks) == len(chunk_indices):
                return translated_blocks
            else:
                logger.warning(
                    f"Delimiter-based split mismatch in retry: "
                    f"expected {len(chunk_indices)} blocks, got {len(translated_blocks)}"
                )
                logger.info(f"Translated output sample: {translated_chunk[:500]}...")
                return None
        return None

    # Process untranslated lines with progressively smaller chunks
    with tqdm(total=len(untranslated_indices), desc="Retrying translations", unit="line") as pbar:
        remaining_indices = untranslated_indices.copy()
        current_chunk_size = min(CHUNK_SIZE, len(remaining_indices))  # Use CHUNK_SIZE constant
        
        while remaining_indices and current_chunk_size > 0:
            # Process chunks of current size
            i = 0
            while i < len(remaining_indices):
                chunk_indices = remaining_indices[i:i + current_chunk_size]
                translated_lines = translate_chunk(chunk_indices)
                
                if translated_lines:
                    # Translation successful - update subtitles
                    for idx, trans_text in zip(chunk_indices, translated_lines):
                        subtitles[idx]["text"] = trans_text.strip()
                        if idx in remaining_indices:
                            remaining_indices.remove(idx)
                        pbar.update(1)
                    i += current_chunk_size
                else:
                    # If chunk_size is 1 and translation failed, keep original and mark as processed
                    if current_chunk_size == 1:
                        logger.warning(f"Translation failed for subtitle index {chunk_indices[0]}. Keeping original text.")
                        remaining_indices.remove(chunk_indices[0])
                        pbar.update(1)
                        i += 1
                    else:
                        # Move to next chunk without updating indices
                        i += current_chunk_size
            
            # Halve the chunk size for next iteration (minimum 1)
            current_chunk_size = max(current_chunk_size // 2, 1)
            
            # If we've tried with size 1 and still have remaining indices, exit
            if current_chunk_size == 1 and i >= len(remaining_indices):
                break

    return subtitles


def final_pass_for_untranslated_lines(subtitles: List[Dict[str, str]], 
                                     config_manager: TranslationConfigManager) -> List[Dict[str, str]]:
    """
    Final pass to translate any remaining untranslated lines individually.
    This is the last-ditch effort to translate lines that failed in previous passes.
    
    Args:
        subtitles (List[Dict[str, str]]): List of subtitle blocks to check
        config_manager (TranslationConfigManager): Manager containing translation configurations
    
    Returns:
        List[Dict[str, str]]: The updated subtitles
    """
    # Reset all configurations before starting final pass
    config_manager.reset_all()
    
    # Find indices of untranslated lines
    untranslated_indices = [
        i for i, subtitle in enumerate(subtitles) 
        if contains_untranslated_text(subtitle["text"])
    ]
    
    if not untranslated_indices:
        logger.info("No untranslated lines found in final pass. All lines translated.")
        return subtitles

    total_untranslated = len(untranslated_indices)
    current_config = config_manager.get_current_config()
    logger.info(f"Final pass: Translating {total_untranslated} remaining untranslated lines one by one using: {current_config.display_name}")
    
    with tqdm(total=total_untranslated, desc="Final pass translations", unit="line") as pbar:
        for idx in untranslated_indices:
            text_to_translate = subtitles[idx]["text"].strip()
            
            # Reset configuration manager for each line to ensure all configs are tried
            config_manager.reset_all()

            translated_text, used_config = translate_text_with_context(text_to_translate, config_manager)

            if translated_text and not contains_untranslated_text(translated_text):
                subtitles[idx]["text"] = translated_text.strip()
                logger.info(f"Successfully translated line {idx+1} in final pass")
            else:
                logger.warning(f"Final pass translation failed for line {idx+1}. Keeping original text.")
            
            pbar.update(1)
    
    # Check final status
    still_untranslated = sum(1 for subtitle in subtitles if contains_untranslated_text(subtitle["text"]))
    if still_untranslated > 0:
        logger.warning(f"{still_untranslated} lines remain untranslated after final pass")
    else:
        logger.info("All lines successfully translated after final pass")
        
    return subtitles


class TranslationProgress:
    """Class to track and save translation progress"""
    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self.checkpoint_path = os.path.join(folder_path, ".translation_progress")
        self.completed_files = set()
        self.current_file = None
        self.current_file_progress = 0  # Number of chunks completed in current file
        self.load_progress()

    def save_progress(self):
        """Save current progress to checkpoint file"""
        try:
            with open(self.checkpoint_path, 'wb') as f:
                pickle.dump({
                    'completed_files': self.completed_files,
                    'current_file': self.current_file,
                    'current_file_progress': self.current_file_progress
                }, f)
        except Exception as e:
            logger.warning(f"Failed to save progress: {e}")

    def load_progress(self):
        """Load progress from checkpoint file if it exists"""
        try:
            if os.path.exists(self.checkpoint_path):
                with open(self.checkpoint_path, 'rb') as f:
                    data = pickle.load(f)
                    self.completed_files = data['completed_files']
                    self.current_file = data['current_file']
                    self.current_file_progress = data['current_file_progress']
                logger.info(f"Loaded progress: {len(self.completed_files)} files completed")
        except Exception as e:
            logger.warning(f"Failed to load progress: {e}")
            self.completed_files = set()
            self.current_file = None
            self.current_file_progress = 0

    def mark_file_complete(self, filename: str):
        """Mark a file as completely translated"""
        self.completed_files.add(filename)
        self.current_file = None
        self.current_file_progress = 0
        self.save_progress()

    def cleanup(self):
        """Remove the checkpoint file when translation is complete"""
        try:
            if os.path.exists(self.checkpoint_path):
                os.remove(self.checkpoint_path)
        except Exception as e:
            logger.warning(f"Failed to cleanup progress file: {e}")


class BatchStatistics:
    """Class to track batch processing statistics"""
    def __init__(self):
        self.start_time = datetime.now()
        self.total_files = 0
        self.completed_files = 0
        self.total_lines = 0
        self.cleaned_files = 0  # Track files that were cleaned
        self.files_stats = []
        self.configurations = None

    def set_api_info(self, provider_type: str, configurations: str):
        """Set API provider and configuration information"""
        self.provider_type = provider_type
        self.configurations = configurations

    def add_file_stats(self, filename: str, processing_time: timedelta, 
                      total_lines: int, first_pass_lines: int, retry_lines: int,
                      final_pass_lines: int = 0, remaining_untranslated: int = 0,
                      was_cleaned: bool = False):  # Updated parameters
        """Add statistics for a processed file"""
        self.files_stats.append({
            'filename': filename,
            'processing_time': processing_time,
            'total_lines': total_lines,
            'first_pass_lines': first_pass_lines,
            'retry_lines': retry_lines,
            'final_pass_lines': final_pass_lines,
            'remaining_untranslated': remaining_untranslated,
            'was_cleaned': was_cleaned
        })
        self.completed_files += 1
        self.total_lines += total_lines
        if was_cleaned:
            self.cleaned_files += 1

    def get_summary(self) -> str:
        """Generate a summary of the batch processing"""
        total_time = datetime.now() - self.start_time
        
        summary = [
            "=== Batch Processing Summary ===",
            f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Processing Time: {total_time}",
            f"Provider Type: {self.provider_type}",
            f"Configurations: {self.configurations}",
            f"Total Files: {self.total_files}",
            f"Completed Files: {self.completed_files}",
            f"Cleaned Files: {self.cleaned_files}",
            f"Total Lines Processed: {self.total_lines}",
            "\n=== Individual File Statistics ==="]

        for stats in self.files_stats:
            summary.extend([
                f"\nFile: {stats['filename']}",
                f"Processing Time: {stats['processing_time']}",
                f"Total Lines: {stats['total_lines']}",
                f"First Pass Lines: {stats['first_pass_lines']}",
                f"Retry Pass Lines: {stats['retry_lines']}",
                f"Final Pass Lines: {stats['final_pass_lines']}",
                f"Remaining Untranslated: {stats['remaining_untranslated']}",
                f"Cleaned: {'Yes' if stats['was_cleaned'] else 'No'}"
            ])

        return "\n".join(summary)


def setup_file_logging(folder_path: str) -> str:
    """
    Set up logging to file with timestamp.
    Returns the path to the log file.
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(folder_path, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"translation_log_{timestamp}.txt")

    # Add file handler to logger
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    return log_file


def translate_srt_file_with_context(input_file: str, output_file: str, 
                                  config_manager: TranslationConfigManager,
                                  chunk_size: int = CHUNK_SIZE, 
                                  progress: Optional[TranslationProgress] = None,
                                  start_chunk: int = 0,
                                  clean_subtitles: bool = True,
                                  replace_subtitles: bool = True,
                                  temperature: float = 0.5) -> Dict[str, int]:
    """
    Translate an SRT file with context by sending chunks of subtitles together.
    Returns statistics about the translation process.
    
    Args:
        input_file (str): Path to the input SRT file
        output_file (str): Path to the output SRT file
        config_manager (TranslationConfigManager): Manager containing translation configurations
        chunk_size (int, optional): Number of subtitles to process in each chunk
        progress (TranslationProgress, optional): Progress tracker
        start_chunk (int, optional): Chunk to start from (for resuming)
        clean_subtitles (bool, optional): Whether to clean subtitles before translation
        replace_subtitles (bool, optional): Whether to replace text patterns after translation
        temperature (float, optional): Temperature for translation (0.4-1.2)
        
    Returns:
        Dict[str, int]: Statistics about the translation process
    """
    file_start_time = datetime.now()
    stats = {'total_lines': 0, 'first_pass_lines': 0, 'retry_lines': 0, 'final_pass_lines': 0}
    filename = os.path.basename(input_file)
    
    try:
        # Clean subtitles if enabled
        if clean_subtitles:
            logger.info(f"Cleaning subtitles for {filename}")
            temp_clean_file = input_file + '.clean.tmp'
            try:
                subs_cleaner.clean_srt_file(input_file, temp_clean_file)
                input_file = temp_clean_file  # Use cleaned file for translation
            except Exception as e:
                logger.error(f"Failed to clean subtitles for {filename}: {e}")
                # Continue with original file if cleaning fails
        
        subtitles = parse_srt(input_file)
        stats['total_lines'] = len(subtitles)

        # Create temporary output file if resuming
        temp_output_file = output_file + '.tmp'
        if start_chunk > 0 and os.path.exists(temp_output_file):
            logger.info(f"Resuming translation from chunk {start_chunk}")
            existing_subtitles = parse_srt(temp_output_file)
            subtitles[:start_chunk * chunk_size] = existing_subtitles[:start_chunk * chunk_size]

        # Reset configurations before starting
        config_manager.reset_all()

        # Track translation statistics for each config
        translation_stats = {config.display_name: 0 for config in config_manager.configs}

        with tqdm(total=len(subtitles), desc="Translating subtitles",
                 initial=start_chunk * chunk_size, unit="subtitle") as pbar:

            # First pass: Translate in chunks
            for i in range(start_chunk * chunk_size, len(subtitles), chunk_size):
                chunk = subtitles[i : i + chunk_size]
                # Join subtitle blocks using delimiter
                text_chunk = SUBTITLE_DELIMITER.join([sub["text"].strip() for sub in chunk])

                translated_chunk, used_config = translate_text_with_context(text_chunk, config_manager, temperature=temperature)

                if translated_chunk is None:
                    logger.warning(f"Translation failed for chunk starting at index {chunk[0]['index']}. Using original text.")
                    # Keep original text
                    for j, sub in enumerate(chunk):
                        if i + j < len(subtitles):
                            subtitles[i + j]["text"] = sub["text"]
                            stats['first_pass_lines'] += 1
                else:
                    # Track which config was used for this translation
                    if used_config:
                        translation_stats[used_config.display_name] += len(chunk)

                    # Split translated chunk by delimiter
                    translated_blocks = [block.strip() for block in translated_chunk.split(SUBTITLE_DELIMITER) if block.strip()]

                    # Validate we got the expected number of blocks
                    if len(translated_blocks) == len(chunk):
                        # Update subtitles with translated blocks
                        for j, block in enumerate(translated_blocks):
                            if i + j < len(subtitles):
                                subtitles[i + j]["text"] = block
                                stats['first_pass_lines'] += 1
                    else:
                        # Mismatch - log warning but try to use what we can
                        logger.warning(
                            f"Delimiter-based split mismatch for chunk {i//chunk_size}: "
                            f"expected {len(chunk)} blocks, got {len(translated_blocks)}."
                        )
                        logger.info(f"Translated output sample: {translated_chunk[:500]}...")

                        # If we have at least some translated blocks, use them where we can
                        if len(translated_blocks) > 0:
                            # Track stats for successful blocks
                            if used_config:
                                translation_stats[used_config.display_name] += min(len(translated_blocks), len(chunk))
                            for j in range(min(len(translated_blocks), len(chunk))):
                                if i + j < len(subtitles):
                                    subtitles[i + j]["text"] = translated_blocks[j]
                                    stats['first_pass_lines'] += 1
                            # For any remaining, keep original
                            for j in range(len(translated_blocks), len(chunk)):
                                if i + j < len(subtitles):
                                    subtitles[i + j]["text"] = chunk[j]["text"]
                                    stats['first_pass_lines'] += 1
                        else:
                            # No valid blocks, keep original text for this chunk
                            for j, sub in enumerate(chunk):
                                if i + j < len(subtitles):
                                    subtitles[i + j]["text"] = sub["text"]
                                    stats['first_pass_lines'] += 1

                # Save progress periodically
                if progress:
                    progress.current_file = filename
                    progress.current_file_progress = i // chunk_size
                    progress.save_progress()

                    # Save intermediate results to temporary file (without credits)
                    write_srt(temp_output_file, subtitles, include_credits=False)

                pbar.update(len(chunk))

            # Second pass: Check for untranslated lines and retry with halving strategy
            untranslated_before_retry = count_untranslated_lines(subtitles)
            subtitles = retry_translation_for_untranslated_lines(subtitles, config_manager)
            stats['retry_lines'] = untranslated_before_retry
            
            # Final pass: One-by-one translation of any remaining untranslated lines
            untranslated_before_final = count_untranslated_lines(subtitles)
            if untranslated_before_final > 0:
                logger.info(f"Starting final pass for {untranslated_before_final} remaining untranslated lines")
                subtitles = final_pass_for_untranslated_lines(subtitles, config_manager)
                stats['final_pass_lines'] = untranslated_before_final

        # Apply text replacement if enabled
        if replace_subtitles:
            logger.info(f"Applying text pattern replacement for {filename}")
            temp_translated_file = output_file + '.translated.tmp'
            temp_replaced_file = temp_translated_file + '.replaced'
            try:
                # Save translated file before replacement (without credits as this is temporary)
                write_srt(temp_translated_file, subtitles, include_credits=False)

                # Apply replacements - this will raise an exception if it fails
                subs_replacer.replace_srt_file(temp_translated_file, temp_replaced_file)

                # Parse the replaced content back into subtitle format
                subtitles = parse_srt(temp_replaced_file)
                logger.info(f"Text pattern replacement completed successfully for {filename}")

            except FileNotFoundError as e:
                logger.warning(f"Replacement list file not found for {filename}: {e}")
                logger.info(f"Skipping text replacement, using original translated text for {filename}")

            except Exception as e:
                logger.error(f"Text pattern replacement failed for {filename}: {e}")
                logger.info(f"Using original translated text for {filename}")

            finally:
                # Clean up temporary replacement files
                if os.path.exists(temp_translated_file):
                    os.remove(temp_translated_file)
                if os.path.exists(temp_replaced_file):
                    os.remove(temp_replaced_file)

        # Determine which model translated the most lines
        top_model = "DeepSeek v3 0324"  # Default fallback
        if translation_stats:
            max_lines = max(translation_stats.values())
            if max_lines > 0:
                # Find the model with the most translations
                top_model = max(translation_stats, key=translation_stats.get)
                logger.info(f"Translation statistics: {translation_stats}")
                logger.info(f"Model that translated the most lines: {top_model} ({max_lines} lines)")

        # Write final output and cleanup temporary files
        write_srt(output_file, subtitles, model_name=top_model)
        if os.path.exists(temp_output_file):
            os.remove(temp_output_file)
        if clean_subtitles and os.path.exists(input_file + '.clean.tmp'):
            os.remove(input_file + '.clean.tmp')
            
        if progress:
            progress.mark_file_complete(filename)
        
        processing_time = datetime.now() - file_start_time
        remaining_untranslated = count_untranslated_lines(subtitles)
        logger.info(
            f"Finished translating {filename} in {processing_time}. "
            f"Total lines: {stats['total_lines']}, "
            f"First pass: {stats['first_pass_lines']}, "
            f"Retry pass: {stats['retry_lines']}, "
            f"Final pass: {stats['final_pass_lines']}, "
            f"Remaining untranslated: {remaining_untranslated}, "
            f"Cleaned: {'Yes' if clean_subtitles else 'No'}, "
            f"Replaced: {'Yes' if replace_subtitles else 'No'}"
        )

        return stats

    except Exception as e:
        logger.error(f"Error translating SRT file {input_file}: {e}")
        raise


def translate_all_srt_in_folder(folder_path: str, output_folder: str, 
                              config_manager: TranslationConfigManager,
                              chunk_size: int = CHUNK_SIZE,
                              clean_subtitles: bool = True,
                              replace_subtitles: bool = True,
                              temperature: float = 0.5) -> None:
    """
    Translate all SRT files in a folder with resume capability and statistics tracking.
    
    Args:
        folder_path (str): Path to the folder containing SRT files
        output_folder (str): Path to the output folder for translated files
        config_manager (TranslationConfigManager): Manager containing translation configurations
        chunk_size (int, optional): Number of subtitles to process in each chunk
        clean_subtitles (bool, optional): Whether to clean subtitles before translation
        replace_subtitles (bool, optional): Whether to replace text patterns after translation
        temperature (float, optional): Temperature for translation (0.4-1.2)
    """
    try:
        # Set up logging to file
        log_file = setup_file_logging(folder_path)
        logger.info(f"Logging to: {log_file}")
        
        # Log configuration information
        logger.info("Using translation configurations in order:")
        for i, config in enumerate(config_manager.configs, 1):
            logger.info(f"{i}. {config.display_name}")

        # Initialize statistics tracking
        batch_stats = BatchStatistics()
        config_names = " / ".join(config.display_name for config in config_manager.configs)
        batch_stats.set_api_info("Multiple", config_names)
        
        # Initialize progress tracking
        progress = TranslationProgress(folder_path)

        # Get list of SRT files in the folder
        srt_files = [filename for filename in os.listdir(folder_path) if filename.endswith(".srt")]
        batch_stats.total_files = len(srt_files)

        if batch_stats.total_files == 0:
            logger.warning(f"No SRT files found in folder: {folder_path}")
            return

        logger.info(f"Found {batch_stats.total_files} SRT files to translate.")
        if clean_subtitles:
            logger.info("Subtitle cleaning is enabled - files will be cleaned before translation")
        if replace_subtitles:
            logger.info("Text pattern replacement is enabled - patterns from subs_replace.json will be applied after translation")

        for i, filename in enumerate(srt_files, start=1):
            # Skip if file was already completed
            if filename in progress.completed_files:
                logger.info(f"Skipping previously completed file: {filename}")
                continue

            # Log active configurations before processing each file
            active_configs = config_manager.get_active_configs()
            logger.info(f"Active configurations for file {filename}:")
            for config in active_configs:
                logger.info(f"- {config.display_name}")
            
            if not active_configs:
                logger.error("No active configurations remaining. Stopping batch processing.")
                break

            input_file = os.path.join(folder_path, filename)
            output_file = os.path.join(output_folder, f"translated_{filename}")

            # Determine starting point for this file
            start_chunk = progress.current_file_progress if filename == progress.current_file else 0
            
            file_start_time = datetime.now()
            logger.info(f"Translating file {i} of {batch_stats.total_files}: {filename}" + 
                       (f" (resuming from chunk {start_chunk})" if start_chunk > 0 else ""))
            
            # Reset configuration manager's index for each file (but keep failed status)
            config_manager.current_index = 0
            
            # Translate file and get statistics
            file_stats = translate_srt_file_with_context(
                input_file, output_file, 
                config_manager,
                chunk_size, progress, start_chunk,
                clean_subtitles=clean_subtitles,
                replace_subtitles=replace_subtitles,
                temperature=temperature
            )
            
            # Count remaining untranslated lines
            subtitles = parse_srt(output_file)
            remaining_untranslated = count_untranslated_lines(subtitles)
            
            # Add file statistics to batch statistics
            batch_stats.add_file_stats(
                filename,
                datetime.now() - file_start_time,
                file_stats['total_lines'],
                file_stats['first_pass_lines'],
                file_stats['retry_lines'],
                file_stats.get('final_pass_lines', 0),
                remaining_untranslated,
                was_cleaned=clean_subtitles
            )

        # Log final batch statistics
        logger.info("\n" + batch_stats.get_summary())
        
        # Log final configuration status
        logger.info("\nFinal configuration status:")
        for config in config_manager.configs:
            status = "FAILED" if config.failed else "ACTIVE"
            logger.info(f"- {config.display_name}: {status}")
        
        # Clean up progress file after successful completion
        progress.cleanup()

    except Exception as e:
        logger.error(f"Error processing folder {folder_path}: {e}")
        if 'batch_stats' in locals():
            logger.info("\n" + batch_stats.get_summary())
        raise


def select_folder(initial_dir: Optional[str] = None) -> Optional[str]:
    """
    Open a dialog to select a folder.

    Args:
        initial_dir (Optional[str]): The initial directory to display in the dialog.

    Returns:
        Optional[str]: The selected folder path, or None if no folder was selected.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    folder_path = filedialog.askdirectory(title="Select Folder Containing SRT Files", initialdir=initial_dir)
    return folder_path if folder_path else None


def select_translation_configs() -> Tuple[TranslationConfigManager, bool, bool, bool, float]:  # Modified return type
    """
    Create a dialog for selecting and ordering translation configurations.
    
    Returns:
        Tuple[TranslationConfigManager, bool, bool, bool, float]: Manager containing the selected configurations,
                                                         shutdown choice, clean subtitles choice,
                                                         replacement toggle choice, and temperature
    """
    root = tk.Tk()
    root.title("Translation Configuration Selection")
    root.geometry("600x650")  # Increased height for replacement toggle
    
    # Available configurations
    available_configs = [
        ("OpenRouter", "deepseek-chat-free", OPENROUTER_API_KEY, OPENROUTER_API_URL, OPENROUTER_MODEL_DEEPSEEKV3FREE),
        ("OpenRouter", "deepseek-chat-v3-0324", OPENROUTER_API_KEY, OPENROUTER_API_URL, OPENROUTER_MODEL_DEEPSEEKV30324),
        ("OpenRouter", "deepseek-v3.2-exp", OPENROUTER_API_KEY, OPENROUTER_API_URL, OPENROUTER_MODEL_DEEPSEEKV32),  # Added new model
        ("OpenRouter", "Gemini-2.5-flash", OPENROUTER_API_KEY, OPENROUTER_API_URL, OPENROUTER_MODEL_GEMINI25FLASH),  # Added new model      
        ("Deepseek", "deepseek-chat", DEEPSEEK_API_KEY, DEEPSEEK_API_URL, DEEPSEEK_MODEL_CHAT),
        ("OpenRouter", "Grok-4.1-Fast", OPENROUTER_API_KEY, OPENROUTER_API_URL, OPENROUTER_MODEL_GROK41FAST),
    ]
    
    # Create frames
    frame = tk.Frame(root)
    frame.pack(fill="both", expand=True, padx=10, pady=10)
    
    # Create listbox for selected configurations
    tk.Label(frame, text="Selected Configurations (in order of use):").pack()
    selected_listbox = tk.Listbox(frame, height=6)
    selected_listbox.pack(fill="both", expand=True, pady=5)
    
    # Create frame for available configurations
    available_frame = tk.LabelFrame(frame, text="Available Configurations")
    available_frame.pack(fill="both", expand=True, pady=10)
    
    selected_configs = []
    
    def add_config():
        """Add selected configuration to the list"""
        for var, config in zip(config_vars, available_configs):
            if var.get() and len(selected_configs) < 3:
                if config not in selected_configs:
                    selected_configs.append(config)
                    selected_listbox.insert(tk.END, f"{config[0]} - {config[1]}")
                var.set(False)
        update_buttons()
    
    def remove_config():
        """Remove selected configuration from the list"""
        selection = selected_listbox.curselection()
        if selection:
            idx = selection[0]
            selected_configs.pop(idx)
            selected_listbox.delete(idx)
        update_buttons()
    
    def move_up():
        """Move selected configuration up in the list"""
        selection = selected_listbox.curselection()
        if selection and selection[0] > 0:
            idx = selection[0]
            selected_configs[idx], selected_configs[idx-1] = selected_configs[idx-1], selected_configs[idx]
            refresh_listbox()
            selected_listbox.selection_set(idx-1)
    
    def move_down():
        """Move selected configuration down in the list"""
        selection = selected_listbox.curselection()
        if selection and selection[0] < len(selected_configs) - 1:
            idx = selection[0]
            selected_configs[idx], selected_configs[idx+1] = selected_configs[idx+1], selected_configs[idx]
            refresh_listbox()
            selected_listbox.selection_set(idx+1)
    
    def refresh_listbox():
        """Refresh the listbox contents"""
        selected_listbox.delete(0, tk.END)
        for config in selected_configs:
            selected_listbox.insert(tk.END, f"{config[0]} - {config[1]}")
    
    def update_buttons():
        """Update button states based on current selection"""
        add_button["state"] = "normal" if len(selected_configs) < 3 else "disabled"
        remove_button["state"] = "normal" if selected_listbox.curselection() else "disabled"
        up_button["state"] = "normal" if selected_listbox.curselection() and selected_listbox.curselection()[0] > 0 else "disabled"
        down_button["state"] = "normal" if selected_listbox.curselection() and selected_listbox.curselection()[0] < len(selected_configs) - 1 else "disabled"
        ok_button["state"] = "normal" if selected_configs else "disabled"
    
    # Create checkboxes for available configurations
    config_vars = []
    for provider, model, _, _, _ in available_configs:
        var = tk.BooleanVar()
        config_vars.append(var)
        cb = tk.Checkbutton(available_frame, text=f"{provider} - {model}", variable=var)
        cb.pack(anchor="w")
    
    # Create buttons
    button_frame = tk.Frame(frame)
    button_frame.pack(fill="x", pady=5)
    
    add_button = tk.Button(button_frame, text="Add Selected", command=add_config)
    add_button.pack(side="left", padx=5)
    
    remove_button = tk.Button(button_frame, text="Remove Selected", command=remove_config)
    remove_button.pack(side="left", padx=5)
    
    up_button = tk.Button(button_frame, text="Move Up", command=move_up)
    up_button.pack(side="left", padx=5)
    
    down_button = tk.Button(button_frame, text="Move Down", command=move_down)
    down_button.pack(side="left", padx=5)
    
    # Create frame for processing options
    options_frame = tk.LabelFrame(frame, text="Processing Options")
    options_frame.pack(fill="x", pady=10)
    
    # Add clean subtitles option
    clean_var = tk.BooleanVar(value=True)  # Default to True
    tk.Checkbutton(options_frame, text="Clean subtitles before translation (remove hallucinations and garbage text)", 
                   variable=clean_var).pack(anchor="w", padx=5, pady=2)
    
    # Add replacement option
    replace_var = tk.BooleanVar(value=True)  # Default to True
    tk.Checkbutton(options_frame, text="Replace text patterns after translation (using subs_replace.json)", 
                   variable=replace_var).pack(anchor="w", padx=5, pady=2)
    
    # Add temperature control
    temp_frame = tk.Frame(options_frame)
    temp_frame.pack(fill="x", padx=5, pady=2)
    tk.Label(temp_frame, text="Temperature (0.4-1.2):").pack(side="left")
    temp_var = tk.DoubleVar(value=0.5)  # Default to 0.5
    temp_scale = tk.Scale(temp_frame, from_=0.4, to=1.2, resolution=0.1, orient="horizontal", 
                         variable=temp_var, length=200)
    temp_scale.pack(side="left", padx=5)
    tk.Label(temp_frame, text="(Try 0.5 from Japanese, 0.8 from Chinese)").pack(side="left", padx=5)
    
    # Create frame for shutdown option
    shutdown_frame = tk.LabelFrame(frame, text="After Completion")
    shutdown_frame.pack(fill="x", pady=10)
    
    shutdown_var = tk.BooleanVar(value=False)
    tk.Radiobutton(shutdown_frame, text="Keep system running", variable=shutdown_var, value=False).pack(anchor="w")
    tk.Radiobutton(shutdown_frame, text="Shutdown system", variable=shutdown_var, value=True).pack(anchor="w")
    
    config_manager = TranslationConfigManager()
    
    def on_ok():
        """Create configuration manager from selections"""
        for provider, model, api_key, api_url, model_id in selected_configs:
            config = TranslationConfig(
                provider_name=provider,
                api_key=api_key,
                api_url=api_url,
                model_name=model_id,
                display_name=f"{provider} - {model}"
            )
            config_manager.add_config(config)
        root.destroy()
    
    ok_button = tk.Button(frame, text="OK", command=on_ok)
    ok_button.pack(pady=10)
    
    update_buttons()
    selected_listbox.bind('<<ListboxSelect>>', lambda e: update_buttons())
    
    # Center the window
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry('{}x{}+{}+{}'.format(width, height, x, y))
    
    root.mainloop()
    
    if not config_manager.configs:
        logger.error("No configurations selected. Exiting.")
        sys.exit(1)
    
    return config_manager, shutdown_var.get(), clean_var.get(), replace_var.get(), temp_var.get()


def load_last_used_settings() -> Dict[str, str]:
    """
    Load the last used settings from the config file.

    Returns:
        Dict[str, str]: A dictionary containing the last used settings.
    """
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as file:
                config = json.load(file)
                return config
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
    return {
        "last_used_folder": None, 
        "primary_provider": None, 
        "primary_model": None, 
        "retry_provider": None, 
        "retry_model": None,
        "temperature": 0.5  # Default temperature
    }


def load_last_used_folder() -> Optional[str]:
    """
    Load the last used folder path from the config file.
    
    Returns:
        Optional[str]: The last used folder path, or None if the config file does not exist.
    """
    settings = load_last_used_settings()
    return settings.get("last_used_folder")


def save_settings(settings: Dict[str, str]) -> None:
    """
    Save settings to the config file.

    Args:
        settings (Dict[str, str]): The settings to save.
    """
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as file:
            json.dump(settings, file)
    except Exception as e:
        logger.error(f"Error saving config file: {e}")


def shutdown_windows(countdown_seconds: int = 10) -> None:
    """
    Initiate Windows shutdown with a countdown.
    
    Args:
        countdown_seconds (int): Number of seconds to wait before shutdown
    """
    logger.info(f"Initiating shutdown countdown: {countdown_seconds} seconds")
    
    for remaining in range(countdown_seconds, 0, -1):
        sys.stdout.write(f"\rShutting down in {remaining} seconds... Press Ctrl+C to cancel ")
        sys.stdout.flush()
        time.sleep(1)
    
    print("\nExecuting shutdown...")
    subprocess.run(["shutdown", "/s", "/t", "0"])


def process_slowed_srt_files(folder_path: str) -> None:
    """
    Process all SRT files in the folder that end with '_sl0w.srt'.
    These files contain subtitles from videos that were slowed to 0.6x speed.
    The function speeds up the subtitles by adjusting timestamps and saves them
    back to the original files, keeping the '_sl0w.srt' suffix.
    
    Args:
        folder_path (str): Path to the folder containing SRT files
    """
    logger.info("Checking for slowed SRT files that need speed adjustment...")
    
    # Find all slowed SRT files
    slowed_files = [f for f in os.listdir(folder_path) 
                   if f.endswith('_sl0w.srt')]
    
    if not slowed_files:
        logger.info("No slowed SRT files found.")
        return
        
    logger.info(f"Found {len(slowed_files)} slowed SRT files to process.")
    
    for filename in slowed_files:
        input_file = os.path.join(folder_path, filename)
        
        logger.info(f"Processing slowed file: {filename}")
        
        try:
            # Read and parse the slowed SRT file
            subtitles = parse_srt(input_file)
            
            # Log first few timestamps before adjustment
            logger.info(f"First few timestamps before adjustment in {filename}:")
            for i, sub in enumerate(subtitles[:3]):
                logger.info(f"Subtitle {i+1}: {sub['timestamps']}")
            
            # Adjust timestamps for each subtitle
            for subtitle in subtitles:
                # Parse start and end times
                start_time, end_time = subtitle['timestamps'].split(' --> ')
                
                # Convert timestamps to seconds, adjust speed, and back to SRT format
                def adjust_timestamp(timestamp: str) -> str:
                    h, m, s = map(int, timestamp.split(',')[0].split(':'))
                    ms = int(timestamp.split(',')[1])
                    total_seconds = (h * 3600 + m * 60 + s + ms / 1000) * 0.6
                    
                    new_h = int(total_seconds // 3600)
                    total_seconds %= 3600
                    new_m = int(total_seconds // 60)
                    total_seconds %= 60
                    new_s = int(total_seconds)
                    new_ms = int((total_seconds - int(total_seconds)) * 1000)
                    
                    return f"{new_h:02d}:{new_m:02d}:{new_s:02d},{new_ms:03d}"
                
                # Update timestamps
                new_start = adjust_timestamp(start_time)
                new_end = adjust_timestamp(end_time)
                subtitle['timestamps'] = f"{new_start} --> {new_end}"
            
            # Log first few timestamps after adjustment
            logger.info(f"First few timestamps after adjustment in {filename}:")
            for i, sub in enumerate(subtitles[:3]):
                logger.info(f"Subtitle {i+1}: {sub['timestamps']}")
            
            # Write the adjusted subtitles back to the same file (without credits - this is an input file)
            write_srt(input_file, subtitles, include_credits=False)
            logger.info(f"Successfully adjusted speed for file: {filename}")
            
        except Exception as e:
            logger.error(f"Error processing slowed file {filename}: {e}")
            continue

def main():
    """
    Main function to handle folder selection and translation process.
    """
    # Load the last used settings
    settings = load_last_used_settings()
    last_used_folder = settings.get("last_used_folder")
    
    # Prompt the user to select API providers and models
    config_manager, shutdown_choice, clean_choice, replace_choice, temperature = select_translation_configs()
    
    # Update settings with selected configurations
    settings["last_used_folder"] = last_used_folder
    settings["configurations"] = [
        {
            "provider": config.provider_name,
            "model": config.model_name,
            "display_name": config.display_name
        }
        for config in config_manager.configs
    ]
    settings["temperature"] = temperature
    
    # Prompt the user to select the input folder
    folder_path = select_folder(initial_dir=last_used_folder)
    if not folder_path:
        logger.error("No folder selected. Exiting.")
        return

    # Save the selected folder and other settings
    save_settings(settings)

    # First, process any slowed SRT files
    process_slowed_srt_files(folder_path)

    # Create the "Translated" sub-folder
    output_folder = os.path.join(folder_path, "Translated")
    os.makedirs(output_folder, exist_ok=True)
    logger.info(f"Translated files will be saved in: {output_folder}")

    # Translate all SRT files in the folder
    try:
        translate_all_srt_in_folder(
            folder_path, 
            output_folder, 
            config_manager,
            clean_subtitles=clean_choice,
            replace_subtitles=replace_choice,
            temperature=temperature
        )
        
        if shutdown_choice:
            logger.info("Translation completed. Preparing for system shutdown.")
            shutdown_windows()
            
    except KeyboardInterrupt:
        logger.info("Process interrupted by user. Shutdown cancelled.")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Script failed: {e}")
        if shutdown_choice:
            logger.warning("Shutdown cancelled due to script error.")


if __name__ == "__main__":
    main()
