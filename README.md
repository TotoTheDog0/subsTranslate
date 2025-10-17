# subsTranslate

A powerful, AI-powered subtitle translation tool that translates SRT subtitle files using various LLM providers (DeepSeek, OpenRouter) with intelligent fallback mechanisms, automatic cleaning, and text replacement capabilities.

## Purpose

subsTranslate is designed to translate subtitle files (SRT format) from various languages to English with high accuracy and cultural sensitivity. The tool is particularly optimized for Asian language content (Japanese, Chinese) and includes specialized handling for adult video (AV) content terminology, slang, and cultural context.

The tool provides:
- Batch processing of multiple SRT files
- Intelligent retry mechanisms with multiple AI model fallbacks
- Automatic subtitle cleaning to remove hallucinations and garbage text
- Pattern-based text replacement for consistency
- Resume capability for interrupted translations
- Detailed logging and statistics

## Features

### Core Translation Features
- **Multiple LLM Support**: Compatible with DeepSeek (direct API) and OpenRouter (multiple models including DeepSeek, Gemini, Horizon)
- **Intelligent Fallback**: Automatically switches between configured models if one fails
- **Context-Aware Translation**: Translates subtitles in chunks to maintain context
- **Three-Pass Translation**: Initial chunk-based pass, retry pass with halving strategy, and final individual line pass
- **Temperature Control**: Adjustable temperature (0.4-1.2) for translation creativity

### Subtitle Processing
- **Automatic Cleaning**: Removes hallucinations, garbage text, and transcription errors
- **Text Pattern Replacement**: Applies custom find-and-replace patterns post-translation
- **Speed Adjustment**: Handles subtitles from slowed-down videos (0.6x speed adjustment)
- **Untranslated Line Detection**: Identifies and retries lines containing non-English characters

### Workflow Features
- **Batch Processing**: Process entire folders of SRT files
- **Resume Capability**: Automatically resume interrupted translations
- **Progress Tracking**: Visual progress bars with tqdm
- **Comprehensive Logging**: Detailed logs with timestamps for each batch
- **Statistics Reporting**: Complete batch statistics including timing and line counts
- **System Shutdown**: Optional automatic shutdown after completion

### Translation Quality
- **Literal Precision**: Preserves metaphors, slang, and explicit language
- **Cultural Awareness**: Maintains honorifics, speech patterns, and cultural context
- **Jargon Retention**: Keeps niche terms in romaji/pinyin with brief explanations
- **No Censorship**: Uses direct sexual terms without euphemisms (for AV content)
- **Whisper Error Correction**: Fixes common transcription errors in Japanese audio

## System Requirements

### Operating System
- Windows (tested)
- Linux (should work, untested)
- macOS (should work, untested)

### Python
- Python 3.8 or higher
- tkinter support (usually included with Python installation)

### API Keys
You'll need API keys for at least one of the following:
- **DeepSeek API** (direct): https://platform.deepseek.com/
- **OpenRouter API**: https://openrouter.ai/

### Hardware
- Minimum 4GB RAM recommended
- Internet connection required for API calls
- Storage space for logs and translated files

## Installation

### Option 1: Installation with Virtual Environment (Recommended)

A virtual environment keeps your project dependencies isolated from other Python projects.

1. **Clone or download the project**:
   ```bash
   cd "D:\path\to\repo\subsTranslate"
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On Linux/macOS:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Configure environment variables**:
   - Copy `.env.template` and rename it to `.env` file in the project directory
   - Add your API keys

6. **Configure optional features**:
   - `garbage_list.json`: List of garbage text patterns to remove during cleaning
   - `subs_replace.json`: Find-and-replace patterns for post-translation text correction

### Option 2: Installation without Virtual Environment

If you prefer to install globally or already manage your Python environment:

1. **Navigate to the project directory**:
   ```bash
   cd "D:\path\to\repo\subsTranslate"
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables** (same as step 5 above)

4. **Configure optional features** (same as step 6 above)

## Usage Guide

### Basic Usage

1. **Start the application**:
   ```bash
   python subsTranslate.py
   ```

2. **Select Translation Models**:
   - A dialog window will appear
   - Check the models you want to use (you can select up to 3)
   - Click "Add Selected" to add them to the configuration
   - Use "Move Up" and "Move Down" to set the priority order
   - The tool will use these models in order, falling back to the next if one fails

3. **Configure Processing Options**:
   - **Clean subtitles**: Remove hallucinations and garbage text (recommended)
   - **Replace text patterns**: Apply find-and-replace patterns from `subs_replace.json` (recommended)
   - **Temperature**: Adjust between 0.4-1.2 (0.5 for Japanese, 0.8 for Chinese recommended)

4. **Select Shutdown Option**:
   - Choose whether to keep the system running or shutdown after completion

5. **Select Input Folder**:
   - Browse to the folder containing your SRT files
   - The tool will remember your last used folder

6. **Monitor Progress**:
   - Watch the console for progress bars and status messages
   - Translated files are saved in a `Translated` subfolder
   - Logs are saved in a `logs` subfolder

### Advanced Features

#### Resume Interrupted Translation
If translation is interrupted (power loss, network issue, etc.):
- Simply restart the application
- Select the same folder
- The tool will automatically detect and resume from where it left off

#### Processing Slowed Subtitles
If you have subtitles from 0.6x slowed videos (files ending in `_sl0w.srt`):
- The tool automatically detects these files
- Timestamps are adjusted back to normal speed before translation

#### Customizing Subtitle Cleaning
Edit `garbage_list.json` to add patterns to remove:
```json
{
  "patterns": [
    "unwanted phrase",
    "another pattern"
  ]
}
```

#### Customizing Text Replacement
Edit `subs_replace.json` to add find-and-replace patterns:
```json
{
  "replacements": [
    {
      "find": "incorrect term",
      "replace": "correct term"
    }
  ]
}
```

### Batch Statistics
After completion, the tool provides detailed statistics:
- Total processing time
- Files processed
- Lines translated per pass
- Remaining untranslated lines
- Individual file timing

All statistics are saved in the log files for reference.

## Troubleshooting

### API Connection Issues

**Problem**: "Translation failed" errors
**Solutions**:
- Verify your API keys in the `.env` file
- Check your internet connection
- Ensure API endpoints are correct
- Check API rate limits or quota

**Problem**: All models failing
**Solutions**:
- Add more fallback models to your configuration
- Check if your API keys are valid and have credits
- Verify model names in the `.env` file match provider documentation

### Translation Quality Issues

**Problem**: Some lines remain untranslated
**Solutions**:
- The tool automatically retries untranslated lines
- Check the logs to see why specific lines failed
- Consider adjusting temperature settings
- Try a different model configuration

**Problem**: Translations are too literal or too creative
**Solutions**:
- Adjust the temperature setting (lower = more literal, higher = more creative)
- Japanese content: try 0.5 temperature
- Chinese content: try 0.8 temperature

### File Processing Issues

**Problem**: "No SRT files found"
**Solutions**:
- Verify files have `.srt` extension
- Check folder permissions
- Ensure files are not corrupted or empty

**Problem**: Encoding errors
**Solutions**:
- Ensure SRT files are UTF-8 encoded
- Re-save files with UTF-8 encoding using a text editor

### Performance Issues

**Problem**: Translation is very slow
**Solutions**:
- API calls naturally take time
- Some models are faster than others (DeepSeek-chat-free is slower)
- Network latency affects speed
- Consider using faster models like Gemini-2.5-flash

**Problem**: Script crashes or hangs
**Solutions**:
- Check log files for error details
- Ensure sufficient RAM is available
- Close other applications
- Update Python and dependencies to latest versions

### Module Import Errors

**Problem**: "ModuleNotFoundError: No module named 'openai'" (or other modules)
**Solutions**:
- Ensure you've installed requirements: `pip install -r requirements.txt`
- If using venv, ensure it's activated
- Try reinstalling: `pip install --upgrade -r requirements.txt`

**Problem**: "No module named 'subs_cleaner'" or "No module named 'subs_replacer'"
**Solutions**:
- Ensure `subs_cleaner.py` and `subs_replacer.py` are in the same directory
- Check file permissions
- Verify files are not corrupted

### Configuration Issues

**Problem**: Can't load `.env` file
**Solutions**:
- Ensure file is named exactly `.env` (not `.env.txt`)
- Verify file is in the same directory as `subsTranslate.py`
- Check file permissions
- On Windows, use a proper editor to create the file (not Notepad which may add extensions)

## License

Copyright 2025

This project is licensed under the **Apache License 2.0**.

### What this means:
- ✅ You can use this software for any purpose (commercial or non-commercial)
- ✅ You can modify the source code
- ✅ You can distribute the software
- ✅ You can distribute modified versions
- ✅ You can sublicense (include in proprietary software)
- ⚠️ You must include the license and copyright notice
- ⚠️ You must state significant changes made to the code
- ⚠️ You must include the NOTICE file if one exists
- ❌ The software is provided "as is" with no warranty
- ❌ The authors are not liable for any damages

See the full license text at: http://www.apache.org/licenses/LICENSE-2.0

## Credits

### Original Development
- **Original Script**: porgate55555 (Akiba-Online.com, August 1, 2025)
  - Original forum post: https://www.akiba-online.com/threads/whisper-and-its-many-forms.2142559/post-4867905
- **Modifications and Enhancements**: Novus.Toto

### Technologies Used
- **OpenAI Python SDK**: For API integration with LLM providers
- **python-dotenv**: For environment variable management
- **tqdm**: For progress bars and visual feedback
- **tkinter**: For GUI dialogs and user interaction

### AI Models Supported
- **DeepSeek**: DeepSeek-chat, DeepSeek-v3-0324, DeepSeek-v3.2-exp, DeepSeek Horizon Alpha
- **Google Gemini**: Gemini 2.5 Flash (via OpenRouter)
- **OpenRouter**: Unified API access to multiple models

### Special Thanks
- The Akiba-Online community for the original concept
- The open-source community for the essential Python libraries

### Related Tools
This project works best with:
- **WhisperWithVAD_Pro**: For initial subtitle transcription from video
- Any SRT-compatible subtitle editor for manual refinement

---

**Note**: This tool is optimized for adult content translation and maintains explicit language. Ensure you comply with local laws and content policies when using this software.
