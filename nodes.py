import os
import subprocess
import inspect
import importlib
import re
import json

DEBUG_MODE = False

CLASS_WARNING_STRING = """
>>>>>> WARNING <<<<<<

You have not changed the class name of 'SilverRunPythonCode'.
This means bad actors can make you run malicious code by infiltrating this node into a workflow that you might download and run without checking if it contains this node with malicious code in it.

To solve this security flaw you need to do the following:

- Create a ~10 length random string with (a-Z,0-9) characters (ex: f8Rca4Cx)
- Add the random string as suffix to the class name in the following lines in  \custom_nodes\ComfyUI-RunPythonCode\ nodes.py:
    - class SilverRunPythonCode:   (line 808)
    - "SILVER.SilverRunPythonCode": SilverRunPythonCode,   (line 1220)
    - "SILVER.SilverRunPythonCode": "[Silver] Run/Execute Python Code",   (line 1230)
    Ex:
    - class SilverRunPythonCodef8Rca4Cx:
    - "SILVER.SilverRunPythonCodef8Rca4Cx": SilverRunPythonCodef8Rca4Cx,
    - "SILVER.SilverRunPythonCodef8Rca4Cx": "[Silver] Run/Execute Python Code",

With this, the security of this node is no longer a concern when downloading and running shared workflows - unless the bad actor somehow guesses your suffix correctly.
Avoid ever sharing an output with metadata/workflow with someone else if it contains your run python node in it because that will reveal its class name.

-------------------------------------\n\n\n
"""

SHARING_DATA_INFO_STRING = """
---------- Sharing Data in "Run Python Code" Nodes ----------
Your "Run Python Code" node offers two ways to share Python variables, functions, and objects: shared_globals and shared_locals.

shared_globals (Universal Access)

    What it is: A global container for anything you want any "Run Python Code" node in your ComfyUI session to access, regardless of connections or workflow.
    
    How it works: At the start of your code, shared_globals is an empty dictionary. Its purpose is to assign data into it that will then become globally available.
    
    To assign a global item:
    
        def my_utility_function():
            return "Hello from global!"
        
        shared_globals['my_func'] = my_utility_function
        shared_globals['version'] = 1.0
    
    To access a global item in another "Run Python Code" node: Once assigned in one node, you MUST directly use the item by its name in any other node:
    
        # In a different node, after the first one has run:
        print(my_func()) # Outputs: Hello from global!
        print(version)   # Outputs: 1.0
    
    To remove a global item use the 'remove_global_item(key: str) -> bool' function:
    
        success = remove_global_item("item_name") # item will still be available in the scope of the node until the end of its execution
    
    Key Point: If you update a global item, the change is reflected everywhere in subsequent executions.


shared_locals (Connected Node Access)

    What it is: A container specifically for passing data between directly connected "Run Python Code" nodes.
    
    How it works: If a previous "Run Python Code" node is connected to this node's shared_locals input, this dictionary will contain any items that node decided to share. If no node is connected, shared_locals will be empty.
    
    To assign a local item for the next connected node:
    
        def process_data(data):
            return data + "_processed"
    
        shared_locals['processor'] = process_data
        shared_locals['intermediate_result'] = "some_data"
    
    To access a local item from the previous connected node:
    
        # In the next connected node:
        result = processor("my_input") # Uses 'processor' from the previous node
        print(intermediate_result)     # Uses 'intermediate_result' from the previous node
    
    Key Point: shared_locals only travels along the specific connection between nodes. It doesn't affect other parts of your workflow or persist globally.
"""

# --- Global Registries ---
_EXEC_GLOBALS = {}
_GLOBAL_FUNCTIONS_REGISTRY = {}
_GLOBAL_FUNCTION_DISPLAY_STRINGS = []
_GLOBAL_OBJECTS_REGISTRY = {}
_GLOBAL_OBJECT_DISPLAY_STRINGS = []
TOP_MODULE_IMPORTS = {}
TOP_MODULE_IMPORT_STRING = ""
EXTRA_MODULE_IMPORT_STRING = ""
SHARED_GLOBALS = {}


def global_func(func):
    """
    A decorator to mark functions as 'global_func' and register them.
    Functions decorated with @global_func will automatically be added to
    _GLOBAL_FUNCTIONS_REGISTRY.
    """
    if not callable(func):
        raise TypeError(f"Decorator 'global_func' can only be applied to functions, got {type(func)}")
    
    _GLOBAL_FUNCTIONS_REGISTRY[func.__name__] = func
    
    # Use inspect.signature to get the function's signature
    signature = inspect.signature(func)
    
    # Format it as a string: "function_name(param1: type, ...) -> return_type"
    # We include the function name here for clarity in the display string
    _GLOBAL_FUNCTION_DISPLAY_STRINGS.append(f"{func.__name__}{signature}\n")
    
    return func

def global_obj(name, obj):
    """
    Registers a top-level object as 'global_obj'.
    This function should be called immediately after the object's definition.
    """
    if not isinstance(name, str):
        raise TypeError("The 'name' argument for global_object must be a string.")
    
    _GLOBAL_OBJECTS_REGISTRY[name] = obj
    
    # Get the type name (e.g., 'int', 'str', 'list')
    type_name = type(obj).__name__
    
    _GLOBAL_OBJECT_DISPLAY_STRINGS.append(f"{name} ({type_name})\n")
    return obj # Return the object so you can still assign it normally


@global_func
def remove_global_item(key: str) -> bool:
    """
    Removes an entry from the global SHARED_GLOBALS dictionary.

    Args:
        key (str): The key of the item to be removed.
    
    Returns:
        A bool that represents removal success 
    """
    global SHARED_GLOBALS
    if key in SHARED_GLOBALS:
        del SHARED_GLOBALS[key]
        print(f"[SilverRunPythonCode] Successfully removed '{key}' from SHARED_GLOBALS.")
        return True
    print(f"[SilverRunPythonCode] Warning: Key '{key}' not found in SHARED_GLOBALS. Nothing removed.")
    return False


NODE_FILE = global_obj("NODE_FILE", os.path.abspath(__file__)) # leaving this just as an example on how to decorate variables from within the script

import math
import random
import copy
import traceback

from pathlib import Path
from typing import TypedDict, Dict, Tuple

import numpy as np
import torch

import PIL
from PIL import Image, ImageOps, ImageFont, ImageDraw

import comfy
import folder_paths


_packages_to_check = [
    ("cv2", "cv2"),
    ("diffusers", "diffusers"),
    ("ffmpeg", "ffmpeg"),
    ("ffmpegcv", "ffmpegcv"),
    ("kornia", "kornia"),
    ("mediapipe", "mediapipe"),
    ("mmcv", "mmcv"),
    ("moviepy", "moviepy"),
    ("natsort", "natsort"),
    ("ninja", "ninja"),
    ("onnx", "onnx"),
    ("onnxruntime", "onnxruntime"),
    ("pandas", "pd"),
    ("pydub", "pydub"),
    ("rembg", "rembg"),
    ("requests", "requests"),
    ("safetensors", "safetensors"),
    ("scipy", "scipy"),
    ("sklearn", "sklearn"),
    ("tokenizers", "tokenizers"),
    ("torchaudio", "torchaudio"),
    ("tqdm", "tqdm"),
    ("transformers", "transformers"),
    ("translators", "translators"),
    ("transparent_background", "transparent_background"),
    ("ultralytics", "ultralytics"),
    ("yaml", "yaml"),
]

_import_status_lines = []
for actual_module_name, desired_alias in _packages_to_check:
    
    try:
        module = importlib.import_module(actual_module_name)
        # Assign the imported module to the desired alias in the global namespace
        globals()[desired_alias] = module
        _import_status_lines.append("import " + actual_module_name + ("" if actual_module_name == desired_alias else " as " + desired_alias) + "\n")
    except:
        pass
EXTRA_MODULE_IMPORT_STRING = "".join(_import_status_lines)


def get_top_level_imports(script_lines):
    """
    Detects top-level imports, and parses them into a dictionary.

    The dictionary's keys are the import lines as they appear in the script.
    The values are dictionaries where keys are the imported names (or aliases)
    and values are the actual imported modules or objects.
    """
    top_level_imports = {}

    i = 0
    while i < len(script_lines):
        line = script_lines[i].strip()

        # Skip empty lines and comments
        if not line or line.startswith('#'):
            i += 1
            continue

        # Handle 'import module' or 'import module as alias' or 'import module1, module2'
        # This regex is modified to capture the *entire import list* in group 1,
        # then we'll parse that list. It now correctly handles trailing comments.
        match_simple_import = re.match(r'^\s*import\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\s+as\s+[a-zA-Z_][a-zA-Z0-9_]*)?(?:\s*,\s*[a-zA-Z_][a-zA-Z0-9_]*(?:\s+as\s+[a-zA-Z_][a-zA-Z0-9_]*)?)*)\s*(?:#.*)?$', line)
        if match_simple_import:
            # The full import line, including the "import" keyword
            full_import_line = match_simple_import.group(0).strip() # Use group(0) for the whole match
            
            # Extract just the modules part from the matched line (e.g., "os, math as m")
            modules_part = match_simple_import.group(1) 
            modules = {}
            parts = modules_part.split(',')
            for part in parts:
                part = part.strip()
                if not part: # Handle empty parts from extra commas like "import os,, math"
                    continue
                
                module_name = part
                alias = None
                
                if ' as ' in part:
                    module_name, alias = part.split(' as ')
                    module_name = module_name.strip()
                    alias = alias.strip()
                
                # The actual name to attempt importing (e.g., 'os', 'math')
                actual_import_name = module_name

                try:
                    # Attempt to import the module
                    imported_module = importlib.import_module(actual_import_name)
                    # Store it under its alias or original name
                    modules[alias if alias else actual_import_name] = imported_module
                except ImportError:
                    print(f"Warning: Could not import module '{actual_import_name}'. It will not be available in SilverRunPythonCode.")
                    pass # Continue processing other imports
                except Exception as e:
                    print(f"Warning: An unexpected error occurred while importing '{actual_import_name}': {e}")
                    pass
            
            if modules: # Only add to top_level_imports if something was successfully imported
                top_level_imports[full_import_line] = modules
            i += 1
            continue

        # Handle 'from module import name, name2' or 'from module import (name, name2)'
        # This regex handles single-line and multi-line 'from import' statements.
        # It's now more robust for trailing comments.
        match_from_import_start = re.match(r'^\s*(from\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s+import\s+.*)\s*(?:#.*)?$', line)

        if match_from_import_start:
            module_name_base = match_from_import_start.group(2) # The module being imported from (e.g., 'PIL', 'os.path')
            
            current_import_line_parts = [match_from_import_start.group(1)] # Store the raw import part for full_import_line
            
            # Basic check for multi-line continuation (improved)
            # Check if the line itself ends with a comma, backslash, or is within parentheses
            is_multiline_continuation = line.strip().endswith(',') or \
                                        line.strip().endswith('\\') or \
                                        ('(' in line and ')' not in line) # Starts with ( but not ends with )
            
            j = i + 1
            while j < len(script_lines) and is_multiline_continuation:
                next_line = script_lines[j].strip()
                if not next_line or next_line.startswith('#'):
                    break # Stop if we hit an empty line or comment
                
                current_import_line_parts.append(next_line)
                
                # Update continuation flag based on the new current line
                is_multiline_continuation = next_line.endswith(',') or \
                                            next_line.endswith('\\') or \
                                            ('(' in "".join(current_import_line_parts) and ')' not in "".join(current_import_line_parts))
                
                # Break if current line ends the parentheses block
                if next_line.endswith(')') and '(' in "".join(current_import_line_parts):
                    is_multiline_continuation = False # We've found the closing parenthesis
                
                j += 1
            
            full_import_line_raw = "\n".join(current_import_line_parts)
            full_import_line_display = " ".join([p.strip() for p in current_import_line_parts]) # For display key
            
            # Attempt to parse the full import line using ast (more robust than regex for internal parts)
            # If ast fails, fallback to your regex parsing, or simplify.
            # For simplicity with the current regex approach, let's keep the regex parsing,
            # but ensure we strip comments from line parts.

            # Reconstruct the import clause by joining parts and stripping comments
            import_clause_full = " ".join([re.sub(r'#.*$', '', p).strip() for p in current_import_line_parts])
            import_clause = import_clause_full[import_clause_full.find("import") + len("import"):].strip()
            
            # Remove parentheses for parsing if present
            if import_clause.startswith('(') and import_clause.endswith(')'):
                import_clause = import_clause[1:-1].strip()

            imported_names = {}
            names = import_clause.split(',')
            for name_part in names:
                name_part = name_part.strip()
                if not name_part:
                    continue

                original_name = name_part
                alias = None
                if ' as ' in name_part:
                    original_name, alias = name_part.split(' as ')
                    original_name = original_name.strip()
                    alias = alias.strip()

                try:
                    module = importlib.import_module(module_name_base)
                    if hasattr(module, original_name):
                        if alias:
                            imported_names[alias] = getattr(module, original_name)
                        else:
                            imported_names[original_name] = getattr(module, original_name)
                    else:
                        print(f"Warning: Object '{original_name}' not found in module '{module_name_base}'.")
                except (ImportError, AttributeError) as e:
                    print(f"Warning: Could not import '{original_name}' from '{module_name_base}': {e}. It will not be available in SilverRunPythonCode.")
                    pass # Handle cases where module/attribute might not be available or not found
                except Exception as e:
                    print(f"Warning: An unexpected error occurred while importing '{original_name}' from '{module_name_base}': {e}")
                    pass
            
            if imported_names:
                top_level_imports[full_import_line_display] = imported_names
            
            i = j # Move the index past the multi-line import
            continue

        i += 1
    
    return top_level_imports



script_lines = []
with open(NODE_FILE, 'r', encoding='utf-8') as f:
    script_lines = f.readlines()

TOP_MODULE_IMPORTS = get_top_level_imports(script_lines)
TOP_MODULE_IMPORT_STRING = "\n".join([f"{key}" for key in TOP_MODULE_IMPORTS])



class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False
# Our any instance wants to be a wildcard string
any = AnyType("*")


# Tensor to PIL
@global_func
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# PIL to Tensor
@global_func
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# Numpy to PIL
@global_func
def numpy2pil(image):
    return Image.fromarray(np.clip(255. * image.squeeze(), 0, 255).astype(np.uint8))

@global_func
def loadPil(path):
    return Image.open(path)


@global_func
def dynamic_prompts(prompt: str, seed: int, line_suffix: str = "", single_line_output: bool = True, remove_whitespaces: bool = True, remove_empty_tags: bool = True, wildcard_dir: str = "") -> str:
    
    # Updated _fix_prompt signature and logic
    def _fix_prompt(
        prompt: str, 
        line_suffix: str, 
        single_line_output: bool,
        remove_whitespaces: bool,
        remove_empty_tags: bool,
    ) -> str:
        """
        Processes the prompt by:
        1. Removing comments.
        2. Applying line suffix and optionally trimming (based on remove_whitespaces).
        3. Combining lines (based on single_line_output).
        4. Applying default prompt cleaning (e.g., ",," -> ",").
        5. Optionally removing empty tags (based on remove_empty_tags).
    
        Args:
            prompt (str): The initial string.
            line_suffix (str): String to append to each line.
            single_line_output (bool): If True, joins lines with a space; otherwise, joins with a newline.
            remove_whitespaces (bool): If True, strips lines and removes empty ones.
            remove_empty_tags (bool): If True, removes redundant separators like ' , ,' or ' , .'
    
        Returns:
            str: The modified string.
        """
        
        # --- Start of Modified Preprocessing Code ---
        cleaned_lines = []
        lines = prompt.splitlines()
    
        for line in lines:
            # Find the index of the first '#' character (comment delimiter)
            comment_start_index = line.find('#')
    
            if comment_start_index != -1:
                line_without_comment = line[:comment_start_index]
            else:
                line_without_comment = line
    
            # Apply trimming if remove_whitespaces is True
            trimmed_line = line_without_comment.strip() if remove_whitespaces else line_without_comment
            if remove_whitespaces:
                while ("  " in trimmed_line):
                    trimmed_line = trimmed_line.replace("  ", " ")
    
            # Apply the specified line_suffix
            if trimmed_line:
                # Only add suffix if the line is not empty after stripping
                final_line = trimmed_line + line_suffix
                
                # Only add non-empty lines to the cleaned list
                cleaned_lines.append(final_line)
    
        # Convert the cleaned lines back into a single/multi-line string
        # Join with " " for single line output, or "\n" for multi-line output
        joiner = " " if single_line_output else "\n"
        prompt = joiner.join(cleaned_lines)
        # --- End of Modified Preprocessing Code ---
        
        # Default cleaning replacements 
        replacements = {}
        replacements[" ,"] = ","
        replacements[",  "] = ", "
        replacements[" ."] = "."
        replacements[".  "] = ". "
        replacements[".,"] = "."
        replacements[",."] = ","
        replacements[",,"] = ","
        replacements[".."] = "."
        
        empty_tag_replacements = [".,", ",.", ",,", ".."]
        
        # Sort replacements by key length in descending order
        sorted_replacements = sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True)
    
        # The replacement loop runs until no changes are made.
        while True:
            replacement_made_in_pass = False
            current_prompt_state = prompt
    
            for old_substring, new_substring in sorted_replacements:
            
                if not remove_empty_tags and old_substring in empty_tag_replacements:
                    continue
            
                temp_prompt = current_prompt_state
    
                pattern = re.compile(re.escape(old_substring), re.IGNORECASE)
    
                replacements_to_make_in_this_pass = []
                for match in pattern.finditer(temp_prompt):
                    start, end = match.span()
    
                    # Check if this match is inside any <...> tag
                    tag_start_index = temp_prompt.rfind('<', 0, start)
                    if tag_start_index != -1:
                        tag_end_index = temp_prompt.find('>', tag_start_index)
                        if tag_end_index != -1 and tag_end_index > start:
                            continue
    
                    replacements_to_make_in_this_pass.append((start, end, new_substring))
    
    
                # Apply replacements from right to left
                for start, end, new_sub in sorted(replacements_to_make_in_this_pass, key=lambda x: x[0], reverse=True):
                    current_prompt_state = current_prompt_state[:start] + new_sub + current_prompt_state[end:]
                    replacement_made_in_pass = True
    
            if not replacement_made_in_pass:
                break
    
            prompt = current_prompt_state
        
        # --- Logic for remove_empty_tags ---
        if remove_empty_tags:
            temp_prompt = prompt
            
            # Simple cleanup of spacing before running the final delimiter removal
            temp_prompt = temp_prompt.replace(", ", ",").replace(" ,", ",").replace(" .", ".").replace(". ", ".")
            temp_prompt = temp_prompt.replace(",", ", ")
            temp_prompt = re.sub(r'\.(?!\d)', '. ', temp_prompt) # replaces '.' -> '. ' Only if there is no immediate digit after the dot
            
            # Use a loop to remove sequences of a delimiter, optional space, and another delimiter.
            while True:
                initial_len = len(temp_prompt)
                # Replace pattern (separator, optional space, separator) with a single separator
                # e.g., ', , ' -> ', '
                temp_prompt = re.sub(r'([.,])\s*([.,])', r'\1 ', temp_prompt)
                
                if len(temp_prompt) == initial_len:
                    break
            
            # Final cleaning of delimiters (e.g. 'cat,, dog' -> 'cat, dog')
            temp_prompt = temp_prompt.replace(",,", ",").replace("..", ".")
            prompt = temp_prompt
            
            
        prompt = prompt.strip()
        # The existing loop to remove leading/trailing delimiters/spaces
        while prompt.startswith(",") or prompt.startswith(".") or prompt.startswith(" ") or prompt.endswith(",") or prompt.endswith(" "):
            try:
                if prompt.startswith(",") or prompt.startswith(".") or prompt.startswith(" "):
                    prompt = prompt[1:].strip() # Strip again after removing
                if prompt.endswith(",") or prompt.endswith(" "):
                    prompt = prompt[:-1].strip() # Strip again after removing
            except:
                break
        
        return prompt
    
    
    def _process_wildcards(prompt: str, wildcard_dir: str, seed: int) -> str:
        """
        Replaces substrings like '__something__' in the prompt with the content of
        the corresponding '.txt' file.
    
        If the file contains multiple lines:
        1. Empty lines and comment lines (#...) are ignored.
        2. One line is randomly selected and returned.
        
        This ensures that only one item (which may contain further dynamic syntax) is
        substituted, regardless of whether combination syntax is present in the file.
        
        Args:
            prompt (str): The input string potentially containing wildcard substrings.
            wildcard_dir (str): The directory to search for wildcard '.txt' files.
            seed (int): An integer seed for the random number generator.
    
        Returns:
            str: The prompt string with wildcards replaced by a single selected line.
        """
        if wildcard_dir is None or not os.path.isdir(wildcard_dir):
            return prompt
        
        # Seed the random number generator for wildcard selection
        random.seed(seed)
    
        # Regex to find '__something__' or '__something.txt__'
        pattern = re.compile(r'__(.+?)__')
        
        def case_insensitive_resolve(base_dir: str, path_parts: list[str]) -> str | None:
            """
            Resolves a nested path inside base_dir in a case-insensitive way.
            Only lists contents one level at a time (no recursion).
            Returns absolute path to the file if found, else None.
            """
            current_dir = base_dir
    
            for part in path_parts[:-1]:
                try:
                    entries = os.listdir(current_dir)
                except OSError:
                    return None
    
                match = next((e for e in entries if e.lower() == part.lower() and 
                            os.path.isdir(os.path.join(current_dir, e))), None)
                if not match:
                    return None
                current_dir = os.path.join(current_dir, match)
    
            # Last part should be a file (case-insensitive match for .txt)
            target_file = path_parts[-1]
            try:
                entries = os.listdir(current_dir)
            except OSError:
                return None
    
            for e in entries:
                base_name, ext = os.path.splitext(e)
                if ext.lower() == '.txt' and base_name.lower() == target_file.lower():
                    return os.path.join(current_dir, e)
    
            return None
        
        def replace_match(match):
            wildcard_name = match.group(1).strip()
    
            if wildcard_name.lower().endswith('.txt'):
                wildcard_name = wildcard_name[:-4]
    
            # Normalize separators and split into parts
            normalized = re.sub(r'[\\/]+', '/', wildcard_name)
            parts = [p for p in normalized.split('/') if p]
    
            if not parts:
                return match.group(0)
    
            # Resolve path case-insensitively
            filepath = case_insensitive_resolve(wildcard_dir, parts)
            if not filepath or not os.path.isfile(filepath):
                return match.group(0)
    
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    file_content = f.read()
    
                # Filter lines (ignore empty and comment lines)
                lines = []
                for line in file_content.splitlines():
                    trimmed = line.strip()
                    if trimmed and not trimmed.startswith('#'):
                        comment_idx = trimmed.find('#')
                        if comment_idx != -1:
                            trimmed = trimmed[:comment_idx].strip()
                        if trimmed:
                            lines.append(trimmed)
    
                if not lines:
                    return ""
    
                # Choose one random valid line
                return random.choice(lines)
    
            except Exception as e:
                print(f"Error reading file {filepath}: {e}")
                return match.group(0)
    
        return pattern.sub(replace_match, prompt)
    
    
    def _process_combinations(prompt: str, seed: int) -> str:
        """
        Replaces substrings enclosed in '{...}' with a randomly selected choice
        from their pipe-separated contents.
        """
        # Seed the random number generator
        random.seed(seed)
    
        pattern = re.compile(r'{([^}{]*)}')
    
        while True:
            match = pattern.search(prompt)
            if not match:
                break
    
            start, end = match.span()
            choices_str = match.group(1)
    
            # --- Parse choices and weights ---
            raw_choices_list = [c for c in choices_str.split('|')]
            
            weighted_choices = []
            unweighted_choices = []
            total_defined_weight = 0.0
    
            for item in raw_choices_list:
                if '::' in item:
                    try:
                        weight_str, choice_text = item.split('::', 1)
                        weight = float(weight_str)
                        if not (0 <= weight <= 1):
                            raise ValueError("Weight must be between 0 and 1.")
                        
                        weighted_choices.append((choice_text, weight))
                        total_defined_weight += weight
                    except ValueError:
                        unweighted_choices.append(item)
                else:
                    unweighted_choices.append(item)
            
            if total_defined_weight > 1.0:
                for i in range(len(weighted_choices)):
                    choice, weight = weighted_choices[i]
                    weighted_choices[i] = (choice, weight / total_defined_weight)
                total_defined_weight = 1.0
                
            remaining_weight = 1.0 - total_defined_weight
            
            if unweighted_choices:
                if remaining_weight < 0:
                    remaining_weight = 0
                    
                equal_share_for_unweighted = remaining_weight / len(unweighted_choices)
                for choice_text in unweighted_choices:
                    weighted_choices.append((choice_text, equal_share_for_unweighted))
    
            # --- Perform selection ---
            selected_choice = ""
            if not weighted_choices:
                selected_choice = ""
            else:
                choices_list = [item[0] for item in weighted_choices]
                weights_list = [item[1] for item in weighted_choices]
    
                selected_choice = random.choices(choices_list, weights=weights_list, k=1)[0]
            
            # Replace the matched inner block with the selected choice
            prompt = prompt[:start] + selected_choice + prompt[end:]
    
        return prompt
    
    
    # --- Main function body: Fix applied here ---
    
    max_proccess_count = 30
    while max_proccess_count > 0:
        
        has_wildcards = "__" in prompt
        has_combinations = "{" in prompt or "}" in prompt
        
        if not has_wildcards and not has_combinations:
            break # Exit the loop if no more dynamic content is found
        
        # Process wildcards recursively (NO _fix_prompt call here)
        if has_wildcards:
            max_subproccess_count = 10
            while max_subproccess_count > 0:
                if "__" in prompt:
                    prompt = _process_wildcards(prompt, wildcard_dir, seed)
                else:
                    break
                max_subproccess_count -= 1
        
        # Process combinations recursively (NO _fix_prompt call here)
        if has_combinations:
            max_subproccess_count = 30
            while max_subproccess_count > 0:
                if "{" in prompt or "}" in prompt:
                    prompt = _process_combinations(prompt, seed)
                else:
                    break
                max_subproccess_count -= 1
        
        max_proccess_count -= 1
    
    # 1. FINAL CLEANING: Run _fix_prompt ONCE on the fully resolved string
    prompt = _fix_prompt(
        prompt=prompt, 
        line_suffix=line_suffix, 
        single_line_output=single_line_output, 
        remove_whitespaces=remove_whitespaces, 
        remove_empty_tags=remove_empty_tags
    )
    
    return prompt




class SilverAnyBridge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "any0": (any, ),
                "any1": (any, ),
                "any2": (any, ),
                "any3": (any, ),
                "any4": (any, ),
                "any5": (any, ),
                "any6": (any, ),
                "any7": (any, ),
                "sort_by_None": ("BOOLEAN", { "default": False, "tooltip": "When True: None inputs will be outputed last."}),
            },
        }

    RETURN_TYPES = (any, any, any, any, any, any, any, any, )
    RETURN_NAMES = ("any0", "any1", "any2", "any3", "any4", "any5", "any6", "any7", )
    FUNCTION = "main"
    CATEGORY = "silver"
    DESCRIPTION = """
Takes up to 8 inputs of any type and outputs them in the same order.
Useful to ensure that certain nodes only run after receiving the (unrequired) outputs from other nodes.
Ex: making a grid of images from a given folder path but only after inference took place (and saved the output into the same folder) in order to ensure the grid contains the generated image.
IMPORTANT: 
   When 'sort_by_None' is True -> None inputs will be added after non-None inputs.
"""
    
    def main(self, any0=None, any1=None, any2=None, any3=None, any4=None, any5=None, any6=None, any7=None, sort_by_None=False):
        
        if sort_by_None:
            l = [e for e in [any0, any1, any2, any3, any4, any5, any6, any7] if e is not None]
            while len(l) < 8:
                l.append(None)
            return (l[0], l[1], l[2], l[3], l[4], l[5], l[6], l[7], )
        
        return (any0, any1, any2, any3, any4, any5, any6, any7, )



class SilverRunPythonCode:
    """
    A ComfyUI node that takes a list and a Python code string as input,
    executes the code against a copy of the list, and outputs the modified list.
    This node allows for dynamic python code execution within ComfyUI workflows.
    """
    def __init__(s):
        pass

    @classmethod
    def INPUT_TYPES(s):
        GLOBAL_FUNCTION_DISPLAY_STRING = "\n\nThese are the natively supported functions:\n\n" + "".join(_GLOBAL_FUNCTION_DISPLAY_STRINGS) if len(_GLOBAL_FUNCTION_DISPLAY_STRINGS) > 0 else ""
        GLOBAL_OBJECT_DISPLAY_STRING = "\n\nThese are the natively supported objects (you can only access their initial value):\n\n" + "".join(_GLOBAL_OBJECT_DISPLAY_STRINGS) if len(_GLOBAL_OBJECT_DISPLAY_STRINGS) > 0 else ""
        CLASS_WARNING_STR = CLASS_WARNING_STRING if s.__name__ == "SilverRunPythonCode" and not DEBUG_MODE else ""
        return {
            "optional": {
                "list_input": ("LIST", {"default": None}),
                "shared_locals": ("DICT", {"default": None}), # Made optional to allow node to be used standalone
                "deepcopy": ("BOOLEAN", { "default": True }),
                "python_code": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": (
                            '"""' +
                        
                            CLASS_WARNING_STR +
                        
                            "\nTIP: copy all of this into a big note node and leave it next to this one for reference.\n" +
                            "These are the natively added imports by this node:\n\n" +
                            
                            TOP_MODULE_IMPORT_STRING + "\n" +
                            
                            EXTRA_MODULE_IMPORT_STRING +
                            
                            GLOBAL_FUNCTION_DISPLAY_STRING +
                            
                            GLOBAL_OBJECT_DISPLAY_STRING +
                            
                            
                            "\n\n---------- INSTRUCTIONS: ----------\n\n"
                            
                            "The 'list_input' variable is available here.\n"
                            "Modify 'list_input' in place, e.g.:\n\n"
                            "list_input[0] = 'new_first_item' # do NOT use return at top-level in your code!\n\n"
                            "You can import standard and venv Python packages:\n\n"
                            "import math\n"
                            "import numpy as np\n" 
                            "from PIL import ImageColor\n" +
                            
                            
                            SHARING_DATA_INFO_STRING +
                            
                            '"""\n# ---------- BEGIN YOUR CODE ----------\n\n\n'
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("LIST", "DICT",)
    RETURN_NAMES = ("list_input", "shared_locals",)
    
    OUTPUT_NODE = True

    FUNCTION = "execute"

    CATEGORY = "silver" # Categorize your node for better organization
    
    
    DESCRIPTION = CLASS_WARNING_STRING if __name__ == "SilverRunPythonCode" and not DEBUG_MODE else "Use [Silver] List Append to import inputs and [Silver] List Splitter or [Silver] List Select/Extract By Index to extract outputs from 'list_input'."

    def execute(self, python_code, deepcopy=True, list_input=None, shared_locals=None):
        """
        Executes the user-provided Python code and handles shared items.

        Args:
            list_input: The input list (or any object).
            python_code: The Python code string to execute.
            shared_locals: A dictionary of functions, classes, and variables
                                 received from another node.
        Returns:
            A tuple containing the modified list and the dictionary of shared items.
        """
        if list_input is None:
            list_input = []
        if shared_locals is None:
            shared_locals = {}

        if not isinstance(list_input, list):
            raise ValueError("'list_input' is not a list!")
        if not isinstance(shared_locals, dict):
            raise ValueError("'shared_locals' is not a dict!")
            
        if self.__class__.__name__ == "SilverRunPythonCode" and not DEBUG_MODE:  # spam the user until they fix the security flaw of this node and prevent code execution
            print(CLASS_WARNING_STRING + "\n[SilverRunPythonCode] 'python_code' execution will be skipped! Returning original inputs...")
            return (list_input, shared_locals)
            
        if deepcopy:
            # Create a deep copy of the input list.
            # This is crucial for two reasons:
            # 1. Prevents accidental modification of the original object connected from another node,
            #    which could lead to unexpected side effects in the workflow.
            # 2. Ensures that if the user's code fails, the original list remains untouched,
            #    allowing the node to return a stable (unmodified) output.
            try:
                # Attempt to deep copy. If list_input is not copyable (e.g., a simple int or a dict with unhashable keys),
                # copy.deepcopy will return it directly or raise an error.
                # In general, this is a robust way to handle various types.
                current_list_data = copy.deepcopy(list_input)
            except TypeError as e:
                print(f"[SilverRunPythonCode] Warning: Could not deep copy input type {type(list_input)}. Proceeding with direct reference. Error: {e}")
                current_list_data = list_input
            except Exception as e:
                print(f"[SilverRunPythonCode] Error during deep copy: {e}. Returning original input.")
                return (list_input, shared_locals)
        
        
        # 'exec_globals' ensures that commonly used modules like torch, numpy, and PIL
        # are available in the global scope of the executed code.
        # This helps resolve NameErrors when functions in the user's code try to
        # access objects from these libraries (e.g., Image from PIL).
        exec_globals = {}
        
        global _EXEC_GLOBALS
        if len(_EXEC_GLOBALS) == 0:
            for import_line, imported_data in TOP_MODULE_IMPORTS.items():
                for imported_module_name, imported_module in imported_data.items():
                    _EXEC_GLOBALS[imported_module_name] = imported_module
            
            for actual_module_name, desired_alias in _packages_to_check:
                if globals().get(desired_alias):
                    _EXEC_GLOBALS[desired_alias] = globals().get(desired_alias)
            
            for function_name in _GLOBAL_FUNCTIONS_REGISTRY:
                _EXEC_GLOBALS[function_name] = _GLOBAL_FUNCTIONS_REGISTRY[function_name]
            
            for object_name in _GLOBAL_OBJECTS_REGISTRY:
                _EXEC_GLOBALS[object_name] = _GLOBAL_OBJECTS_REGISTRY[object_name]
        
        
        # attempt imports from imports in the node
        if "import " in python_code:
            try:
                node_imports = get_top_level_imports(python_code.splitlines())
                if len(node_imports) > 0:
                    for import_line, imported_data in node_imports.items():
                        for imported_module_name, imported_module in imported_data.items():
                            if imported_module_name not in _EXEC_GLOBALS:
                                _EXEC_GLOBALS[imported_module_name] = imported_module
                                print(f"[SilverRunPythonCode] Successfully imported: {imported_module_name}")
            except Exception as e:
                print(f"[SilverRunPythonCode] Execution Error: {e}\n{traceback.format_exc()}")
        
        global SHARED_GLOBALS
        SHARED_GLOBALS.update(_EXEC_GLOBALS)
        exec_globals.update(SHARED_GLOBALS)
        # Add shared items from the input (shared_locals) directly to exec_globals. This makes them callable/accessible at the global scope within the user's code.
        exec_globals.update(shared_locals)
        
        # Define the execution environment for the user's code.
        # 'exec_locals' is where variables accessible to the user's script are defined.
        # The user's code will operate on 'list_input' within this scope.
        exec_locals = {
            "list_input": current_list_data if deepcopy else list_input,
            "shared_locals": shared_locals,
            "shared_local": shared_locals, # Handle user typo
            "shared_globals": {},
            "shared_global": {}, # Handle user typo
            "__builtins__": __builtins__ # Provide access to standard built-in functions like print()
        }
        
        try:
            # Execute the user-provided Python code.
            # The code is expected to modify 'list_input' within the `exec_locals` scope in place.
            #exec(python_code, exec_globals, exec_locals) # this no longer works after ComfyUI update!
            for key in exec_globals:
                if key != "list_input" and key != "shared_locals" and key != "shared_local" and key != "shared_globals" and key != "shared_global":
                    exec_locals[key] = exec_globals[key]
            exec(python_code, exec_locals)
            
            # Retrieve the (potentially) modified list from the execution scope.
            # If the user's code re-assigned 'list_input' (e.g., list_input = [1,2,3]),
            # exec_locals["list_input"] will reflect that new assignment.
            modified_list_output = exec_locals["list_input"]
            
            shared_locals.update(exec_locals.get("shared_locals", {}))
            shared_locals.update(exec_locals.get("shared_local", {}))
            
            SHARED_GLOBALS.update(exec_locals.get("shared_globals", {}))
            SHARED_GLOBALS.update(exec_locals.get("shared_global", {}))
            
            # Return the modified list as a tuple (ComfyUI expects a tuple for outputs).
            return (modified_list_output, shared_locals,)

        except Exception as e:
            # Catch any exceptions that occur during code execution.
            error_message = f"[SilverRunPythonCode] Execution Error: {e}\n{traceback.format_exc()}"
            print(error_message) # Print the error to the ComfyUI console/log
            # In case of an error, return the original, unmodified list_input.
            # This prevents the workflow from crashing and ensures a graceful failure.
            return (list_input, {})



class SilverListSelectExtractByIndex:
    """
    A simple ComfyUI node that takes an input list (or any indexable object)
    and an integer index, then outputs the item found at that index.
    Includes basic error handling for common issues like out-of-bounds indices
    or non-indexable inputs.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        """
        Defines the input types for the node.

        - list_any: Accepts any type of input, expected to be a list or tuple.
                      'forceInput': True ensures it must be connected from another node.
        - index: An integer representing the index to retrieve from the list.
                 A default value of 0 is provided.
        """
        return {
            "required": {
                "list_any": (any, {"forceInput": True}),
                "index": ("INT", {"default": 0, "min": 0}), # Min value of 0 makes sense for most list indexing
            },
        }

    RETURN_TYPES = (any,)
    RETURN_NAMES = ("item_at_index",)

    FUNCTION = "get_item"

    CATEGORY = "silver" # Placing it in the same category

    def get_item(self, list_any, index):
        """
        Retrieves an item from the input 'list_any' at the specified 'index'.

        Args:
            list_any: The input object, expected to be an indexable type (like a list or tuple).
            index: The integer index of the item to retrieve.

        Returns:
            A tuple containing the item at the specified index, or None if an error occurs.
        """
        try:
            # Attempt to access the item at the given index.
            # This handles both lists and other indexable types like tuples or strings.
            item = list_any[index]
            #print(f"ComfyUI ListIndexer: Successfully retrieved item at index {index}. Item type: {type(item)}")
            return (item,) # Return as a tuple as ComfyUI expects.
        except IndexError:
            # This error occurs if the index is out of bounds for the list/object.
            print(f"[SilverListSelectExtractByIndex] Error: Index {index} is out of range for the input. Input length might be too small or index is negative.")
            return (None,) # Return None or a meaningful default value
        except TypeError:
            # This error occurs if list_any is not an indexable type (e.g., an integer, a float, or a dictionary without proper key access).
            print(f"[SilverListSelectExtractByIndex] Error: Input 'list_any' is not an indexable type (e.g., not a list, tuple, or string). Type received: {type(list_any)}")
            return (None,) # Return None
        except Exception as e:
            # Catch any other unexpected errors.
            print(f"[SilverListSelectExtractByIndex] Error: An unexpected error occurred: {e}")
            return (None,) # Return None



class SilverListAppend:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "any0": (any, ),
                "any1": (any, ),
                "any2": (any, ),
                "any3": (any, ),
                "any4": (any, ),
                "any5": (any, ),
                "any6": (any, ),
                "any7": (any, ),
                "append_None_inputs": ("BOOLEAN", { "default": True }),
            },
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("list",)

    FUNCTION = "main"
    
    CATEGORY = "silver"
    DESCRIPTION = "Creates a list with elements of any type in the given order."

    def main(self, any0=None, any1=None, any2=None, any3=None, any4=None, any5=None, any6=None, any7=None, append_None_inputs=True):
        return ([e for e in [any0, any1, any2, any3, any4, any5, any6, any7] if append_None_inputs or (not append_None_inputs and e is not None)],)



class SilverBigListAppend:
    @classmethod
    def INPUT_TYPES(s):
        input_dict = {"optional": {}}
        for i in range(30):
            input_dict["optional"][f"any{i}"] = (any, )
        input_dict["optional"]["append_None_inputs"] = ("BOOLEAN", { "default": True })
        return input_dict
    
    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("list",)

    FUNCTION = "main"
    
    CATEGORY = "silver"
    DESCRIPTION = "Creates a list with elements of any type in the given order."
    
    def main(self, **kwargs):
        ordered_inputs = []
        for i in range(30):
            key = f"any{i}"
            if key in kwargs:
                ordered_inputs.append(kwargs[key])
        
        append_None_inputs = "append_None_inputs" in kwargs and kwargs["append_None_inputs"]
        if append_None_inputs:
            result_list = ordered_inputs
        else:
            result_list = [item for item in ordered_inputs if item is not None]
            
        return (result_list,)



class SilverListMerge:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "list1": ("LIST", ),
                "list2": ("LIST", ),
                "list3": ("LIST", ),
                "list4": ("LIST", ),
                "list5": ("LIST", ),
                "list6": ("LIST", ),
                "list7": ("LIST", ),
                "list8": ("LIST", ),
            },
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("list",)

    FUNCTION = "main"
    
    CATEGORY = "silver"
    DESCRIPTION = "Creates a new list with the merged contents of all given lists while preserving input and list element order. Ignores None type lists and elements."

    def main(self, list1=None, list2=None, list3=None, list4=None, list5=None, list6=None, list7=None, list8=None):
        
        ml = []
        for l in [list1, list2, list3, list4, list5, list6, list7, list8]:
            if l is not None and len(l) > 0:
                for e in l:
                    if e is not None:
                        ml.append(e)
        
        return (ml,)



class SilverListSplitter:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "list_input": ("LIST", ),
            }
        }

    RETURN_TYPES = tuple([any for _ in range(8)])
    RETURN_NAMES = tuple([f"list_input[{i}]" for i in range(8)])
    FUNCTION = "split_list"
    CATEGORY = "silver"

    def split_list(self, list_input=[]):
        outputs = [None] * 8
        
        for i in range(8):
            if i < len(list_input):
                outputs[i] = list_input[i]

        return tuple(outputs)



class SilverBigListSplitter:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "list_input": ("LIST", ),
            }
        }

    RETURN_TYPES = tuple([any for _ in range(30)])
    RETURN_NAMES = tuple([f"list_input[{i}]" for i in range(30)])
    FUNCTION = "split_list"
    CATEGORY = "silver"

    def split_list(self, list_input=[]):
        outputs = [None] * 30
        
        for i in range(30):
            if i < len(list_input):
                outputs[i] = list_input[i]

        return tuple(outputs)





NODE_CLASS_MAPPINGS = {
    "SILVER.SilverRunPythonCode": SilverRunPythonCode,
    "SILVER.SilverAnyBridge": SilverAnyBridge,
    "SILVER.SilverListSelectExtractByIndex": SilverListSelectExtractByIndex,
    "SILVER.SilverListAppend": SilverListAppend,
    "SILVER.SilverBigListAppend": SilverBigListAppend,
    "SILVER.SilverListMerge": SilverListMerge,
    "SILVER.SilverListSplitter": SilverListSplitter,
    "SILVER.SilverBigListSplitter": SilverBigListSplitter,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SILVER.SilverRunPythonCode": "[Silver] Run/Execute Python Code",
    "SILVER.SilverAnyBridge": "[Silver] Any Bridge",
    "SILVER.SilverListSelectExtractByIndex": "[Silver] List Select/Extract By Index",
    "SILVER.SilverListAppend": "[Silver] List Append",
    "SILVER.SilverBigListAppend": "[Silver] List Append BIG",
    "SILVER.SilverListMerge": "[Silver] List Merge",
    "SILVER.SilverListSplitter": "[Silver] List Splitter",
    "SILVER.SilverBigListSplitter": "[Silver] List Splitter BIG",
}





