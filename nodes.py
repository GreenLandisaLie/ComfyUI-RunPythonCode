import os
import subprocess
import inspect
import importlib
import re
import json





# --- Global Registries ---
_EXEC_GLOBALS = {}
_PUBLIC_FUNCTIONS_REGISTRY = {}
_PUBLIC_FUNCTION_DISPLAY_STRINGS = []
_PUBLIC_OBJECTS_REGISTRY = {}
_PUBLIC_OBJECT_DISPLAY_STRINGS = []
TOP_MODULE_IMPORTS = {}
TOP_MODULE_IMPORT_STRING = ""
EXTRA_MODULE_IMPORT_STRING = ""
SHARED_GLOBALS = {}



def public_func(func):
    """
    A decorator to mark functions as 'public_func' and register them.
    Functions decorated with @public_func will automatically be added to
    _PUBLIC_FUNCTIONS_REGISTRY.
    """
    if not callable(func):
        raise TypeError(f"Decorator 'public_func' can only be applied to functions, got {type(func)}")
    
    _PUBLIC_FUNCTIONS_REGISTRY[func.__name__] = func
    
    # Use inspect.signature to get the function's signature
    signature = inspect.signature(func)
    
    # Format it as a string: "function_name(param1: type, ...) -> return_type"
    # We include the function name here for clarity in the display string
    _PUBLIC_FUNCTION_DISPLAY_STRINGS.append(f"# {func.__name__}{signature}\n")
    
    return func

def public_obj(name, obj):
    """
    Registers a top-level object as 'public_obj'.
    This function should be called immediately after the object's definition.
    """
    if not isinstance(name, str):
        raise TypeError("The 'name' argument for public_object must be a string.")
    
    _PUBLIC_OBJECTS_REGISTRY[name] = obj
    
    # Get the type name (e.g., 'int', 'str', 'list')
    type_name = type(obj).__name__
    
    _PUBLIC_OBJECT_DISPLAY_STRINGS.append(f"# {name} ({type_name})\n")
    return obj # Return the object so you can still assign it normally


NODE_FILE = public_obj("NODE_FILE", os.path.abspath(__file__)) # leaving this just as an example on how to decorate variables from within the script



import os.path
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
import comfy.sd
import comfy.model_management
import comfy.utils
import comfy.comfy_types
from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict, FileLocator
import folder_paths


# List of packages to check. Each item is a tuple: (actual_module_name, desired_alias)
# If no alias is desired, you can use (actual_module_name, actual_module_name)
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
        _import_status_lines.append("# import " + actual_module_name + ("" if actual_module_name == desired_alias else " as " + desired_alias) + "\n")
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
TOP_MODULE_IMPORT_STRING = "\n".join([f"# {key}" for key in TOP_MODULE_IMPORTS])



class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False
# Our any instance wants to be a wildcard string
any = AnyType("*")


# Tensor to PIL
@public_func
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# PIL to Tensor
@public_func
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# Numpy to PIL
@public_func
def numpy2pil(image):
    return Image.fromarray(np.clip(255. * image.squeeze(), 0, 255).astype(np.uint8))

@public_func
def loadPil(path):
    return Image.open(path)





class SilverAnyBridge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "any1": (any, ),
                "any2": (any, ),
                "any3": (any, ),
                "any4": (any, ),
                "any5": (any, ),
                "any6": (any, ),
                "any7": (any, ),
                "any8": (any, ),
                "sort_by_None": ("BOOLEAN", { "default": False, "tooltip": "When True: None inputs will be outputed last."}),
            },
        }

    RETURN_TYPES = (any, any, any, any, any, any, any, any, )
    RETURN_NAMES = ("any1", "any2", "any3", "any4", "any5", "any6", "any7", "any8", )
    FUNCTION = "main"
    CATEGORY = "silver"
    DESCRIPTION = """
Takes up to 8 inputs of any type and outputs them in the same order.
Useful to ensure that certain nodes only run after receiving the (unrequired) outputs from other nodes.
Ex: making a grid of images from a given folder path but only after inference took place (and saved the output into the same folder) in order to ensure the grid contains the generated image.
IMPORTANT: 
   When 'sort_by_None' is True -> None inputs will be added after non-None inputs.
"""
    
    def main(self, any1=None, any2=None, any3=None, any4=None, any5=None, any6=None, any7=None, any8=None, sort_by_None=False):
        
        if sort_by_None:
            l = [e for e in [any1, any2, any3, any4, any5, any6, any7, any8] if e is not None]
            while len(l) < 8:
                l.append(None)
            return (l[0], l[1], l[2], l[3], l[4], l[5], l[6], l[7], )
        
        return (any1, any2, any3, any4, any5, any6, any7, any8, )



class SilverRunPythonCode:
    """
    A ComfyUI node that takes a list and a Python code string as input,
    executes the code against a copy of the list, and outputs the modified list.
    This node allows for dynamic python code execution within ComfyUI workflows.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        """
        Defines the input types for the node.

        - list_input: Expects any type of input, but is intended for lists.
        - python_code: A multiline string where the user can type Python code.
                       A default example is provided to guide the user.
        """
        PUBLIC_FUNCTION_DISPLAY_STRING = "\n\n# These are the natively supported functions:\n\n" + "".join(_PUBLIC_FUNCTION_DISPLAY_STRINGS) if len(_PUBLIC_FUNCTION_DISPLAY_STRINGS) > 0 else ""
        PUBLIC_OBJECT_DISPLAY_STRING = "\n\n# These are the natively supported objects (you can only access their initial value):\n\n" + "".join(_PUBLIC_OBJECT_DISPLAY_STRINGS) if len(_PUBLIC_OBJECT_DISPLAY_STRINGS) > 0 else ""
        return {
            "optional": {
                "list_input": (any, {"default": []}),
                "shared_locals": (any, {"default": {}}), # Made optional to allow node to be used standalone
                "python_code": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": (
                            "# TIP: copy all of this into a big note node and leave it next to this one for reference.\n" +
                            "# These are the natively added imports by this node:\n\n" +
                            
                            TOP_MODULE_IMPORT_STRING + "\n" +
                            
                            EXTRA_MODULE_IMPORT_STRING +
                            
                            PUBLIC_FUNCTION_DISPLAY_STRING +
                            
                            PUBLIC_OBJECT_DISPLAY_STRING +
                            
                            
                            "\n\n# ---------- INSTRUCTIONS: ----------\n\n" +
                            
                            "# The 'list_input' variable is available here.\n" +
                            "# Modify 'list_input' in place, e.g.:\n" +
                            "# list_input[0] = 'new_first_item'\n\n" +
                            "# You can import standard and venv Python modules:\n\n" +
                            "# import math\n" +
                            "# import numpy as np\n" + 
                            "# from PIL import ImageColor\n\n\n"
                            "# You can define local variables/functions/classes within a node and give linked nodes exclusive access to it via 'shared_locals'.\n" +
                            "# To do so you have to assign them in the node's code like so:\n\n" +
                            "# def awesome_function():\n#   return 'something'\n" +
                            "# shared_locals['awesome_function'] = awesome_function\n\n" +
                            "# Then you can access those through linked nodes using their names directly:\n\n" +
                            "# s = awesome_function()\n\n"
                            "# You can also define them globaly which will give other nodes access to them even if they are not linked or in a different workflow:\n"
                            "# shared_globals['awesome_function'] = awesome_function\n\n\n" +
                            "# Be mindeful of the node execution order in your workflow when using shared_globals.\n" +
                            "# And remember to re-assign variables/functions/classes if you make changes to them and you want those changes to be reflected in other nodes.\n\n"
                            
                            "# ---------- BEGIN YOUR CODE ----------\n\n"
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = (any, any,)
    RETURN_NAMES = ("list_input", "shared_locals",)

    FUNCTION = "execute"

    CATEGORY = "silver" # Categorize your node for better organization

    def execute(self, python_code, list_input=[], shared_locals={}):
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
            print(f"ComfyUI SilverRunPythonCode Warning: Could not deep copy input type {type(list_input)}. Proceeding with direct reference. Error: {e}")
            current_list_data = list_input
        except Exception as e:
            print(f"ComfyUI SilverRunPythonCode Error during deep copy: {e}. Returning original input.")
            return (list_input, {})


        # Define the execution environment for the user's code.
        # 'exec_locals' is where variables accessible to the user's script are defined.
        # The user's code will operate on 'list_input' within this scope.
        exec_locals = {
            "list_input": current_list_data,
            "shared_locals": {}, # User will populate this to export items
            "shared_local": {}, # Handle user typo
            "shared_globals": {}, # User will populate this to export items
            "shared_global": {}, # Handle user typo
            "__builtins__": __builtins__ # Provide access to standard built-in functions like print()
        }
        
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
            
            for function_name in _PUBLIC_FUNCTIONS_REGISTRY:
                _EXEC_GLOBALS[function_name] = _PUBLIC_FUNCTIONS_REGISTRY[function_name]
            
            for object_name in _PUBLIC_OBJECTS_REGISTRY:
                _EXEC_GLOBALS[object_name] = _PUBLIC_OBJECTS_REGISTRY[object_name]
        
        
        # attempt imports from imports in the node
        if "import " in python_code:
            try:
                node_imports = get_top_level_imports(python_code.splitlines())
                if len(node_imports) > 0:
                    for import_line, imported_data in node_imports.items():
                        for imported_module_name, imported_module in imported_data.items():
                            if imported_module_name not in _EXEC_GLOBALS:
                                _EXEC_GLOBALS[imported_module_name] = imported_module
                                print(f"Successfully imported {imported_module_name} via SilverRunPythonCode node.")
            except Exception as e:
                print(f"ComfyUI SilverRunPythonCode Execution Error: {e}\n{traceback.format_exc()}")
        
        
        exec_globals.update(_EXEC_GLOBALS)
        
        global SHARED_GLOBALS
        for key in SHARED_GLOBALS:
            exec_globals[key] = SHARED_GLOBALS[key]
        
        # Add shared items from the input (shared_locals) directly to exec_globals.
        # This makes them callable/accessible at the global scope within the user's code.
        if isinstance(shared_locals, dict):
            for key, value in shared_locals.items():
                exec_globals[key] = value
        
        
        try:
            # Execute the user-provided Python code.
            # The code is expected to modify 'list_input' within the `exec_locals` scope in place.
            exec(python_code, exec_globals, exec_locals)

            # Retrieve the (potentially) modified list from the execution scope.
            # If the user's code re-assigned 'list_input' (e.g., list_input = [1,2,3]),
            # exec_locals["list_input"] will reflect that new assignment.
            modified_list_output = exec_locals["list_input"]
            
            shared_locals_output = exec_locals.get("shared_locals", {})
            shared_locals_output.update(exec_locals.get("shared_local", {}))
            
            SHARED_GLOBALS.update(exec_locals.get("shared_globals", {}))
            SHARED_GLOBALS.update(exec_locals.get("shared_global", {}))
            
            # Return the modified list as a tuple (ComfyUI expects a tuple for outputs).
            return (modified_list_output, shared_locals_output,)

        except Exception as e:
            # Catch any exceptions that occur during code execution.
            error_message = f"ComfyUI SilverRunPythonCode Execution Error: {e}\n{traceback.format_exc()}"
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
            print(f"ComfyUI ListIndexer Error: Index {index} is out of range for the input. Input length might be too small or index is negative.")
            return (None,) # Return None or a meaningful default value
        except TypeError:
            # This error occurs if list_any is not an indexable type (e.g., an integer, a float, or a dictionary without proper key access).
            print(f"ComfyUI ListIndexer Error: Input 'list_any' is not an indexable type (e.g., not a list, tuple, or string). Type received: {type(list_any)}")
            return (None,) # Return None
        except Exception as e:
            # Catch any other unexpected errors.
            print(f"ComfyUI ListIndexer Error: An unexpected error occurred: {e}")
            return (None,) # Return None



class SilverListAppend:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "any1": (any, ),
                "any2": (any, ),
                "any3": (any, ),
                "any4": (any, ),
                "any5": (any, ),
                "any6": (any, ),
                "any7": (any, ),
                "any8": (any, ),
                "allow_none_type": ("BOOLEAN", { "default": True, "tooltip": "Leaving this True is useful to preserve the order even if some inputs are None."}),
            },
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("list",)

    FUNCTION = "main"
    
    CATEGORY = "silver"
    DESCRIPTION = "Creates a list with elements of any type in the given order."

    def main(self, any1=None, any2=None, any3=None, any4=None, any5=None, any6=None, any7=None, any8=None, allow_none_type=False):
        
        l = [e for e in [any1, any2, any3, any4, any5, any6, any7, any8] if (e is not None and not allow_none_type) or allow_none_type]
        
        return (l,)



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
    DESCRIPTION = "Creates a new list with the merged contents of all given lists while preserving input and list element order."

    def main(self, list1=None, list2=None, list3=None, list4=None, list5=None, list6=None, list7=None, list8=None):
        
        ml = []
        for l in [list1, list2, list3, list4, list5, list6, list7, list8]:
            if l is not None and len(l) > 0:
                for e in l:
                    ml.append(e)
        
        return (ml,)



NODE_CLASS_MAPPINGS = {
    "SILVER.SilverAnyBridge": SilverAnyBridge, 
    "SILVER.SilverRunPythonCode": SilverRunPythonCode, 
    "SILVER.SilverListSelectExtractByIndex": SilverListSelectExtractByIndex, 
    "SILVER.SilverListAppend": SilverListAppend, 
    "SILVER.SilverListMerge": SilverListMerge, 
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SILVER.SilverAnyBridge": "[Silver] Any Bridge", 
    "SILVER.SilverRunPythonCode": "[Silver] Run/Execute Python Code", 
    "SILVER.SilverListSelectExtractByIndex": "[Silver] List Select/Extract By Index", 
    "SILVER.SilverListAppend": "[Silver] List Append", 
    "SILVER.SilverListMerge": "[Silver] List Merge", 
}
