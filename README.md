# ComfyUI-RunPythonCode
Run arbitrary Python code to save you the hassle of connecting dozens of nodes for simple stuff.

Do you know basic Python and are you tired of trying to figure out which custom nodes do exactly what you want - from a myriad of nodes that don't even come with description - and often find out you still need half a dozen of different nodes and all of this just to do something you could write in python in just 2min?
If so then this node is for you.

**ATTENTION: (Speaking the obvious)**

Allowing a node run arbitrary Python code poses a light security risk whenever you download a workflow from the internet and run it without being sure it does not contain this node.

***To solve this security flaw you need to manually edit the nodes classnames in 'nodes.py' and change (at the bottom of the script) 'NODE_CLASS_MAPPINGS' and 'NODE_DISPLAY_NAME_MAPPINGS' to reflect your edits.***

ComfyUI attemps to load workflows using node's classnames - so if you make them unique and never share them with anyone (via workflows or outputs) then no one should be able to take advantage of this security flaw.

Features:
- Safety: Executes arbritary code within a try-except block.
- Supports package imports from standard python modules as well as any package installed in your venv.
- Allows sharing variables/functions/classes written within a node with other linked nodes.
- Allows sharing variables/functions/classes globally with non-linked nodes from the same or even other workflows.
- Supports any number of inputs of any type. But they must be appended into a list.
- Comes bundled with 4 helper nodes: List Append, List Merge, List Extract by Index and Any Bridge.

![1](https://github.com/user-attachments/assets/3b9b819c-fdd4-44bb-b800-06ac20725bb2)


Fun fact: Most of the code was written by Gemini.
