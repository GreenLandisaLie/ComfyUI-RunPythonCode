# ComfyUI-RunPythonCode

Run arbitrary Python code to save you the hassle of connecting dozens of nodes for simple stuff.



## Installation

1. Clone this repository into your ComfyUI's custom_nodes directory:
```
cd ComfyUI/custom_nodes
git clone https://github.com/GreenLandisaLie/ComfyUI-RunPythonCode.git
```

2. Create a ~10 length random string with (a-Z,0-9) characters (ex: f8Rca4Cx)

3. (With a text editor of your choice) MANUALLY add the random string as suffix to the class name in the following lines in ComfyUI/custom_nodes/ComfyUI-RunPythonCode/nodes.py
```
class SilverRunPythonCode:   (line 385)
"SILVER.SilverRunPythonCode": SilverRunPythonCode,   (line 800)
"SILVER.SilverRunPythonCode": "[Silver] Run/Execute Python Code",   (line 810)
```
   They should look something like this:
```
class SilverRunPythonCodef8Rca4Cx:
"SILVER.SilverRunPythonCodef8Rca4Cx": SilverRunPythonCodef8Rca4Cx,
"SILVER.SilverRunPythonCodef8Rca4Cx": "[Silver] Run/Execute Python Code",
``` 

4. Save the file. 


## About safety

Renaming the 'SilverRunPythonCode' class will prevent bad actors from successfully trying to run malicious code on your machine by sharing workflows infected with this node and malicious code in it.

The execution of the 'input code' will never happen until the user follows these instructions. When this happens, the original inputs will be returned completely unchanged.

With this approach the node can be considered safe. 

But there is still 1 possible exploit which can happen when an user is interchanging outputs with metadata/workflows with someone else and running the other party's workflows without inspecting the metadata/.json file.

In that scenario, if the other party decides to copy your unique class name and add a hidden node with malicious code in it - its possible that you would not notice it and run it.

***To prevent this, all you have to do is NEVER share any output/workflow with someone if it uses this node. Simple.***


## Features:

- Safety: Does not execute 'input code' until the user renames the node's class.
- Comes bundled with 5 major helper nodes: List Append, List Merge, List Splitter, List Extract from Index and Any Bridge.
- Supports any number of inputs of any type.
- Supports package imports from standard python modules as well as any package installed in your venv.
- Allows sharing variables/functions/classes written within a node with other linked nodes.
- Allows sharing variables/functions/classes globally with non-linked nodes from the same or even other loaded workflows.


## Usage:

- Simply connect your inputs to a provided List Append node and send its output to 'list_input'.
- To retrieve the outputs, connect the output ('list_input') to a List Splitter/Extract by Index node.
- Much more detailed instructions are included within the main node when you place it into a workflow.


![1](https://github.com/user-attachments/assets/3b9b819c-fdd4-44bb-b800-06ac20725bb2)



Fun fact: Most of the code was written by Gemini.
