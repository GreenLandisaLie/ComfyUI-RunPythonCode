import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "Comfy.SilverRunPythonCode",
    beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name.indexOf('SilverRunPythonCode') === -1) return;

        // Advanced syntax highlighting
		const highlight = (text) => {
			let work = text;
		
			const tokens = [];
			const protect = (frag) => {
				const tok = `@@@TOKEN${tokens.length}@@@`;
				tokens.push(frag);
				return tok;
			};
		
			const escapeHTML = (s) => s
				.replace(/&/g, "&amp;")
				.replace(/</g, "&lt;")
				.replace(/>/g, "&gt;")
				.replace(/"/g, "&quot;")
				.replace(/'/g, "&#39;");
		
			// ==================================================
			// 1️⃣ PROTECT MULTILINE STRINGS ("""...""" or '''...''')
			// ==================================================
			work = work.replace(/("""|''')[\s\S]*?(?:\1|$)/g, (match) => {
				const safe = escapeHTML(match);
				return protect(`<span style="color:#90EE90;">${safe}</span>`);
			});
		
			// ==================================================
			// 2️⃣ PROCESS LINE BY LINE
			// ==================================================
			const lines = work.split(/\r?\n/).map((line) => {
				// skip protected triple-quoted string lines
				if (line.includes("@@@TOKEN")) return line;
		
				let safeLine = line;
		
				// Decorators (line starts with @)
				if (safeLine.trim().startsWith("@")) {
					const safe = escapeHTML(safeLine);
					return protect(`<span style="color:#0000FF; font-style:italic;">${safe}</span>`);
				}
		
				// Comments: detect first '#' not in a string
				let commentIndex = -1;
				let inStr = false;
				let quote = null;
		
				for (let i = 0; i < safeLine.length; i++) {
					const ch = safeLine[i];
					if (!inStr && (ch === '"' || ch === "'")) {
						inStr = true;
						quote = ch;
					} else if (inStr && ch === quote && safeLine[i - 1] !== "\\") {
						inStr = false;
						quote = null;
					} else if (!inStr && ch === "#") {
						commentIndex = i;
						break;
					}
				}
		
				let codePart = safeLine;
				let commentPart = "";
		
				if (commentIndex !== -1) {
					codePart = safeLine.slice(0, commentIndex);
					commentPart = safeLine.slice(commentIndex);
				}
		
				// Highlight strings inside codePart only (not in comment)
				codePart = codePart.replace(/(['"])(?:\\.|(?!\1)[^\\\r\n])*\1/g, (match) => {
					const safe = escapeHTML(match);
					return protect(`<span style="color:#FFB6C1;">${safe}</span>`);
				});
		
				// Highlight keywords, special vars, numbers
				codePart = codePart
					.replace(
						/\b(if|else|elif|def|class|while|for|in|or|is|not|import|del|return|->)\b/g,
						(m) => protect(`<span style="color:#FFFF00;">${m}</span>`)
					)
					.replace(
						/\b(list_input|shared_local|shared_locals|shared_global|shared_globals)\b/g,
						(m) => protect(`<span style="color:#F4A460;">${m}</span>`)
					)
					.replace(/\b\d+(\.\d+)?\b/g, (m) =>
						protect(`<span style="color:#00FFFF;">${m}</span>`)
					);
		
				// Now handle commentPart (if exists)
				if (commentPart) {
					let color = "#6A9955"; // normal
					if (commentPart.startsWith("###")) color = "#FFA500";
					else if (commentPart.startsWith("##")) color = "#A020F0";
		
					commentPart = protect(
						`<span style="color:${color}; font-style:italic;">${escapeHTML(commentPart)}</span>`
					);
				}
		
				return escapeHTML(codePart) + commentPart;
			});
		
			work = lines.join("\n");
		
			// ==================================================
			// 3️⃣ Restore tokens
			// ==================================================
			for (let i = 0; i < tokens.length; i++) {
				work = work.split(`@@@TOKEN${i}@@@`).join(tokens[i]);
			}
		
			return work;
		};

		



        const getPlainCursorPosition = (editor, selection) => {
            const range = selection.getRangeAt(0);
            const preRange = range.cloneRange();
            preRange.selectNodeContents(editor);
            preRange.setEnd(range.startContainer, range.startOffset);
            return preRange.cloneContents().textContent.length;
        };

        const setPlainCursorPosition = (editor, offset) => {
            let currentOffset = 0;
            const walker = document.createTreeWalker(
                editor,
                NodeFilter.SHOW_TEXT,
                null,
                false
            );
            let node;

            while (currentOffset <= offset && (node = walker.nextNode())) {
                const nodeLength = node.textContent.length;
                
                if (currentOffset + nodeLength >= offset) {
                    const range = document.createRange();
                    range.setStart(node, offset - currentOffset);
                    range.collapse(true);

                    const sel = window.getSelection();
                    sel.removeAllRanges();
                    sel.addRange(range);
                    return;
                }
                currentOffset += nodeLength;
            }
            
            if (offset >= currentOffset) {
                const range = document.createRange();
                range.selectNodeContents(editor);
                range.collapse(false);

                const sel = window.getSelection();
                sel.removeAllRanges();
                sel.addRange(range);
            }
        };
		

        // --- [End of helper functions] ---


        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            origOnNodeCreated?.apply(this, arguments);
            console.log("[SilverRunPythonCode] JS initialized for:", this.title);

            const w = this.widgets?.find(w => w.name === "python_code");
			w.computeSize = () => [0, 0]; // Force the widget to take 0 height and 0 width
			w.y = -600; // Keep this just in case, to push it off-screen visually
            w.hidden = true; 

            const editor = document.createElement("div");
            editor.contentEditable = "true";
			editor.spellcheck = false;
			
            // ... (CSS styles for editor)
			editor.style.cssText = `
                border: 1px solid var(--border-color);
                border-radius: 6px;
                padding: 6px;
                min-height: 50px;
                white-space: pre-wrap;
                overflow-y: auto;
                font-family: monospace;
                color: #ffffff;
                background: #222222;
                outline: none;
            `;
			
            // Function to synchronize the custom editor from the ComfyUI widget value
            const updateEditorContent = () => {
                const text = w.value || "";
                editor.innerHTML = highlight(text);
                // Ensure the canvas updates its size if content changes on load
                this.setDirtyCanvas(true, true); 
            };
            
            // --- FIX FOR REFRESH: INITIAL VALUE LOADING ---
            // 1. Redefine onCreated to use the actual loaded value
            w.onCreated = () => {
                // This ensures the custom editor is populated with the saved value
                // AFTER ComfyUI has loaded it from the backend.
                updateEditorContent(); 
            };
			
            // 2. Add an event listener to the ComfyUI widget to force a visual update
            // if the value is ever changed externally (e.g., via a Load function)
            w.callback = updateEditorContent;
			
			// Explicitly call the update function at the end of onNodeCreated.
            // This forces the initial visual update using the value already confirmed to be
            // in w.value for a newly created node.
            updateEditorContent();
			
            // Stop ComfyUI shortcuts
            editor.addEventListener("keydown", (e) => {
                e.stopPropagation();
				
				if (e.key === 'Tab') { // add text editor behavior with the TAB key but use 4 spaces instead of '\t'
					e.preventDefault(); // CRITICAL: Stop the browser from blurring the element/changing focus
			
					const sel = window.getSelection();
					if (!sel || sel.rangeCount === 0) return;
			
					const plainOffset = getPlainCursorPosition(editor, sel);
					let plainText = editor.innerText;
					
					const indentation = '    '; // Using 4 spaces
					
					plainText = plainText.substring(0, plainOffset) + indentation + plainText.substring(plainOffset);
			
					w.value = plainText; // Update ComfyUI widget
					updateEditorContent(); // Re-highlight (this calls editor.innerHTML = highlight(text);)
					
					// Set cursor to the position after the inserted characters
					setPlainCursorPosition(editor, plainOffset + indentation.length); 
				}
            });
			
            // Intercept 'Enter' to control newlines and cursor movement
            editor.addEventListener("keypress", (e) => {
                if (e.key === 'Enter') {
                    e.preventDefault(); 
                    const sel = window.getSelection();
                    if (!sel || sel.rangeCount === 0) return;

                    const plainOffset = getPlainCursorPosition(editor, sel);
                    let plainText = editor.innerText;
                    plainText = plainText.substring(0, plainOffset) + "\n" + plainText.substring(plainOffset);

                    w.value = plainText; // Update ComfyUI widget
                    updateEditorContent(); // Re-highlight
                    
                    setPlainCursorPosition(editor, plainOffset + 1);
                }
            });

            // Refactored input handler for cursor stability
            editor.addEventListener("input", () => {
                const sel = window.getSelection();
                if (!sel || sel.rangeCount === 0) return;
                
                const plainOffset = getPlainCursorPosition(editor, sel);
                const plainText = editor.innerText;
                w.value = plainText; // Update ComfyUI widget
                
                updateEditorContent(); // Re-highlight
			
                setPlainCursorPosition(editor, plainOffset);
            });
			
			// --- Ensure the element is truly deselected on leaving focus ---
			editor.addEventListener('blur', () => {
				const sel = window.getSelection();
				// Crucial: remove any active selection ranges from the contentEditable element
				if (sel.rangeCount > 0) {
					sel.removeAllRanges();
				}
				// Explicitly call blur
				editor.blur();
			});
			
			// --- Allow ComfyUI default zoom behavior with mouse wheel ---
			editor.addEventListener("wheel", (e) => {
				e.stopPropagation();
				e.preventDefault();
				// Re-dispatch to ComfyUI canvas manually
				const canvas = document.querySelector("#graph-canvas");
				if (canvas) {
					const newEvent = new WheelEvent(e.type, e);
					canvas.dispatchEvent(newEvent);
				}
			}, { passive: false });
			
            
            // --- Use ComfyUI's DOM widget system ---
            const widget = this.addDOMWidget(`richprompt_widget_${this.id}`, "dom", editor, {
                //computeSize: (w, h) => [w, Math.max(50, Math.max(50, editor.scrollHeight + 10))]
				//computeSize: (w, h) => [w, h]
            });			

			// FIX issue caused by: https://github.com/Comfy-Org/ComfyUI_frontend/pull/6087/files
			const stopPropagation = (e) => {
				// Prevent the event from bubbling up to the ComfyUI canvas listeners
				e.stopPropagation();
				
				// Optional: Stop the default action, though the browser should handle it
				// for contentEditable elements correctly if propagation is stopped.
				// e.preventDefault(); 
			};
			editor.addEventListener("copy", stopPropagation);
			editor.addEventListener("paste", stopPropagation);
			editor.addEventListener("cut", stopPropagation);
            
            this.setDirtyCanvas(true, true);
            
            // cleanup
            this.onRemoved = function() {
                editor.remove();
            };
        };
    },

});
