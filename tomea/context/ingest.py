import os
import ast
import logging

# Try importing gitingest, handle if not installed yet
try:
    from gitingest import ingest
except ImportError:
    ingest = None

logger = logging.getLogger(__name__)

class ContextIngestor:
    """
    The Librarian.
    Reads code from URLs or local files and strips it down to a 'Skeleton'.
    """
    
    @staticmethod
    def extract_skeleton(code_text: str) -> str:
        """
        Parses Python code and returns ONLY class/function definitions and docstrings.
        Removes all implementation details (function bodies).
        """
        try:
            tree = ast.parse(code_text)
        except SyntaxError:
            return "# Error: Could not parse code syntax for skeleton extraction."

        skeleton = []
        
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                if isinstance(node, ast.ClassDef):
                    skeleton.append(f"class {node.name}:")
                    # Capture methods inside class
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            skeleton.append(f"    def {item.name}(self, ...):")
                            if ast.get_docstring(item):
                                doc = ast.get_docstring(item).split('\n')[0]
                                skeleton.append(f"        \"\"\"{doc}...\"\"\"")
                            skeleton.append("        pass\n")
                            
                elif isinstance(node, ast.FunctionDef):
                    skeleton.append(f"def {node.name}(...):")
                    skeleton.append("    pass\n")

        return "\n".join(skeleton)

    @staticmethod
    def read_repo(url: str) -> str:
        """
        Uses gitingest to turn a GitHub URL into a single text summary.
        """
        if not ingest:
            return "Error: 'gitingest' library not installed. Run `pip install gitingest`."
        
        logger.info(f"--- 📥 Ingesting Context from {url}... ---")
        try:
            # Gitingest returns (summary, tree, content)
            summary, tree, content = ingest(url)
            # Combine relevant parts
            return f"{summary}\n\nFile Tree:\n{tree}\n\nSelected Content:\n{content[:15000]}..." # Limit token usage
        except Exception as e:
            return f"Error ingesting repo: {e}"
        
    @staticmethod
    async def read_repo_async(url: str) -> str:
        """
        Async version of read_repo for use in async contexts.
        Uses gitingest to turn a GitHub URL into a single text summary.
        """
        import asyncio
        
        if not ingest:
            return "Error: 'gitingest' library not installed. Run `pip install gitingest`."
        
        logger.info(f"--- 📥 Ingesting Context from {url}... ---")
        try:
            # Gitingest's ingest() is synchronous but does I/O
            # Run it in executor to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            summary, tree, content = await loop.run_in_executor(None, ingest, url)
            
            # Combine relevant parts
            result = f"{summary}\n\nFile Tree:\n{tree}\n\nSelected Content:\n{content[:15000]}..."
            logger.info(f"   ✅ Ingested {len(result)} chars from repo")
            return result
        except Exception as e:
            logger.warning(f"   ⚠️  Repo ingestion failed: {e}")
            return f"Error ingesting repo: {e}"