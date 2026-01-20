"""
Dependency extraction for Python ML repositories.

Extracts custom components (attention, embeddings, etc.) along with
all their dependencies, inlined into a single self-contained file.

This module handles complex ML repos where custom components depend on
other custom classes, extracting complete dependency chains and producing
self-contained Python code.

Usage:
    from dependency_extractor import extract_component
    
    code, metadata = extract_component(
        repo_path="/tmp/gated_attention_repo",
        component_name="Qwen3Attention",
        component_type="attention"
    )
"""

import ast
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import networkx as nx
except ImportError:
    raise ImportError("networkx is required: pip install networkx")

try:
    from tree_sitter_languages import get_language, get_parser
except ImportError:
    raise ImportError("tree-sitter-languages is required: pip install tree-sitter-languages")

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# Custom Exceptions
# =============================================================================


class ComponentNotFoundError(Exception):
    """Raised when target component not found in repository."""

    def __init__(self, component_name: str, suggestions: Optional[List[str]] = None):
        self.component_name = component_name
        self.suggestions = suggestions or []
        msg = f"Component '{component_name}' not found in repository."
        if self.suggestions:
            msg += f" Did you mean: {', '.join(self.suggestions[:5])}?"
        super().__init__(msg)


class ExtractionError(Exception):
    """Raised when extraction fails for general reasons."""

    pass


class ComplexityError(ExtractionError):
    """Raised when dependencies are too complex (>1500 lines total)."""

    def __init__(self, total_lines: int, max_lines: int = 1500):
        self.total_lines = total_lines
        self.max_lines = max_lines
        super().__init__(
            f"Dependency chain too complex: {total_lines} lines exceeds {max_lines} limit. "
            "Consider extracting a smaller component or manually simplifying dependencies."
        )


class CircularDependencyError(ExtractionError):
    """Raised when circular dependencies are detected."""

    def __init__(self, cycle: List[str]):
        self.cycle = cycle
        super().__init__(f"Circular dependency detected: {' -> '.join(cycle)}")


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ClassInfo:
    """Information about a parsed class."""

    name: str
    file_path: Path
    start_line: int
    end_line: int
    base_classes: List[str] = field(default_factory=list)
    source_code: str = ""
    decorators: List[str] = field(default_factory=list)
    references: Set[str] = field(default_factory=set)
    imports: Set[str] = field(default_factory=set)
    docstring: Optional[str] = None
    is_stubbed: bool = False


@dataclass
class FunctionInfo:
    """Information about a parsed function."""

    name: str
    file_path: Path
    start_line: int
    end_line: int
    source_code: str = ""
    decorators: List[str] = field(default_factory=list)
    references: Set[str] = field(default_factory=set)


@dataclass
class ImportInfo:
    """Information about an import statement."""

    module: str
    names: List[str]  # e.g., ['ClassA', 'ClassB'] for "from X import ClassA, ClassB"
    alias: Optional[str] = None
    is_relative: bool = False
    level: int = 0  # For relative imports: . = 1, .. = 2, etc.


# =============================================================================
# Constants
# =============================================================================

# Standard library modules (partial list of commonly used)
STDLIB_MODULES = {
    "abc",
    "argparse",
    "ast",
    "asyncio",
    "base64",
    "collections",
    "contextlib",
    "copy",
    "dataclasses",
    "datetime",
    "enum",
    "functools",
    "gc",
    "hashlib",
    "importlib",
    "inspect",
    "io",
    "itertools",
    "json",
    "logging",
    "math",
    "multiprocessing",
    "operator",
    "os",
    "pathlib",
    "pickle",
    "random",
    "re",
    "shutil",
    "signal",
    "socket",
    "string",
    "subprocess",
    "sys",
    "tempfile",
    "threading",
    "time",
    "traceback",
    "types",
    "typing",
    "unittest",
    "urllib",
    "uuid",
    "warnings",
    "weakref",
}

# Third-party ML libraries
THIRD_PARTY_MODULES = {
    "torch",
    "numpy",
    "scipy",
    "pandas",
    "sklearn",
    "transformers",
    "datasets",
    "tokenizers",
    "accelerate",
    "safetensors",
    "einops",
    "flash_attn",
    "triton",
    "xformers",
    "bitsandbytes",
    "peft",
    "trl",
    "wandb",
    "tensorboard",
    "matplotlib",
    "seaborn",
    "tqdm",
    "yaml",
    "omegaconf",
    "hydra",
}

# Maximum limits
MAX_RECURSION_DEPTH = 3
MAX_TOTAL_LINES = 1500
MAX_SINGLE_CLASS_LINES = 500


# =============================================================================
# Repository Parser
# =============================================================================


class RepositoryParser:
    """Parses Python files in a repository using tree-sitter."""

    def __init__(self):
        """Initialize the parser with tree-sitter for Python."""
        self.parser = get_parser("python")
        self.language = get_language("python")
        self._cache: Dict[Path, bytes] = {}

    def parse_repository(self, repo_path: Path) -> Dict[str, ClassInfo]:
        """
        Parse all Python files in repository and extract class information.

        Args:
            repo_path: Path to the repository root

        Returns:
            Dictionary mapping class names to ClassInfo objects
        """
        classes: Dict[str, ClassInfo] = {}
        functions: Dict[str, FunctionInfo] = {}

        py_files = list(repo_path.rglob("*.py"))
        logger.info(f"Found {len(py_files)} Python files in {repo_path}")

        for py_file in py_files:
            # Skip test files and setup files
            if any(
                part in py_file.parts
                for part in ["test", "tests", "testing", "__pycache__"]
            ):
                continue

            try:
                file_classes, file_functions = self._parse_file(py_file)
                for name, info in file_classes.items():
                    # Handle duplicate class names - prefer non-test files
                    if name in classes:
                        logger.debug(f"Duplicate class {name} found, keeping first")
                    else:
                        classes[name] = info
                functions.update(file_functions)
            except Exception as e:
                logger.warning(f"Failed to parse {py_file}: {e}")

        logger.info(f"Parsed {len(classes)} classes and {len(functions)} functions")

        self._cache.clear() 

        return classes

    def _parse_file(
        self, file_path: Path
    ) -> Tuple[Dict[str, ClassInfo], Dict[str, FunctionInfo]]:
        """Parse a single Python file."""
        source = file_path.read_bytes()
        self._cache[file_path] = source
        tree = self.parser.parse(source)

        classes = {}
        functions = {}

        for node in self._walk_tree(tree.root_node):
            if node.type == "class_definition":
                class_info = self._extract_class_info(node, file_path, source)
                if class_info:
                    classes[class_info.name] = class_info

            elif node.type == "function_definition":
                # Only top-level functions
                if node.parent and node.parent.type == "module":
                    func_info = self._extract_function_info(node, file_path, source)
                    if func_info:
                        functions[func_info.name] = func_info

        return classes, functions

    def _walk_tree(self, node):
        """Walk tree-sitter AST nodes."""
        yield node
        for child in node.children:
            yield from self._walk_tree(child)

    def _extract_class_info(
        self, node, file_path: Path, source: bytes
    ) -> Optional[ClassInfo]:
        """Extract class information from a class_definition node."""
        # Get class name
        name_node = node.child_by_field_name("name")
        if not name_node:
            return None
        class_name = source[name_node.start_byte : name_node.end_byte].decode("utf-8")

        # Get base classes
        base_classes = []
        superclasses_node = node.child_by_field_name("superclasses")
        if superclasses_node:
            for child in superclasses_node.children:
                if child.type == "identifier":
                    base_classes.append(
                        source[child.start_byte : child.end_byte].decode("utf-8")
                    )
                elif child.type == "attribute":
                    base_classes.append(
                        source[child.start_byte : child.end_byte].decode("utf-8")
                    )

        # Get decorators
        decorators = []
        prev_sibling = node.prev_sibling
        while prev_sibling and prev_sibling.type == "decorator":
            dec_text = source[
                prev_sibling.start_byte : prev_sibling.end_byte
            ].decode("utf-8")
            decorators.insert(0, dec_text)
            prev_sibling = prev_sibling.prev_sibling

        # Get source code (including decorators)
        if decorators:
            # Find the start of first decorator
            dec_start = node.start_byte
            temp = node.prev_sibling
            while temp and temp.type == "decorator":
                dec_start = temp.start_byte
                temp = temp.prev_sibling
            source_code = source[dec_start : node.end_byte].decode("utf-8")
            start_line = (
                source[:dec_start].count(b"\n") + 1
            )
        else:
            source_code = source[node.start_byte : node.end_byte].decode("utf-8")
            start_line = source[: node.start_byte].count(b"\n") + 1

        end_line = source[: node.end_byte].count(b"\n") + 1

        # Find references (other classes/types used in this class)
        references = self._find_references(node, source)

        # Get docstring
        body_node = node.child_by_field_name("body")
        docstring = None
        if body_node and body_node.children:
            first_stmt = body_node.children[0]
            if first_stmt.type == "expression_statement":
                expr = first_stmt.children[0] if first_stmt.children else None
                if expr and expr.type == "string":
                    docstring = source[expr.start_byte : expr.end_byte].decode("utf-8")

        return ClassInfo(
            name=class_name,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            base_classes=base_classes,
            source_code=source_code,
            decorators=decorators,
            references=references,
            docstring=docstring,
        )

    def _extract_function_info(
        self, node, file_path: Path, source: bytes
    ) -> Optional[FunctionInfo]:
        """Extract function information from a function_definition node."""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return None
        func_name = source[name_node.start_byte : name_node.end_byte].decode("utf-8")

        # Skip private functions
        if func_name.startswith("_") and not func_name.startswith("__"):
            return None

        # Get decorators
        decorators = []
        prev_sibling = node.prev_sibling
        while prev_sibling and prev_sibling.type == "decorator":
            dec_text = source[
                prev_sibling.start_byte : prev_sibling.end_byte
            ].decode("utf-8")
            decorators.insert(0, dec_text)
            prev_sibling = prev_sibling.prev_sibling

        # Get source code
        if decorators:
            dec_start = node.start_byte
            temp = node.prev_sibling
            while temp and temp.type == "decorator":
                dec_start = temp.start_byte
                temp = temp.prev_sibling
            source_code = source[dec_start : node.end_byte].decode("utf-8")
            start_line = source[:dec_start].count(b"\n") + 1
        else:
            source_code = source[node.start_byte : node.end_byte].decode("utf-8")
            start_line = source[: node.start_byte].count(b"\n") + 1

        end_line = source[: node.end_byte].count(b"\n") + 1

        references = self._find_references(node, source)

        return FunctionInfo(
            name=func_name,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            source_code=source_code,
            decorators=decorators,
            references=references,
        )

    def _find_references(self, node, source: bytes) -> Set[str]:
        """Find all identifier references in a node (potential dependencies)."""
        references = set()

        for child in self._walk_tree(node):
            # Type annotations
            if child.type == "type":
                type_text = source[child.start_byte : child.end_byte].decode("utf-8")
                # Extract class names from type annotations
                for match in re.findall(r"\b([A-Z][a-zA-Z0-9_]*)\b", type_text):
                    references.add(match)

            # Identifiers that look like class names (PascalCase)
            elif child.type == "identifier":
                ident = source[child.start_byte : child.end_byte].decode("utf-8")
                # Only consider PascalCase identifiers as potential class references
                if ident[0].isupper() and not ident.isupper():
                    references.add(ident)

            # Function calls that might be class instantiation
            elif child.type == "call":
                func = child.child_by_field_name("function")
                if func and func.type == "identifier":
                    func_name = source[func.start_byte : func.end_byte].decode("utf-8")
                    if func_name[0].isupper():
                        references.add(func_name)
                elif func and func.type == "attribute":
                    attr_text = source[func.start_byte : func.end_byte].decode("utf-8")
                    parts = attr_text.split(".")
                    if parts[-1][0].isupper():
                        references.add(parts[-1])

        return references


# =============================================================================
# Dependency Graph Builder
# =============================================================================


class DependencyGraphBuilder:
    """Builds and manages the dependency graph for classes."""

    def __init__(self, classes: Dict[str, ClassInfo]):
        """
        Initialize with parsed classes.

        Args:
            classes: Dictionary mapping class names to ClassInfo
        """
        self.classes = classes
        self.graph = nx.DiGraph()
        self._build_graph()

    def _build_graph(self):
        """Build the dependency graph from class references."""
        # Add all classes as nodes
        for name, info in self.classes.items():
            self.graph.add_node(
                name,
                file_path=str(info.file_path),
                lines=info.end_line - info.start_line + 1,
            )

        # Add edges based on references
        for name, info in self.classes.items():
            for ref in info.references:
                if ref in self.classes and ref != name:
                    self.graph.add_edge(name, ref)

            # Also add edges for base classes
            for base in info.base_classes:
                # Strip module prefix if present
                base_name = base.split(".")[-1]
                if base_name in self.classes and base_name != name:
                    self.graph.add_edge(name, base_name)

    def get_transitive_dependencies(
        self,
        component_name: str,
        max_depth: int = MAX_RECURSION_DEPTH,
    ) -> Set[str]:
        """
        Get all transitive dependencies of a component.

        Args:
            component_name: Name of the target component
            max_depth: Maximum recursion depth

        Returns:
            Set of dependency class names
        """
        if component_name not in self.graph:
            return set()

        dependencies = set()
        visited = set()

        def _collect_deps(node: str, depth: int):
            if depth > max_depth or node in visited:
                return
            visited.add(node)

            for successor in self.graph.successors(node):
                if successor not in visited:
                    dependencies.add(successor)
                    _collect_deps(successor, depth + 1)

        _collect_deps(component_name, 0)
        return dependencies

    def topological_sort(self, nodes: Set[str]) -> List[str]:
        """
        Topologically sort a subset of nodes (dependencies before dependents).

        Args:
            nodes: Set of node names to sort

        Returns:
            List of node names in dependency order
        """
        subgraph = self.graph.subgraph(nodes).copy()

        # Handle cycles by breaking them
        while True:
            try:
                cycles = list(nx.simple_cycles(subgraph))
                if not cycles:
                    break
                # Break the cycle at the node with fewest incoming edges
                cycle = cycles[0]
                min_node = min(cycle, key=lambda n: subgraph.in_degree(n))
                # Remove one edge to break the cycle
                pred = cycle[(cycle.index(min_node) - 1) % len(cycle)]
                subgraph.remove_edge(pred, min_node)
                logger.warning(f"Broke circular dependency: {pred} -> {min_node}")
            except nx.NetworkXNoCycle:
                break

        try:
            # Reverse topological sort so dependencies come first
            return list(reversed(list(nx.topological_sort(subgraph))))
        except nx.NetworkXUnfeasible:
            # Fallback: just return nodes in some order
            logger.warning("Could not perform topological sort, using arbitrary order")
            return list(nodes)

    def detect_cycles(self) -> List[List[str]]:
        """Detect cycles in the dependency graph."""
        return list(nx.simple_cycles(self.graph))


# =============================================================================
# Import Analyzer
# =============================================================================


class ImportAnalyzer:
    """Analyzes and categorizes imports from Python source code."""

    def __init__(self):
        self.parser = get_parser("python")

    def extract_imports(self, source_code: str) -> List[ImportInfo]:
        """Extract all imports from source code."""
        imports = []

        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            # Fallback to regex-based extraction
            return self._extract_imports_regex(source_code)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(
                        ImportInfo(
                            module=alias.name,
                            names=[alias.name.split(".")[-1]],
                            alias=alias.asname,
                        )
                    )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(
                        ImportInfo(
                            module=node.module,
                            names=[alias.name for alias in node.names],
                            is_relative=node.level > 0,
                            level=node.level,
                        )
                    )

        return imports

    def _extract_imports_regex(self, source_code: str) -> List[ImportInfo]:
        """Fallback regex-based import extraction."""
        imports = []

        # Match "import X" and "import X as Y"
        for match in re.finditer(r"^import\s+([\w.]+)(?:\s+as\s+(\w+))?", source_code, re.MULTILINE):
            imports.append(
                ImportInfo(
                    module=match.group(1),
                    names=[match.group(1).split(".")[-1]],
                    alias=match.group(2),
                )
            )

        # Match "from X import Y, Z"
        for match in re.finditer(
            r"^from\s+(\.*)?([\w.]*)\s+import\s+(.+?)$", source_code, re.MULTILINE
        ):
            level = len(match.group(1)) if match.group(1) else 0
            module = match.group(2) or ""
            names_str = match.group(3)
            names = [n.strip().split(" as ")[0] for n in names_str.split(",")]
            imports.append(
                ImportInfo(
                    module=module,
                    names=names,
                    is_relative=level > 0,
                    level=level,
                )
            )

        return imports

    def categorize_import(self, import_info: ImportInfo) -> str:
        """
        Categorize an import as stdlib, third_party, or internal.

        Returns:
            One of: "stdlib", "third_party", "internal"
        """
        if import_info.is_relative:
            return "internal"

        module_root = import_info.module.split(".")[0] if import_info.module else ""

        if module_root in STDLIB_MODULES:
            return "stdlib"
        elif module_root in THIRD_PARTY_MODULES:
            return "third_party"
        else:
            # Heuristic: if it looks like a package name, treat as internal
            return "third_party" if module_root else "internal"

    def collect_required_imports(
        self,
        classes: List[ClassInfo],
        internal_class_names: Set[str],
    ) -> Dict[str, Set[str]]:
        """
        Collect all required external imports from a list of classes.

        Args:
            classes: List of ClassInfo objects
            internal_class_names: Names that are being inlined (don't need import)

        Returns:
            Dictionary with keys "stdlib", "third_party" containing import statements
        """
        result = {"stdlib": set(), "third_party": set()}

        for class_info in classes:
            imports = self.extract_imports(class_info.source_code)

            for imp in imports:
                category = self.categorize_import(imp)

                if category == "internal":
                    # Skip internal imports - these are being inlined
                    continue

                # Generate import statement
                if imp.is_relative:
                    continue

                if imp.module:
                    # Filter out names that are being inlined
                    external_names = [
                        n for n in imp.names if n not in internal_class_names
                    ]
                    if external_names:
                        if len(external_names) == 1 and external_names[0] == "*":
                            stmt = f"from {imp.module} import *"
                        else:
                            stmt = f"from {imp.module} import {', '.join(external_names)}"
                        result[category].add(stmt)
                else:
                    stmt = f"import {imp.names[0]}"
                    if imp.alias:
                        stmt += f" as {imp.alias}"
                    result[category].add(stmt)

        return result


# =============================================================================
# Stub Generator
# =============================================================================


class StubGenerator:
    """Generates minimal stubs for complex dependencies."""

    def __init__(self, classes: Dict[str, ClassInfo]):
        self.classes = classes

    def generate_stub(
        self, class_name: str, used_attributes: Optional[Set[str]] = None
    ) -> str:
        """
        Generate a minimal stub for a class.

        Args:
            class_name: Name of the class to stub
            used_attributes: Attributes actually used (if known)

        Returns:
            Stub source code
        """
        if class_name not in self.classes:
            return self._generate_basic_stub(class_name)

        class_info = self.classes[class_name]

        # Analyze what attributes are accessed
        if used_attributes is None:
            used_attributes = self._find_used_attributes(class_name)

        # Generate stub
        lines = []

        # Add docstring
        lines.append(f'class {class_name}:')
        lines.append(f'    """')
        lines.append(f"    Stub for {class_name}.")
        lines.append(f"    Original: {class_info.file_path}")
        lines.append(f'    """')
        lines.append("")

        # Generate __init__ with used attributes
        lines.append("    def __init__(self, **kwargs):")
        if used_attributes:
            for attr in sorted(used_attributes):
                lines.append(f"        self.{attr} = kwargs.get('{attr}', None)")
        else:
            lines.append("        pass")

        return "\n".join(lines)

    def _generate_basic_stub(self, class_name: str) -> str:
        """Generate a basic stub when class info is not available."""
        return f'''class {class_name}:
    """Stub for {class_name}."""
    
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
'''

    def _find_used_attributes(self, class_name: str) -> Set[str]:
        """Find attributes of a class that are actually used in other classes."""
        used = set()

        for other_name, other_info in self.classes.items():
            if other_name == class_name:
                continue

            # Look for patterns like "config.hidden_size" or "self.config.hidden_size"
            for match in re.finditer(
                rf"\.({class_name.lower()}|config|cfg)\.([\w_]+)",
                other_info.source_code,
                re.IGNORECASE,
            ):
                used.add(match.group(2))

        return used


# =============================================================================
# Code Inliner
# =============================================================================


class CodeInliner:
    """Inlines extracted code into a single self-contained file."""

    def __init__(self, repo_url: Optional[str] = None):
        self.repo_url = repo_url or "unknown"
        self.import_analyzer = ImportAnalyzer()

    def inline(
        self,
        component: ClassInfo,
        dependencies: List[ClassInfo],
        metadata: Dict[str, Any],
    ) -> str:
        """
        Inline component and dependencies into a single file.

        Args:
            component: The main component ClassInfo
            dependencies: List of dependency ClassInfo objects (topologically sorted)
            metadata: Extraction metadata

        Returns:
            Complete Python source code
        """
        lines = []

        # Header
        lines.append('"""')
        lines.append(f"Auto-extracted from {self.repo_url}")
        lines.append(f"Component: {component.name}")
        lines.append(f"Dependencies: {', '.join(d.name for d in dependencies)}")
        lines.append(f"Total lines: {metadata.get('total_lines', 'unknown')}")
        lines.append('"""')
        lines.append("")

        # Collect all class names being inlined
        all_classes = {component.name} | {d.name for d in dependencies}

        # Collect and organize imports
        all_code_classes = dependencies + [component]
        imports = self.import_analyzer.collect_required_imports(
            all_code_classes, all_classes
        )

        # Add imports section
        lines.append("# " + "=" * 77)
        lines.append("# IMPORTS")
        lines.append("# " + "=" * 77)
        lines.append("")

        # Stdlib imports
        if imports["stdlib"]:
            for stmt in sorted(imports["stdlib"]):
                lines.append(stmt)
            lines.append("")

        # Third-party imports
        if imports["third_party"]:
            for stmt in sorted(imports["third_party"]):
                lines.append(stmt)
            lines.append("")

        # Add common ML imports if not present
        common_imports = [
            "import torch",
            "import torch.nn as nn",
            "import torch.nn.functional as F",
        ]
        for imp in common_imports:
            if not any(imp in s for s in imports["third_party"]):
                lines.append(imp)
        lines.append("")

        # Dependencies section
        if dependencies:
            lines.append("# " + "=" * 77)
            lines.append("# DEPENDENCIES (topologically sorted)")
            lines.append("# " + "=" * 77)
            lines.append("")

            for dep in dependencies:
                # Add file origin comment
                lines.append(f"# From: {dep.file_path.name}")
                lines.append("")

                # Clean up the source code
                source = self._clean_source(dep.source_code, all_classes)
                lines.append(source)
                lines.append("")
                lines.append("")

        # Main component section
        lines.append("# " + "=" * 77)
        lines.append("# MAIN COMPONENT")
        lines.append("# " + "=" * 77)
        lines.append("")
        lines.append(f"# From: {component.file_path.name}")
        lines.append("")

        source = self._clean_source(component.source_code, all_classes)
        lines.append(source)
        lines.append("")

        return "\n".join(lines)

    def _clean_source(self, source: str, internal_classes: Set[str]) -> str:
        """
        Clean source code by removing internal imports.

        Args:
            source: Original source code
            internal_classes: Class names that are being inlined

        Returns:
            Cleaned source code
        """
        lines = source.split("\n")
        cleaned = []

        for line in lines:
            # Skip relative imports
            if re.match(r"^\s*from\s+\.\S*\s+import", line):
                continue
            # Skip imports of inlined classes
            skip = False
            for cls_name in internal_classes:
                if re.search(rf"\bimport\b.*\b{cls_name}\b", line):
                    skip = True
                    break
            if not skip:
                cleaned.append(line)

        return "\n".join(cleaned)


# =============================================================================
# Main Extractor Class
# =============================================================================


class DependencyExtractor:
    """
    Extracts components with dependencies from Python repositories.

    This class handles the complete extraction pipeline:
    1. Parse repository using tree-sitter
    2. Build dependency graph
    3. Find target component
    4. Compute transitive dependencies
    5. Extract and inline code
    """

    def __init__(
        self,
        max_depth: int = MAX_RECURSION_DEPTH,
        max_lines: int = MAX_TOTAL_LINES,
        max_class_lines: int = MAX_SINGLE_CLASS_LINES,
    ):
        """
        Initialize the extractor.

        Args:
            max_depth: Maximum recursion depth for dependencies
            max_lines: Maximum total lines to extract
            max_class_lines: Maximum lines for a single class before stubbing
        """
        self.max_depth = max_depth
        self.max_lines = max_lines
        self.max_class_lines = max_class_lines

        self.parser = RepositoryParser()
        self.import_analyzer = ImportAnalyzer()
        self.stub_generator: Optional[StubGenerator] = None

        # State
        self.classes: Dict[str, ClassInfo] = {}
        self.graph_builder: Optional[DependencyGraphBuilder] = None

    def extract_component(
        self,
        repo_path: str,
        component_name: str,
        component_type: str = "attention",
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Extract a component and ALL its dependencies from a repository.

        Args:
            repo_path: Path to cloned git repository
            component_name: Name of class to extract (e.g., "Qwen3Attention")
            component_type: Type hint for component ("attention", "embedding", "norm", "layer")

        Returns:
            Tuple of (extracted_code, metadata)

            extracted_code: Single Python file with all dependencies inlined
            metadata: {
                'component_name': str,
                'dependencies': List[str],
                'total_lines': int,
                'imports': List[str],
                'files_processed': int,
                'stub_generated': bool
            }

        Raises:
            ComponentNotFoundError: If component_name not found in repo
            ExtractionError: If extraction fails for other reasons
            ComplexityError: If dependencies exceed line limit
        """
        repo = Path(repo_path)
        if not repo.exists():
            raise ExtractionError(f"Repository path does not exist: {repo_path}")

        logger.info(f"Starting extraction of {component_name} from {repo_path}")

        # Step 1: Parse repository
        logger.info("Parsing repository...")
        self.classes = self.parser.parse_repository(repo)
        self.stub_generator = StubGenerator(self.classes)

        if not self.classes:
            raise ExtractionError("No Python classes found in repository")

        # Step 2: Find target component
        logger.info(f"Looking for component: {component_name}")
        if component_name not in self.classes:
            suggestions = self._find_similar_names(component_name)
            raise ComponentNotFoundError(component_name, suggestions)

        component = self.classes[component_name]
        logger.info(f"Found {component_name} in {component.file_path}")

        # Step 3: Build dependency graph
        logger.info("Building dependency graph...")
        self.graph_builder = DependencyGraphBuilder(self.classes)

        # Step 4: Get transitive dependencies
        logger.info("Computing transitive dependencies...")
        dep_names = self.graph_builder.get_transitive_dependencies(
            component_name, self.max_depth
        )
        logger.info(f"Found {len(dep_names)} dependencies: {dep_names}")

        # Step 5: Check for external dependencies (not in repo)
        external_refs = self._find_external_references(component, dep_names)
        logger.debug(f"External references: {external_refs}")

        # Step 6: Sort dependencies topologically
        all_names = dep_names | {component_name}
        sorted_names = self.graph_builder.topological_sort(all_names)
        sorted_names = [n for n in sorted_names if n != component_name]

        # Step 7: Extract code and handle complexity
        dependencies: List[ClassInfo] = []
        total_lines = component.end_line - component.start_line + 1
        stub_generated = False

        for dep_name in sorted_names:
            if dep_name not in self.classes:
                continue

            dep_info = self.classes[dep_name]
            dep_lines = dep_info.end_line - dep_info.start_line + 1

            # Check if we need to stub this dependency
            if dep_lines > self.max_class_lines:
                logger.info(f"Stubbing {dep_name} ({dep_lines} lines > {self.max_class_lines})")
                stub_code = self.stub_generator.generate_stub(dep_name)
                dep_info = ClassInfo(
                    name=dep_name,
                    file_path=dep_info.file_path,
                    start_line=0,
                    end_line=stub_code.count("\n"),
                    source_code=stub_code,
                    is_stubbed=True,
                )
                stub_generated = True

            # Check total line limit
            if total_lines + dep_lines > self.max_lines:
                logger.warning(
                    f"Line limit reached ({total_lines + dep_lines} > {self.max_lines}), "
                    f"stubbing remaining dependencies"
                )
                stub_code = self.stub_generator.generate_stub(dep_name)
                dep_info = ClassInfo(
                    name=dep_name,
                    file_path=dep_info.file_path,
                    start_line=0,
                    end_line=stub_code.count("\n"),
                    source_code=stub_code,
                    is_stubbed=True,
                )
                stub_generated = True

            dependencies.append(dep_info)
            total_lines += dep_info.end_line - dep_info.start_line + 1

        # Step 8: Inline code
        logger.info("Inlining dependencies...")
        repo_url = self._detect_repo_url(repo)
        inliner = CodeInliner(repo_url)

        metadata = {
            "component_name": component_name,
            "component_type": component_type,
            "dependencies": [d.name for d in dependencies],
            "total_lines": total_lines,
            "files_processed": len(set(d.file_path for d in dependencies) | {component.file_path}),
            "stub_generated": stub_generated,
            "imports": [],  # Will be populated below
        }

        extracted_code = inliner.inline(component, dependencies, metadata)

        # Extract actual imports from generated code
        actual_imports = self.import_analyzer.extract_imports(extracted_code)
        metadata["imports"] = [
            f"{imp.module}: {', '.join(imp.names)}" for imp in actual_imports if imp.module
        ]

        # Step 9: Validate output
        logger.info("Validating extracted code...")
        self._validate_code(extracted_code)

        logger.info(
            f"Extraction complete: {total_lines} lines, "
            f"{len(dependencies)} dependencies"
        )

        return extracted_code, metadata

    def _find_similar_names(self, target: str, max_suggestions: int = 5) -> List[str]:
        """Find class names similar to target for error suggestions."""
        from difflib import SequenceMatcher

        def similarity(a: str, b: str) -> float:
            return SequenceMatcher(None, a.lower(), b.lower()).ratio()

        scored = [(name, similarity(name, target)) for name in self.classes.keys()]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [name for name, score in scored[:max_suggestions] if score > 0.3]

    def _find_external_references(
        self, component: ClassInfo, internal_deps: Set[str]
    ) -> Set[str]:
        """Find references that aren't satisfied by internal dependencies."""
        all_refs = component.references.copy()
        for dep_name in internal_deps:
            if dep_name in self.classes:
                all_refs |= self.classes[dep_name].references

        # Filter out internal classes and known external
        internal = set(self.classes.keys())
        known_external = {"nn", "F", "torch", "Tensor", "Module", "Optional", "List", "Dict", "Tuple", "Any", "Union", "Callable"}

        return all_refs - internal - internal_deps - known_external

    def _detect_repo_url(self, repo_path: Path) -> str:
        """Try to detect the repository URL from git config."""
        git_config = repo_path / ".git" / "config"
        if git_config.exists():
            try:
                content = git_config.read_text()
                match = re.search(r'url\s*=\s*(.+)', content)
                if match:
                    return match.group(1).strip()
            except Exception:
                pass
        return str(repo_path)

    def _validate_code(self, code: str) -> bool:
        """Validate that extracted code is syntactically correct."""
        try:
            ast.parse(code)
            logger.info("Code validation passed (syntax OK)")
            return True
        except SyntaxError as e:
            logger.error(f"Syntax error in extracted code: {e}")
            raise ExtractionError(f"Generated code has syntax error: {e}")


# =============================================================================
# Convenience Function
# =============================================================================


def extract_component(
    repo_path: str,
    component_name: str,
    component_type: str = "attention",
) -> Tuple[str, Dict[str, Any]]:
    """
    Convenience wrapper around DependencyExtractor.

    Args:
        repo_path: Path to cloned git repository
        component_name: Name of class to extract (e.g., "Qwen3Attention")
        component_type: Type hint for component ("attention", "embedding", "norm", "layer")

    Returns:
        Tuple of (extracted_code, metadata)

    Example:
        >>> code, meta = extract_component(
        ...     "/tmp/gated_attention",
        ...     "Qwen3Attention"
        ... )
        >>> print(f"Extracted {meta['total_lines']} lines")
        >>> print(f"Dependencies: {meta['dependencies']}")
    """
    extractor = DependencyExtractor()
    return extractor.extract_component(repo_path, component_name, component_type)


# =============================================================================
# CLI Interface
# =============================================================================


def main():
    """Command-line interface for dependency extraction."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Extract ML components with dependencies from Python repositories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python dependency_extractor.py /path/to/repo Qwen3Attention
    python dependency_extractor.py /path/to/repo MyCustomLayer -o extracted.py
    python dependency_extractor.py /path/to/repo AttentionModule --max-depth 2
        """,
    )
    parser.add_argument("repo_path", help="Path to the repository")
    parser.add_argument("component_name", help="Name of the component to extract")
    parser.add_argument(
        "-t",
        "--type",
        default="attention",
        choices=["attention", "embedding", "norm", "layer"],
        help="Component type (default: attention)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=MAX_RECURSION_DEPTH,
        help=f"Maximum dependency depth (default: {MAX_RECURSION_DEPTH})",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=MAX_TOTAL_LINES,
        help=f"Maximum total lines (default: {MAX_TOTAL_LINES})",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        extractor = DependencyExtractor(
            max_depth=args.max_depth,
            max_lines=args.max_lines,
        )

        code, metadata = extractor.extract_component(
            args.repo_path,
            args.component_name,
            args.type,
        )

        # Output
        if args.output:
            Path(args.output).write_text(code)
            print(f"Extracted code written to {args.output}", file=sys.stderr)
        else:
            print(code)

        # Print metadata to stderr
        print("\n--- Extraction Metadata ---", file=sys.stderr)
        print(f"Component: {metadata['component_name']}", file=sys.stderr)
        print(f"Dependencies: {', '.join(metadata['dependencies'])}", file=sys.stderr)
        print(f"Total lines: {metadata['total_lines']}", file=sys.stderr)
        print(f"Files processed: {metadata['files_processed']}", file=sys.stderr)
        print(f"Stubs generated: {metadata['stub_generated']}", file=sys.stderr)

    except ComponentNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ExtractionError as e:
        print(f"Extraction failed: {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(3)


if __name__ == "__main__":
    main()