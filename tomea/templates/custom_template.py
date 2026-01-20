"""
Custom Architecture Template - Generate adapters for papers with novel architectures.

Handles novel attention mechanisms, custom layers, and architectural modifications.
"""

import logging
import re
from typing import Dict, List, Optional, Any
from pathlib import Path

from .base import BaseTemplate, AdapterCode

logger = logging.getLogger(__name__)


class TemplateError(Exception):
    """Base exception for template errors."""
    pass


class ExtractionError(TemplateError):
    """Error during code extraction."""
    pass


class ComplexityError(TemplateError):
    """Error when code is too complex to handle."""
    pass


class CustomClassInfo:
    """Information about a custom class found in the repository."""
    
    def __init__(
        self,
        class_name: str,
        file_path: str,
        estimated_loc: int,
        base_classes: List[str] = None,
        methods: List[str] = None,
        is_nn_module: bool = False
    ):
        self.class_name = class_name
        self.file_path = file_path
        self.estimated_loc = estimated_loc
        self.base_classes = base_classes or []
        self.methods = methods or []
        self.is_nn_module = is_nn_module
        self.imports = []
        self.dependencies = []
    
    def to_dict(self) -> Dict:
        return {
            "class_name": self.class_name,
            "file_path": self.file_path,
            "estimated_loc": self.estimated_loc,
            "base_classes": self.base_classes,
            "is_nn_module": self.is_nn_module,
            "methods": self.methods,
        }


class CustomArchTemplate(BaseTemplate):
    """
    Template for custom architecture papers.
    
    Handles:
    - Novel attention mechanisms (Gated Attention, etc.)
    - Custom layers (< 500 lines)
    - Architectural modifications
    
    Strategy: Delegate to dependency_extractor for code extraction,
    then integration_generator handles integration (in builder.py)
    """
    MAX_CLASS_LOC = 500
    MAX_FILES = 5
    
    # Patterns indicating a class is a custom NN module
    NN_MODULE_PATTERNS = [
        r"\(nn\.Module\)",
        r"\(torch\.nn\.Module\)",
        r"\(Module\)",
    ]
    
    # Methods indicating an nn.Module
    NN_MODULE_METHODS = ["forward", "__init__", "reset_parameters"]
    
    def __init__(self, llm_client=None):
        super().__init__()
        self.llm = llm_client
        self.max_tokens = 1000
    
    def generate_adapter(self, repo_map: str, context: Dict) -> AdapterCode:
        """
        Generate adapter - DELEGATE to dependency extractor if repo_path available.
        
        Args:
            repo_map: Repository map from cartographer
            context: Must contain:
                - repo_path: Path to cloned repo (optional)
                - component_type: Type of component (attention/embedding/etc)
                - paper_name: Name of paper
        
        Returns:
            AdapterCode with extracted component + dependencies
        """
        repo_path = context.get('repo_path')
        
        # NEW PATH: If we have local repo, use dependency extractor
        if repo_path:
            return self._generate_with_extraction(repo_map, context, repo_path)
        
        # FALLBACK: No local repo → Cannot extract → Fail gracefully
        else:
            return self._generate_without_extraction(repo_map, context)
    
    def _generate_with_extraction(
        self,
        repo_map: str,
        context: Dict,
        repo_path: str
    ) -> AdapterCode:
        """Generate adapter using dependency extraction (NEW PATH)."""
        from tomea.core.dependency_extractor import DependencyExtractor
        
        # DEBUG: Print repo map
        print("\n[DEBUG] Repository map:")
        print(repo_map[:2000])  # First 2000 chars
        print("\n[DEBUG] Looking for custom classes...")
        
        # Identify component from repo map
        custom_classes = self._identify_custom_classes(repo_map)
        print(f"[DEBUG] Found {len(custom_classes)} custom classes: {[c.class_name for c in custom_classes]}")
        
        if not custom_classes:
            raise ExtractionError("No custom classes found in repository map")
        
        main_class = self._identify_main_class(custom_classes)
        if not main_class:
            raise ExtractionError("Could not identify main class to extract")
        
        component_name = main_class.class_name
        component_type = context.get('component_type', 'attention')
        
        logger.info(f"Extracting {component_name} ({component_type}) from {repo_path}")
        
        # Extract with dependencies using OPUS-generated module
        try:
            extractor = DependencyExtractor()
            print(f"[DEBUG] Extractor created successfully")
            
            component_code, metadata = extractor.extract_component(
                repo_path,
                component_name,
                component_type
            )
            print(f"[DEBUG] Extraction completed successfully")
            
        except Exception as e:
            import traceback
            print(f"[DEBUG] Full error traceback:")
            traceback.print_exc()
            raise ExtractionError(f"Failed to extract {component_name}: {e}")
        
        logger.info(
            f"Extracted {metadata['total_lines']} lines "
            f"with {len(metadata['dependencies'])} dependencies"
        )
        
        # Return as AdapterCode
        # NOTE: This is JUST the extracted component code
        # The integration (get_model function) will be generated by
        # integration_generator in builder.py
        return AdapterCode(
            code=component_code,
            method_type='custom_arch',
            estimated_loc=metadata['total_lines'],
            required_imports=metadata['imports'],
            config_params={
                'component_name': component_name,
                'dependencies': metadata['dependencies']
            }
        )
    
    def _generate_without_extraction(
        self,
        repo_map: str,
        context: Dict
    ) -> AdapterCode:
        """
        Fallback when local repo not available.
        
        Cannot extract code without local files, so we provide
        clear error message with manual steps.
        
        Raises:
            ExtractionError: Always (cannot proceed without repo)
        """
        # Identify what component we need (for helpful error message)
        try:
            custom_classes = self._identify_custom_classes(repo_map)
            
            if custom_classes:
                nn_modules = [c for c in custom_classes if c.is_nn_module]
                if not nn_modules:
                    nn_modules = custom_classes
                
                main_class = self._identify_main_class(nn_modules)
                component_name = main_class.class_name if main_class else "custom component"
                files_needed = list({c.file_path for c in nn_modules[:3]})
            else:
                component_name = "custom component"
                files_needed = ["unknown"]
        
        except Exception:
            component_name = "custom component"
            files_needed = ["unknown"]
        
        # Provide helpful error with manual steps
        error_msg = f"""Cannot extract {component_name} without local repository access.

Repository cloning failed or was not attempted.

REQUIRED FILES:
{chr(10).join(f'  - {f}' for f in files_needed)}

MANUAL STEPS:
1. Clone repository locally:
   git clone <repo_url>

2. Extract {component_name} class manually

3. Inline any dependencies (Config classes, utility functions)

4. Adapt initialization for target base model

5. Test with: python -c "from adapter import get_model; model = get_model('bert-base-uncased', 2)"

Alternatively, ensure git is installed and accessible for automatic extraction.
"""
        
        raise ExtractionError(error_msg)
    
    def _identify_custom_classes(self, repo_map: str) -> List[CustomClassInfo]:
        """
        Find custom nn.Module classes in repo map.
        
        Handles BOTH formats:
        1. Tree format: └── class Foo(nn.Module):
        2. List format: class Foo, class Bar, class Baz
        """
        custom_classes = []
        current_file = None
        
        for line in repo_map.split("\n"):
            # Match file path
            file_match = re.match(r"^([^\s:]+\.py):\s*\[Rank:", line)
            if file_match:
                current_file = file_match.group(1)
                continue
        
            if current_file and "class " in line and line.startswith("  "):
                # Extract all class names from "  class Foo, class Bar, class Baz"
                class_names = re.findall(r"class\s+(\w+)", line)
                
                for class_name in class_names:
                    # Filter to attention-related classes
                    if any(pattern in class_name for pattern in ["Attention", "Layer", "Block", "MLP", "Norm"]):
                        custom_classes.append(
                            CustomClassInfo(
                                class_name=class_name,
                                file_path=current_file,
                                estimated_loc=100,
                                base_classes=[],
                                methods=["__init__", "forward"],
                                is_nn_module=True
                            )
                        )

        print(f"[DEBUG] Parsed classes: {[c.class_name for c in custom_classes]}")
        return custom_classes
    
    def _create_class_info(
        self,
        class_name: str,
        file_path: str,
        methods: List[str],
        base_classes: List[str]
    ) -> CustomClassInfo:
        """Create CustomClassInfo object."""
        # Estimate LOC (rough: 15 lines per method)
        estimated_loc = max(20, len(methods) * 15)
        
        return CustomClassInfo(
            class_name=class_name,
            file_path=file_path,
            estimated_loc=estimated_loc,
            base_classes=base_classes,
            methods=methods,
            is_nn_module=False  # Set later
        )
    
    def _identify_main_class(
        self,
        classes: List[CustomClassInfo]
    ) -> Optional[CustomClassInfo]:
        """
        Identify main class to extract.
        
        Priority:
        1. Classes with "Attention" in name
        2. Classes with "Layer" in name
        3. Classes with most methods
        """
        if not classes:
            return None
        
        def priority(cls):
            attention_score = 10 if "attention" in cls.class_name.lower() else 0
            layer_score = 5 if "layer" in cls.class_name.lower() else 0
            method_score = len(cls.methods)
            return (attention_score, layer_score, method_score)
        
        sorted_classes = sorted(classes, key=priority, reverse=True)
        return sorted_classes[0]
    

    def get_required_files(self, repo_map: str) -> List[str]:
        """
        Identify files needed (required by BaseTemplate).
        
        For custom arch, we extract via dependency_extractor,
        so this is just for interface compliance.
        """
        # Parse repo map to find main component file
        custom_classes = self._identify_custom_classes(repo_map)
        if not custom_classes:
            return []
        
        main_class = self._identify_main_class(custom_classes)
        if not main_class:
            return []
        
        return [main_class.file_path]