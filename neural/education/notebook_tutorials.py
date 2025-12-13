"""
Interactive Jupyter notebook tutorials for Neural DSL.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import nbformat
    from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
    NBFORMAT_AVAILABLE = True
except ImportError:
    nbformat = None
    NBFORMAT_AVAILABLE = False

from .models import DifficultyLevel


class NotebookTutorial:
    """Interactive Jupyter notebook tutorial."""
    
    def __init__(
        self,
        tutorial_id: str,
        title: str,
        description: str,
        difficulty: DifficultyLevel = DifficultyLevel.BEGINNER,
        estimated_time: int = 30,
        learning_objectives: Optional[List[str]] = None,
        prerequisites: Optional[List[str]] = None,
    ):
        self.tutorial_id = tutorial_id
        self.title = title
        self.description = description
        self.difficulty = difficulty
        self.estimated_time = estimated_time
        self.learning_objectives = learning_objectives or []
        self.prerequisites = prerequisites or []
        self.cells: List[Dict[str, Any]] = []
        self.xp_reward = self._calculate_xp()
    
    def _calculate_xp(self) -> int:
        """Calculate XP reward based on difficulty."""
        base_xp = {
            DifficultyLevel.BEGINNER: 100,
            DifficultyLevel.INTERMEDIATE: 250,
            DifficultyLevel.ADVANCED: 500,
            DifficultyLevel.EXPERT: 1000,
        }
        return base_xp.get(self.difficulty, 100)
    
    def add_markdown(self, content: str) -> None:
        """Add a markdown cell to the tutorial."""
        self.cells.append({"type": "markdown", "content": content})
    
    def add_code(
        self,
        code: str,
        explanation: str = "",
        validation: Optional[str] = None,
    ) -> None:
        """Add a code cell with optional validation."""
        self.cells.append({
            "type": "code",
            "content": code,
            "explanation": explanation,
            "validation": validation,
        })
    
    def add_exercise(
        self,
        instruction: str,
        starter_code: str = "",
        solution: str = "",
        hints: Optional[List[str]] = None,
    ) -> None:
        """Add an exercise cell."""
        self.cells.append({
            "type": "exercise",
            "instruction": instruction,
            "starter_code": starter_code,
            "solution": solution,
            "hints": hints or [],
        })
    
    def to_notebook(self) -> Any:
        """Convert to Jupyter notebook format."""
        if not NBFORMAT_AVAILABLE:
            raise ImportError("nbformat is required for notebook generation")
        
        nb = new_notebook()
        
        title_cell = new_markdown_cell(f"# {self.title}\n\n{self.description}")
        nb.cells.append(title_cell)
        
        if self.learning_objectives:
            objectives = "\n".join(f"- {obj}" for obj in self.learning_objectives)
            nb.cells.append(new_markdown_cell(f"## Learning Objectives\n\n{objectives}"))
        
        for cell in self.cells:
            if cell["type"] == "markdown":
                nb.cells.append(new_markdown_cell(cell["content"]))
            elif cell["type"] == "code":
                if cell.get("explanation"):
                    nb.cells.append(new_markdown_cell(cell["explanation"]))
                nb.cells.append(new_code_cell(cell["content"]))
                if cell.get("validation"):
                    nb.cells.append(new_code_cell(cell["validation"]))
            elif cell["type"] == "exercise":
                ex_md = f"## Exercise\n\n{cell['instruction']}"
                if cell.get("hints"):
                    ex_md += "\n\n**Hints:**\n" + "\n".join(
                        f"- {hint}" for hint in cell["hints"]
                    )
                nb.cells.append(new_markdown_cell(ex_md))
                nb.cells.append(new_code_cell(cell.get("starter_code", "")))
        
        return nb
    
    def save(self, path: str) -> None:
        """Save tutorial as Jupyter notebook."""
        if not NBFORMAT_AVAILABLE:
            raise ImportError("nbformat is required for saving notebooks")
        
        nb = self.to_notebook()
        with open(path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tutorial_id": self.tutorial_id,
            "title": self.title,
            "description": self.description,
            "difficulty": self.difficulty.value,
            "estimated_time": self.estimated_time,
            "learning_objectives": self.learning_objectives,
            "prerequisites": self.prerequisites,
            "cells": self.cells,
            "xp_reward": self.xp_reward,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> NotebookTutorial:
        """Create from dictionary."""
        data = data.copy()
        data["difficulty"] = DifficultyLevel(data["difficulty"])
        tutorial = cls(**{k: v for k, v in data.items() if k not in ["cells", "xp_reward"]})
        tutorial.cells = data.get("cells", [])
        return tutorial


class TutorialLibrary:
    """Library of pre-built tutorials."""
    
    def __init__(self, storage_dir: str = "neural_education_tutorials"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.tutorials: Dict[str, NotebookTutorial] = {}
        self._load_tutorials()
    
    def _load_tutorials(self) -> None:
        """Load tutorials from storage."""
        index_file = self.storage_dir / "index.json"
        if index_file.exists():
            with open(index_file, 'r') as f:
                index = json.load(f)
                for tutorial_data in index.get("tutorials", []):
                    tutorial = NotebookTutorial.from_dict(tutorial_data)
                    self.tutorials[tutorial.tutorial_id] = tutorial
    
    def _save_index(self) -> None:
        """Save tutorial index."""
        index = {
            "tutorials": [t.to_dict() for t in self.tutorials.values()]
        }
        with open(self.storage_dir / "index.json", 'w') as f:
            json.dump(index, f, indent=2)
    
    def add_tutorial(self, tutorial: NotebookTutorial) -> None:
        """Add a tutorial to the library."""
        self.tutorials[tutorial.tutorial_id] = tutorial
        tutorial.save(str(self.storage_dir / f"{tutorial.tutorial_id}.ipynb"))
        self._save_index()
    
    def get_tutorial(self, tutorial_id: str) -> Optional[NotebookTutorial]:
        """Get a tutorial by ID."""
        return self.tutorials.get(tutorial_id)
    
    def list_tutorials(
        self,
        difficulty: Optional[DifficultyLevel] = None,
    ) -> List[NotebookTutorial]:
        """List tutorials, optionally filtered by difficulty."""
        tutorials = list(self.tutorials.values())
        if difficulty:
            tutorials = [t for t in tutorials if t.difficulty == difficulty]
        return sorted(tutorials, key=lambda t: t.estimated_time)
    
    @classmethod
    def create_default_tutorials(cls) -> TutorialLibrary:
        """Create library with default tutorials."""
        library = cls()
        
        intro_tutorial = NotebookTutorial(
            tutorial_id="intro-neural-dsl",
            title="Introduction to Neural DSL",
            description="Learn the basics of Neural DSL syntax and concepts.",
            difficulty=DifficultyLevel.BEGINNER,
            estimated_time=30,
            learning_objectives=[
                "Understand Neural DSL syntax",
                "Define simple neural networks",
                "Compile DSL to Python code",
            ],
        )
        
        intro_tutorial.add_markdown("## Welcome to Neural DSL!")
        intro_tutorial.add_markdown(
            "Neural DSL is a domain-specific language for defining neural networks. "
            "Let's start by defining a simple network."
        )
        
        intro_tutorial.add_code(
            code="""from neural.parser import create_parser

dsl_code = '''
network SimpleNet {
    input: [28, 28, 1]
    
    layer conv1: Conv2D(filters=32, kernel_size=3, activation='relu')
    layer pool1: MaxPool2D(pool_size=2)
    layer flatten: Flatten()
    layer dense1: Dense(units=128, activation='relu')
    layer output: Dense(units=10, activation='softmax')
    
    flow: input -> conv1 -> pool1 -> flatten -> dense1 -> output
}
'''

parser = create_parser('network')
ast = parser.parse(dsl_code)
print("Parsed successfully!")
""",
            explanation="This code parses a simple convolutional network definition.",
        )
        
        intro_tutorial.add_exercise(
            instruction="Now try defining your own network with at least 3 layers.",
            starter_code="""dsl_code = '''
network MyNet {
    input: [32, 32, 3]
    
    # Your layers here
    
    flow: # Your flow here
}
'''
""",
            hints=[
                "Start with a Conv2D layer",
                "Add pooling for dimensionality reduction",
                "End with a Dense layer for classification",
            ],
        )
        
        library.add_tutorial(intro_tutorial)
        
        cnn_tutorial = NotebookTutorial(
            tutorial_id="building-cnns",
            title="Building Convolutional Neural Networks",
            description="Learn to build CNNs for image classification.",
            difficulty=DifficultyLevel.INTERMEDIATE,
            estimated_time=60,
            learning_objectives=[
                "Understand CNN architectures",
                "Implement conv and pooling layers",
                "Use batch normalization and dropout",
            ],
            prerequisites=["intro-neural-dsl"],
        )
        
        cnn_tutorial.add_markdown("## Convolutional Neural Networks")
        cnn_tutorial.add_markdown(
            "CNNs are the backbone of modern computer vision. "
            "Let's build a CNN for MNIST digit classification."
        )
        
        cnn_tutorial.add_code(
            code="""dsl_code = '''
network MNIST_CNN {
    input: [28, 28, 1]
    
    layer conv1: Conv2D(filters=32, kernel_size=3, activation='relu')
    layer bn1: BatchNorm()
    layer pool1: MaxPool2D(pool_size=2)
    layer dropout1: Dropout(rate=0.25)
    
    layer conv2: Conv2D(filters=64, kernel_size=3, activation='relu')
    layer bn2: BatchNorm()
    layer pool2: MaxPool2D(pool_size=2)
    layer dropout2: Dropout(rate=0.25)
    
    layer flatten: Flatten()
    layer dense1: Dense(units=128, activation='relu')
    layer dropout3: Dropout(rate=0.5)
    layer output: Dense(units=10, activation='softmax')
    
    flow: input -> conv1 -> bn1 -> pool1 -> dropout1 
          -> conv2 -> bn2 -> pool2 -> dropout2 
          -> flatten -> dense1 -> dropout3 -> output
}
'''
""",
            explanation="This CNN uses batch normalization and dropout for better training.",
        )
        
        library.add_tutorial(cnn_tutorial)
        
        return library
