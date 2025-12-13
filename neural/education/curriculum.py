"""
Curriculum and course management.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import DifficultyLevel


@dataclass
class Lesson:
    """A single lesson in a course."""
    lesson_id: str
    title: str
    description: str
    content: str
    difficulty: DifficultyLevel = DifficultyLevel.BEGINNER
    estimated_time: int = 30
    objectives: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    resources: List[Dict[str, str]] = field(default_factory=list)
    tutorial_ids: List[str] = field(default_factory=list)
    assignment_ids: List[str] = field(default_factory=list)
    order: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "lesson_id": self.lesson_id,
            "title": self.title,
            "description": self.description,
            "content": self.content,
            "difficulty": self.difficulty.value,
            "estimated_time": self.estimated_time,
            "objectives": self.objectives,
            "prerequisites": self.prerequisites,
            "resources": self.resources,
            "tutorial_ids": self.tutorial_ids,
            "assignment_ids": self.assignment_ids,
            "order": self.order,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Lesson:
        """Create from dictionary."""
        data = data.copy()
        data["difficulty"] = DifficultyLevel(data["difficulty"])
        return cls(**data)


@dataclass
class Course:
    """A course containing multiple lessons."""
    course_id: str
    name: str
    description: str
    teacher_id: str
    created_at: datetime = field(default_factory=datetime.now)
    difficulty: DifficultyLevel = DifficultyLevel.BEGINNER
    lessons: List[str] = field(default_factory=list)
    enrolled_students: List[str] = field(default_factory=list)
    syllabus: str = ""
    learning_outcomes: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    duration_weeks: int = 8
    credits: float = 3.0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "course_id": self.course_id,
            "name": self.name,
            "description": self.description,
            "teacher_id": self.teacher_id,
            "created_at": self.created_at.isoformat(),
            "difficulty": self.difficulty.value,
            "lessons": self.lessons,
            "enrolled_students": self.enrolled_students,
            "syllabus": self.syllabus,
            "learning_outcomes": self.learning_outcomes,
            "prerequisites": self.prerequisites,
            "duration_weeks": self.duration_weeks,
            "credits": self.credits,
            "tags": self.tags,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Course:
        """Create from dictionary."""
        data = data.copy()
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["difficulty"] = DifficultyLevel(data["difficulty"])
        return cls(**data)


class Curriculum:
    """Manages courses and lessons."""
    
    def __init__(self, storage_dir: str = "neural_education_curriculum"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.courses: Dict[str, Course] = {}
        self.lessons: Dict[str, Lesson] = {}
        self._load_data()
    
    def _load_data(self) -> None:
        """Load curriculum data from storage."""
        courses_file = self.storage_dir / "courses.json"
        if courses_file.exists():
            with open(courses_file, 'r') as f:
                data = json.load(f)
                self.courses = {
                    cid: Course.from_dict(cdata)
                    for cid, cdata in data.items()
                }
        
        lessons_file = self.storage_dir / "lessons.json"
        if lessons_file.exists():
            with open(lessons_file, 'r') as f:
                data = json.load(f)
                self.lessons = {
                    lid: Lesson.from_dict(ldata)
                    for lid, ldata in data.items()
                }
    
    def _save_data(self) -> None:
        """Save curriculum data to storage."""
        with open(self.storage_dir / "courses.json", 'w') as f:
            json.dump(
                {cid: c.to_dict() for cid, c in self.courses.items()},
                f,
                indent=2,
            )
        
        with open(self.storage_dir / "lessons.json", 'w') as f:
            json.dump(
                {lid: l.to_dict() for lid, l in self.lessons.items()},
                f,
                indent=2,
            )
    
    def create_course(
        self,
        name: str,
        description: str,
        teacher_id: str,
        **kwargs,
    ) -> Course:
        """Create a new course."""
        course = Course(
            course_id=str(uuid.uuid4()),
            name=name,
            description=description,
            teacher_id=teacher_id,
            **kwargs,
        )
        self.courses[course.course_id] = course
        self._save_data()
        return course
    
    def create_lesson(
        self,
        title: str,
        description: str,
        content: str,
        **kwargs,
    ) -> Lesson:
        """Create a new lesson."""
        lesson = Lesson(
            lesson_id=str(uuid.uuid4()),
            title=title,
            description=description,
            content=content,
            **kwargs,
        )
        self.lessons[lesson.lesson_id] = lesson
        self._save_data()
        return lesson
    
    def add_lesson_to_course(self, course_id: str, lesson_id: str) -> bool:
        """Add a lesson to a course."""
        course = self.courses.get(course_id)
        lesson = self.lessons.get(lesson_id)
        
        if not course or not lesson:
            return False
        
        if lesson_id not in course.lessons:
            course.lessons.append(lesson_id)
            self._save_data()
        
        return True
    
    def enroll_student(self, course_id: str, student_id: str) -> bool:
        """Enroll a student in a course."""
        course = self.courses.get(course_id)
        if not course:
            return False
        
        if student_id not in course.enrolled_students:
            course.enrolled_students.append(student_id)
            self._save_data()
        
        return True
    
    def get_course(self, course_id: str) -> Optional[Course]:
        """Get a course by ID."""
        return self.courses.get(course_id)
    
    def get_lesson(self, lesson_id: str) -> Optional[Lesson]:
        """Get a lesson by ID."""
        return self.lessons.get(lesson_id)
    
    def get_course_lessons(self, course_id: str) -> List[Lesson]:
        """Get all lessons for a course in order."""
        course = self.courses.get(course_id)
        if not course:
            return []
        
        lessons = [self.lessons[lid] for lid in course.lessons if lid in self.lessons]
        return sorted(lessons, key=lambda l: l.order)
    
    def list_courses(
        self,
        teacher_id: Optional[str] = None,
        difficulty: Optional[DifficultyLevel] = None,
    ) -> List[Course]:
        """List courses with optional filtering."""
        courses = list(self.courses.values())
        if teacher_id:
            courses = [c for c in courses if c.teacher_id == teacher_id]
        if difficulty:
            courses = [c for c in courses if c.difficulty == difficulty]
        return sorted(courses, key=lambda c: c.created_at, reverse=True)


class CurriculumTemplate:
    """Pre-built curriculum templates."""
    
    @staticmethod
    def intro_to_neural_networks() -> Course:
        """Create an introductory Neural Networks course."""
        course = Course(
            course_id="template-intro-nn",
            name="Introduction to Neural Networks",
            description="A comprehensive introduction to neural networks using Neural DSL",
            teacher_id="template",
            difficulty=DifficultyLevel.BEGINNER,
            duration_weeks=8,
            credits=3.0,
            learning_outcomes=[
                "Understand fundamental neural network concepts",
                "Build basic feedforward networks",
                "Implement convolutional neural networks",
                "Train and evaluate models",
            ],
            tags=["neural-networks", "deep-learning", "beginner"],
        )
        return course
    
    @staticmethod
    def deep_learning_fundamentals() -> Course:
        """Create a Deep Learning fundamentals course."""
        course = Course(
            course_id="template-dl-fundamentals",
            name="Deep Learning Fundamentals",
            description="Core concepts and architectures in deep learning",
            teacher_id="template",
            difficulty=DifficultyLevel.INTERMEDIATE,
            duration_weeks=12,
            credits=4.0,
            learning_outcomes=[
                "Master deep learning architectures",
                "Implement CNNs, RNNs, and Transformers",
                "Understand optimization techniques",
                "Apply transfer learning",
            ],
            prerequisites=["Intro to Neural Networks"],
            tags=["deep-learning", "intermediate", "architectures"],
        )
        return course
    
    @staticmethod
    def advanced_nlp() -> Course:
        """Create an Advanced NLP course."""
        course = Course(
            course_id="template-advanced-nlp",
            name="Advanced Natural Language Processing",
            description="State-of-the-art NLP with transformers and attention",
            teacher_id="template",
            difficulty=DifficultyLevel.ADVANCED,
            duration_weeks=10,
            credits=4.0,
            learning_outcomes=[
                "Implement transformer architectures",
                "Build attention mechanisms",
                "Fine-tune pre-trained models",
                "Deploy NLP applications",
            ],
            prerequisites=["Deep Learning Fundamentals"],
            tags=["nlp", "transformers", "advanced"],
        )
        return course
    
    @staticmethod
    def computer_vision() -> Course:
        """Create a Computer Vision course."""
        course = Course(
            course_id="template-computer-vision",
            name="Computer Vision with Deep Learning",
            description="Image classification, object detection, and segmentation",
            teacher_id="template",
            difficulty=DifficultyLevel.INTERMEDIATE,
            duration_weeks=10,
            credits=3.5,
            learning_outcomes=[
                "Build CNN architectures",
                "Implement object detection models",
                "Create semantic segmentation networks",
                "Apply data augmentation techniques",
            ],
            prerequisites=["Intro to Neural Networks"],
            tags=["computer-vision", "cnn", "intermediate"],
        )
        return course
    
    @staticmethod
    def create_lessons_for_intro_course() -> List[Lesson]:
        """Create lessons for the intro course."""
        lessons = []
        
        lessons.append(Lesson(
            lesson_id="intro-lesson-1",
            title="Getting Started with Neural DSL",
            description="Introduction to Neural DSL syntax and basic concepts",
            content="Learn the fundamentals of Neural DSL...",
            difficulty=DifficultyLevel.BEGINNER,
            estimated_time=60,
            objectives=[
                "Understand DSL syntax",
                "Define your first network",
                "Compile to Python code",
            ],
            order=1,
        ))
        
        lessons.append(Lesson(
            lesson_id="intro-lesson-2",
            title="Building Feedforward Networks",
            description="Create simple feedforward neural networks",
            content="Feedforward networks are the foundation...",
            difficulty=DifficultyLevel.BEGINNER,
            estimated_time=90,
            objectives=[
                "Define layer types",
                "Connect layers sequentially",
                "Understand activation functions",
            ],
            prerequisites=["intro-lesson-1"],
            order=2,
        ))
        
        lessons.append(Lesson(
            lesson_id="intro-lesson-3",
            title="Introduction to CNNs",
            description="Convolutional neural networks for image data",
            content="CNNs are designed for processing images...",
            difficulty=DifficultyLevel.INTERMEDIATE,
            estimated_time=120,
            objectives=[
                "Understand convolution operations",
                "Implement pooling layers",
                "Build image classifiers",
            ],
            prerequisites=["intro-lesson-2"],
            order=3,
        ))
        
        return lessons
