"""
Data models for the education module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class DifficultyLevel(Enum):
    """Difficulty levels for educational content."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class BadgeType(Enum):
    """Types of badges that can be earned."""
    COMPLETION = "completion"
    SPEED = "speed"
    ACCURACY = "accuracy"
    CREATIVITY = "creativity"
    CONSISTENCY = "consistency"
    MASTERY = "mastery"


class SubmissionStatus(Enum):
    """Status of assignment submissions."""
    DRAFT = "draft"
    SUBMITTED = "submitted"
    GRADING = "grading"
    GRADED = "graded"
    RETURNED = "returned"


@dataclass
class StudentProfile:
    """Profile for a student user."""
    student_id: str
    name: str
    email: str
    created_at: datetime = field(default_factory=datetime.now)
    total_xp: int = 0
    level: int = 1
    badges: List[str] = field(default_factory=list)
    completed_tutorials: List[str] = field(default_factory=list)
    completed_assignments: List[str] = field(default_factory=list)
    enrolled_courses: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "student_id": self.student_id,
            "name": self.name,
            "email": self.email,
            "created_at": self.created_at.isoformat(),
            "total_xp": self.total_xp,
            "level": self.level,
            "badges": self.badges,
            "completed_tutorials": self.completed_tutorials,
            "completed_assignments": self.completed_assignments,
            "enrolled_courses": self.enrolled_courses,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StudentProfile:
        """Create from dictionary."""
        data = data.copy()
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


@dataclass
class TeacherProfile:
    """Profile for a teacher user."""
    teacher_id: str
    name: str
    email: str
    created_at: datetime = field(default_factory=datetime.now)
    courses: List[str] = field(default_factory=list)
    institution: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "teacher_id": self.teacher_id,
            "name": self.name,
            "email": self.email,
            "created_at": self.created_at.isoformat(),
            "courses": self.courses,
            "institution": self.institution,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TeacherProfile:
        """Create from dictionary."""
        data = data.copy()
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


@dataclass
class ClassRoom:
    """Represents a classroom or course section."""
    classroom_id: str
    name: str
    course_id: str
    teacher_id: str
    students: List[str] = field(default_factory=list)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "classroom_id": self.classroom_id,
            "name": self.name,
            "course_id": self.course_id,
            "teacher_id": self.teacher_id,
            "students": self.students,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ClassRoom:
        """Create from dictionary."""
        data = data.copy()
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        if data.get("start_date"):
            data["start_date"] = datetime.fromisoformat(data["start_date"])
        if data.get("end_date"):
            data["end_date"] = datetime.fromisoformat(data["end_date"])
        return cls(**data)
