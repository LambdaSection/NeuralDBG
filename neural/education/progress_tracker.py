"""
Progress tracking and gamification system.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import BadgeType, StudentProfile


@dataclass
class Achievement:
    """An achievement or milestone."""
    achievement_id: str
    name: str
    description: str
    xp_reward: int
    badge_type: BadgeType
    criteria: Dict[str, Any]
    icon: str = "🏆"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "achievement_id": self.achievement_id,
            "name": self.name,
            "description": self.description,
            "xp_reward": self.xp_reward,
            "badge_type": self.badge_type.value,
            "criteria": self.criteria,
            "icon": self.icon,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Achievement:
        """Create from dictionary."""
        data = data.copy()
        data["badge_type"] = BadgeType(data["badge_type"])
        return cls(**data)


@dataclass
class Badge:
    """A badge earned by a student."""
    badge_id: str
    student_id: str
    achievement_id: str
    earned_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "badge_id": self.badge_id,
            "student_id": self.student_id,
            "achievement_id": self.achievement_id,
            "earned_at": self.earned_at.isoformat(),
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Badge:
        """Create from dictionary."""
        data = data.copy()
        data["earned_at"] = datetime.fromisoformat(data["earned_at"])
        return cls(**data)


class ProgressTracker:
    """Tracks student progress and manages gamification."""
    
    def __init__(self, storage_dir: str = "neural_education_progress"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.students: Dict[str, StudentProfile] = {}
        self.achievements: Dict[str, Achievement] = {}
        self.badges: Dict[str, List[Badge]] = {}
        self._load_data()
        self._initialize_achievements()
    
    def _load_data(self) -> None:
        """Load progress data from storage."""
        students_file = self.storage_dir / "students.json"
        if students_file.exists():
            with open(students_file, 'r') as f:
                data = json.load(f)
                self.students = {
                    sid: StudentProfile.from_dict(sdata)
                    for sid, sdata in data.items()
                }
        
        badges_file = self.storage_dir / "badges.json"
        if badges_file.exists():
            with open(badges_file, 'r') as f:
                data = json.load(f)
                self.badges = {
                    sid: [Badge.from_dict(b) for b in badges]
                    for sid, badges in data.items()
                }
    
    def _save_data(self) -> None:
        """Save progress data to storage."""
        with open(self.storage_dir / "students.json", 'w') as f:
            json.dump(
                {sid: s.to_dict() for sid, s in self.students.items()},
                f,
                indent=2,
            )
        
        with open(self.storage_dir / "badges.json", 'w') as f:
            json.dump(
                {sid: [b.to_dict() for b in badges] for sid, badges in self.badges.items()},
                f,
                indent=2,
            )
    
    def _initialize_achievements(self) -> None:
        """Initialize default achievements."""
        self.achievements = {
            "first_tutorial": Achievement(
                achievement_id="first_tutorial",
                name="First Steps",
                description="Complete your first tutorial",
                xp_reward=100,
                badge_type=BadgeType.COMPLETION,
                criteria={"completed_tutorials": 1},
                icon="🎓",
            ),
            "tutorial_master": Achievement(
                achievement_id="tutorial_master",
                name="Tutorial Master",
                description="Complete 10 tutorials",
                xp_reward=500,
                badge_type=BadgeType.MASTERY,
                criteria={"completed_tutorials": 10},
                icon="🏅",
            ),
            "first_assignment": Achievement(
                achievement_id="first_assignment",
                name="Assignment Complete",
                description="Submit your first assignment",
                xp_reward=200,
                badge_type=BadgeType.COMPLETION,
                criteria={"completed_assignments": 1},
                icon="📝",
            ),
            "perfect_score": Achievement(
                achievement_id="perfect_score",
                name="Perfect Score",
                description="Get 100% on an assignment",
                xp_reward=300,
                badge_type=BadgeType.ACCURACY,
                criteria={"perfect_score": True},
                icon="💯",
            ),
            "speed_learner": Achievement(
                achievement_id="speed_learner",
                name="Speed Learner",
                description="Complete a tutorial in under 15 minutes",
                xp_reward=150,
                badge_type=BadgeType.SPEED,
                criteria={"completion_time": 900},
                icon="⚡",
            ),
            "consistent": Achievement(
                achievement_id="consistent",
                name="Consistent Learner",
                description="Complete tutorials 7 days in a row",
                xp_reward=400,
                badge_type=BadgeType.CONSISTENCY,
                criteria={"streak_days": 7},
                icon="🔥",
            ),
        }
    
    def register_student(
        self,
        student_id: str,
        name: str,
        email: str,
    ) -> StudentProfile:
        """Register a new student."""
        if student_id in self.students:
            return self.students[student_id]
        
        student = StudentProfile(
            student_id=student_id,
            name=name,
            email=email,
        )
        self.students[student_id] = student
        self.badges[student_id] = []
        self._save_data()
        return student
    
    def get_student(self, student_id: str) -> Optional[StudentProfile]:
        """Get a student profile."""
        return self.students.get(student_id)
    
    def award_xp(self, student_id: str, xp: int) -> None:
        """Award XP to a student."""
        student = self.students.get(student_id)
        if not student:
            return
        
        student.total_xp += xp
        new_level = 1 + (student.total_xp // 1000)
        
        if new_level > student.level:
            student.level = new_level
        
        self._save_data()
    
    def complete_tutorial(
        self,
        student_id: str,
        tutorial_id: str,
        completion_time: Optional[int] = None,
    ) -> List[Badge]:
        """Mark a tutorial as complete and check for achievements."""
        student = self.students.get(student_id)
        if not student or tutorial_id in student.completed_tutorials:
            return []
        
        student.completed_tutorials.append(tutorial_id)
        new_badges = []
        
        if len(student.completed_tutorials) == 1:
            badge = self._award_badge(student_id, "first_tutorial")
            if badge:
                new_badges.append(badge)
        
        if len(student.completed_tutorials) >= 10:
            badge = self._award_badge(student_id, "tutorial_master")
            if badge:
                new_badges.append(badge)
        
        if completion_time and completion_time <= 900:
            badge = self._award_badge(student_id, "speed_learner")
            if badge:
                new_badges.append(badge)
        
        self._save_data()
        return new_badges
    
    def complete_assignment(
        self,
        student_id: str,
        assignment_id: str,
        score: float,
    ) -> List[Badge]:
        """Mark an assignment as complete and check for achievements."""
        student = self.students.get(student_id)
        if not student or assignment_id in student.completed_assignments:
            return []
        
        student.completed_assignments.append(assignment_id)
        new_badges = []
        
        if len(student.completed_assignments) == 1:
            badge = self._award_badge(student_id, "first_assignment")
            if badge:
                new_badges.append(badge)
        
        if score >= 100.0:
            badge = self._award_badge(student_id, "perfect_score")
            if badge:
                new_badges.append(badge)
        
        self._save_data()
        return new_badges
    
    def _award_badge(
        self,
        student_id: str,
        achievement_id: str,
    ) -> Optional[Badge]:
        """Award a badge to a student."""
        student = self.students.get(student_id)
        achievement = self.achievements.get(achievement_id)
        
        if not student or not achievement:
            return None
        
        existing_badges = self.badges.get(student_id, [])
        if any(b.achievement_id == achievement_id for b in existing_badges):
            return None
        
        badge = Badge(
            badge_id=str(uuid.uuid4()),
            student_id=student_id,
            achievement_id=achievement_id,
            earned_at=datetime.now(),
        )
        
        self.badges[student_id].append(badge)
        student.badges.append(badge.badge_id)
        self.award_xp(student_id, achievement.xp_reward)
        
        return badge
    
    def get_student_badges(self, student_id: str) -> List[Badge]:
        """Get all badges earned by a student."""
        return self.badges.get(student_id, [])
    
    def get_leaderboard(self, limit: int = 10) -> List[StudentProfile]:
        """Get top students by XP."""
        students = sorted(
            self.students.values(),
            key=lambda s: s.total_xp,
            reverse=True,
        )
        return students[:limit]
    
    def get_progress_stats(self, student_id: str) -> Dict[str, Any]:
        """Get detailed progress statistics for a student."""
        student = self.students.get(student_id)
        if not student:
            return {}
        
        badges = self.get_student_badges(student_id)
        
        return {
            "student_id": student_id,
            "name": student.name,
            "level": student.level,
            "total_xp": student.total_xp,
            "xp_to_next_level": 1000 - (student.total_xp % 1000),
            "tutorials_completed": len(student.completed_tutorials),
            "assignments_completed": len(student.completed_assignments),
            "badges_earned": len(badges),
            "courses_enrolled": len(student.enrolled_courses),
            "badge_breakdown": {
                badge_type.value: len([
                    b for b in badges
                    if self.achievements.get(b.achievement_id, Achievement(
                        "", "", "", 0, badge_type, {}
                    )).badge_type == badge_type
                ])
                for badge_type in BadgeType
            },
        }
