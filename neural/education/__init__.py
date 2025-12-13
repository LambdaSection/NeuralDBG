"""
Education module for Neural DSL.

Provides interactive tutorials, progress tracking, gamification,
assignment systems, automated grading, curriculum templates,
LMS integration, and teacher dashboards.
"""

from __future__ import annotations

from .notebook_tutorials import NotebookTutorial, TutorialLibrary
from .progress_tracker import ProgressTracker, Achievement, Badge
from .assignments import Assignment, AssignmentSubmission, AssignmentManager
from .grading import AutoGrader, GradingCriteria, GradingResult
from .curriculum import Curriculum, Course, Lesson, CurriculumTemplate
from .lms_integration import LMSConnector, CanvasLMS, MoodleLMS, BlackboardLMS
from .teacher_dashboard import TeacherDashboard

__all__ = [
    "NotebookTutorial",
    "TutorialLibrary",
    "ProgressTracker",
    "Achievement",
    "Badge",
    "Assignment",
    "AssignmentSubmission",
    "AssignmentManager",
    "AutoGrader",
    "GradingCriteria",
    "GradingResult",
    "Curriculum",
    "Course",
    "Lesson",
    "CurriculumTemplate",
    "LMSConnector",
    "CanvasLMS",
    "MoodleLMS",
    "BlackboardLMS",
    "TeacherDashboard",
]
