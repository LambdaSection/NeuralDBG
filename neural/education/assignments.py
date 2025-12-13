"""
Assignment creation and submission system.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import SubmissionStatus


@dataclass
class Assignment:
    """An assignment for students to complete."""
    assignment_id: str
    title: str
    description: str
    course_id: str
    created_by: str
    created_at: datetime = field(default_factory=datetime.now)
    due_date: Optional[datetime] = None
    points: int = 100
    requirements: Dict[str, Any] = field(default_factory=dict)
    starter_code: str = ""
    test_cases: List[Dict[str, Any]] = field(default_factory=list)
    rubric: Dict[str, int] = field(default_factory=dict)
    allow_late: bool = True
    late_penalty: float = 0.1
    max_attempts: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "assignment_id": self.assignment_id,
            "title": self.title,
            "description": self.description,
            "course_id": self.course_id,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "points": self.points,
            "requirements": self.requirements,
            "starter_code": self.starter_code,
            "test_cases": self.test_cases,
            "rubric": self.rubric,
            "allow_late": self.allow_late,
            "late_penalty": self.late_penalty,
            "max_attempts": self.max_attempts,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Assignment:
        """Create from dictionary."""
        data = data.copy()
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        if data.get("due_date"):
            data["due_date"] = datetime.fromisoformat(data["due_date"])
        return cls(**data)
    
    def is_late(self, submission_time: datetime) -> bool:
        """Check if a submission is late."""
        if not self.due_date:
            return False
        return submission_time > self.due_date
    
    def calculate_late_penalty(self, submission_time: datetime) -> float:
        """Calculate late penalty multiplier."""
        if not self.is_late(submission_time):
            return 1.0
        if not self.allow_late:
            return 0.0
        
        hours_late = (submission_time - self.due_date).total_seconds() / 3600
        days_late = int(hours_late / 24) + 1
        penalty = max(0.0, 1.0 - (self.late_penalty * days_late))
        return penalty


@dataclass
class AssignmentSubmission:
    """A student's submission for an assignment."""
    submission_id: str
    assignment_id: str
    student_id: str
    submitted_at: datetime
    status: SubmissionStatus = SubmissionStatus.SUBMITTED
    code: str = ""
    files: Dict[str, str] = field(default_factory=dict)
    score: Optional[float] = None
    feedback: str = ""
    graded_at: Optional[datetime] = None
    graded_by: Optional[str] = None
    attempt_number: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "submission_id": self.submission_id,
            "assignment_id": self.assignment_id,
            "student_id": self.student_id,
            "submitted_at": self.submitted_at.isoformat(),
            "status": self.status.value,
            "code": self.code,
            "files": self.files,
            "score": self.score,
            "feedback": self.feedback,
            "graded_at": self.graded_at.isoformat() if self.graded_at else None,
            "graded_by": self.graded_by,
            "attempt_number": self.attempt_number,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AssignmentSubmission:
        """Create from dictionary."""
        data = data.copy()
        data["submitted_at"] = datetime.fromisoformat(data["submitted_at"])
        data["status"] = SubmissionStatus(data["status"])
        if data.get("graded_at"):
            data["graded_at"] = datetime.fromisoformat(data["graded_at"])
        return cls(**data)


class AssignmentManager:
    """Manages assignments and submissions."""
    
    def __init__(self, storage_dir: str = "neural_education_assignments"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.assignments: Dict[str, Assignment] = {}
        self.submissions: Dict[str, List[AssignmentSubmission]] = {}
        self._load_data()
    
    def _load_data(self) -> None:
        """Load assignments and submissions from storage."""
        assignments_file = self.storage_dir / "assignments.json"
        if assignments_file.exists():
            with open(assignments_file, 'r') as f:
                data = json.load(f)
                self.assignments = {
                    aid: Assignment.from_dict(adata)
                    for aid, adata in data.items()
                }
        
        submissions_file = self.storage_dir / "submissions.json"
        if submissions_file.exists():
            with open(submissions_file, 'r') as f:
                data = json.load(f)
                self.submissions = {
                    aid: [AssignmentSubmission.from_dict(s) for s in subs]
                    for aid, subs in data.items()
                }
    
    def _save_data(self) -> None:
        """Save assignments and submissions to storage."""
        with open(self.storage_dir / "assignments.json", 'w') as f:
            json.dump(
                {aid: a.to_dict() for aid, a in self.assignments.items()},
                f,
                indent=2,
            )
        
        with open(self.storage_dir / "submissions.json", 'w') as f:
            json.dump(
                {
                    aid: [s.to_dict() for s in subs]
                    for aid, subs in self.submissions.items()
                },
                f,
                indent=2,
            )
    
    def create_assignment(
        self,
        title: str,
        description: str,
        course_id: str,
        created_by: str,
        **kwargs,
    ) -> Assignment:
        """Create a new assignment."""
        assignment = Assignment(
            assignment_id=str(uuid.uuid4()),
            title=title,
            description=description,
            course_id=course_id,
            created_by=created_by,
            **kwargs,
        )
        self.assignments[assignment.assignment_id] = assignment
        self.submissions[assignment.assignment_id] = []
        self._save_data()
        return assignment
    
    def get_assignment(self, assignment_id: str) -> Optional[Assignment]:
        """Get an assignment by ID."""
        return self.assignments.get(assignment_id)
    
    def list_assignments(
        self,
        course_id: Optional[str] = None,
        created_by: Optional[str] = None,
    ) -> List[Assignment]:
        """List assignments with optional filtering."""
        assignments = list(self.assignments.values())
        if course_id:
            assignments = [a for a in assignments if a.course_id == course_id]
        if created_by:
            assignments = [a for a in assignments if a.created_by == created_by]
        return sorted(assignments, key=lambda a: a.created_at, reverse=True)
    
    def submit_assignment(
        self,
        assignment_id: str,
        student_id: str,
        code: str,
        files: Optional[Dict[str, str]] = None,
    ) -> Optional[AssignmentSubmission]:
        """Submit an assignment."""
        assignment = self.assignments.get(assignment_id)
        if not assignment:
            return None
        
        existing_submissions = self.get_student_submissions(assignment_id, student_id)
        attempt_number = len(existing_submissions) + 1
        
        if assignment.max_attempts > 0 and attempt_number > assignment.max_attempts:
            return None
        
        submission = AssignmentSubmission(
            submission_id=str(uuid.uuid4()),
            assignment_id=assignment_id,
            student_id=student_id,
            submitted_at=datetime.now(),
            code=code,
            files=files or {},
            attempt_number=attempt_number,
        )
        
        self.submissions[assignment_id].append(submission)
        self._save_data()
        return submission
    
    def get_submission(self, submission_id: str) -> Optional[AssignmentSubmission]:
        """Get a submission by ID."""
        for submissions in self.submissions.values():
            for submission in submissions:
                if submission.submission_id == submission_id:
                    return submission
        return None
    
    def get_student_submissions(
        self,
        assignment_id: str,
        student_id: str,
    ) -> List[AssignmentSubmission]:
        """Get all submissions by a student for an assignment."""
        submissions = self.submissions.get(assignment_id, [])
        return [s for s in submissions if s.student_id == student_id]
    
    def get_assignment_submissions(
        self,
        assignment_id: str,
        status: Optional[SubmissionStatus] = None,
    ) -> List[AssignmentSubmission]:
        """Get all submissions for an assignment."""
        submissions = self.submissions.get(assignment_id, [])
        if status:
            submissions = [s for s in submissions if s.status == status]
        return sorted(submissions, key=lambda s: s.submitted_at, reverse=True)
    
    def grade_submission(
        self,
        submission_id: str,
        score: float,
        feedback: str,
        graded_by: str,
    ) -> Optional[AssignmentSubmission]:
        """Grade a submission."""
        submission = self.get_submission(submission_id)
        if not submission:
            return None
        
        assignment = self.assignments.get(submission.assignment_id)
        if assignment:
            late_penalty = assignment.calculate_late_penalty(submission.submitted_at)
            score = score * late_penalty
        
        submission.score = score
        submission.feedback = feedback
        submission.graded_at = datetime.now()
        submission.graded_by = graded_by
        submission.status = SubmissionStatus.GRADED
        
        self._save_data()
        return submission
    
    def get_course_statistics(self, course_id: str) -> Dict[str, Any]:
        """Get statistics for assignments in a course."""
        assignments = self.list_assignments(course_id=course_id)
        
        total_submissions = 0
        graded_submissions = 0
        total_score = 0.0
        
        for assignment in assignments:
            subs = self.submissions.get(assignment.assignment_id, [])
            total_submissions += len(subs)
            for sub in subs:
                if sub.score is not None:
                    graded_submissions += 1
                    total_score += sub.score
        
        return {
            "course_id": course_id,
            "total_assignments": len(assignments),
            "total_submissions": total_submissions,
            "graded_submissions": graded_submissions,
            "pending_submissions": total_submissions - graded_submissions,
            "average_score": total_score / graded_submissions if graded_submissions > 0 else 0.0,
        }
    
    def get_student_statistics(
        self,
        student_id: str,
        course_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get statistics for a student."""
        assignments = self.list_assignments(course_id=course_id) if course_id else list(self.assignments.values())
        
        total_assignments = len(assignments)
        submitted = 0
        graded = 0
        total_score = 0.0
        
        for assignment in assignments:
            subs = self.get_student_submissions(assignment.assignment_id, student_id)
            if subs:
                submitted += 1
                latest = max(subs, key=lambda s: s.submitted_at)
                if latest.score is not None:
                    graded += 1
                    total_score += latest.score
        
        return {
            "student_id": student_id,
            "course_id": course_id,
            "total_assignments": total_assignments,
            "submitted": submitted,
            "graded": graded,
            "pending": submitted - graded,
            "average_score": total_score / graded if graded > 0 else 0.0,
            "completion_rate": submitted / total_assignments if total_assignments > 0 else 0.0,
        }
