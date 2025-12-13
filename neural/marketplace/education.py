"""
Educational Resources - Supporting universities and learning institutions.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class EducationalResources:
    """Educational resources and academic program support."""

    def __init__(self, data_dir: str = "educational_resources"):
        """Initialize educational resources.

        Parameters
        ----------
        data_dir : str
            Directory to store educational data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)

        self.courses_file = self.data_dir / "courses.json"
        self.assignments_file = self.data_dir / "assignments.json"
        self.universities_file = self.data_dir / "universities.json"
        self.tutorials_file = self.data_dir / "tutorials.json"

        self._load_data()

    def _load_data(self):
        """Load all educational data."""
        self.courses = self._load_json(self.courses_file, [])
        self.assignments = self._load_json(self.assignments_file, [])
        self.universities = self._load_json(self.universities_file, {})
        self.tutorials = self._load_json(self.tutorials_file, [])

    def _load_json(self, file_path: Path, default: Any) -> Any:
        """Load JSON from file."""
        if file_path.exists():
            with open(file_path, 'r') as f:
                return json.load(f)
        return default

    def _save_json(self, file_path: Path, data: Any):
        """Save JSON to file."""
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

    def add_course_material(
        self,
        title: str,
        instructor: str,
        university: str,
        description: str,
        level: str,
        topics: List[str],
        resources: Optional[Dict[str, str]] = None
    ) -> str:
        """Add course material.

        Parameters
        ----------
        title : str
            Course title
        instructor : str
            Instructor name
        university : str
            University name
        description : str
            Course description
        level : str
            Level: beginner, intermediate, advanced
        topics : List[str]
            Course topics
        resources : Dict, optional
            Links to syllabus, slides, code, etc.

        Returns
        -------
        str
            Course ID
        """
        course_id = f"course_{len(self.courses) + 1}"

        course = {
            "id": course_id,
            "title": title,
            "instructor": instructor,
            "university": university,
            "description": description,
            "level": level,
            "topics": topics,
            "resources": resources or {},
            "created_at": datetime.now().isoformat(),
            "students_enrolled": 0
        }

        self.courses.append(course)
        self._save_json(self.courses_file, self.courses)

        self._track_university(university)

        return course_id

    def add_assignment_template(
        self,
        title: str,
        description: str,
        difficulty: str,
        learning_objectives: List[str],
        starter_code: Optional[str] = None,
        solution: Optional[str] = None,
        grading_rubric: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add assignment template for educators.

        Parameters
        ----------
        title : str
            Assignment title
        description : str
            Assignment description
        difficulty : str
            Difficulty: easy, medium, hard
        learning_objectives : List[str]
            Learning objectives
        starter_code : str, optional
            Starter code template
        solution : str, optional
            Solution code (optional)
        grading_rubric : Dict, optional
            Grading criteria

        Returns
        -------
        str
            Assignment ID
        """
        assignment_id = f"assignment_{len(self.assignments) + 1}"

        assignment = {
            "id": assignment_id,
            "title": title,
            "description": description,
            "difficulty": difficulty,
            "learning_objectives": learning_objectives,
            "starter_code": starter_code,
            "solution": solution,
            "grading_rubric": grading_rubric or {},
            "created_at": datetime.now().isoformat(),
            "downloads": 0
        }

        self.assignments.append(assignment)
        self._save_json(self.assignments_file, self.assignments)

        return assignment_id

    def add_tutorial(
        self,
        title: str,
        author: str,
        description: str,
        difficulty: str,
        duration_minutes: int,
        content: str,
        code_examples: Optional[List[str]] = None,
        prerequisites: Optional[List[str]] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """Add tutorial.

        Parameters
        ----------
        title : str
            Tutorial title
        author : str
            Author name
        description : str
            Tutorial description
        difficulty : str
            Difficulty level
        duration_minutes : int
            Estimated duration
        content : str
            Tutorial content (markdown)
        code_examples : List[str], optional
            Code examples
        prerequisites : List[str], optional
            Prerequisites
        tags : List[str], optional
            Tags

        Returns
        -------
        str
            Tutorial ID
        """
        tutorial_id = f"tutorial_{len(self.tutorials) + 1}"

        tutorial = {
            "id": tutorial_id,
            "title": title,
            "author": author,
            "description": description,
            "difficulty": difficulty,
            "duration_minutes": duration_minutes,
            "content": content,
            "code_examples": code_examples or [],
            "prerequisites": prerequisites or [],
            "tags": tags or [],
            "created_at": datetime.now().isoformat(),
            "views": 0,
            "completed": 0
        }

        self.tutorials.append(tutorial)
        self._save_json(self.tutorials_file, self.tutorials)

        return tutorial_id

    def _track_university(self, university: str):
        """Track university usage."""
        if university not in self.universities:
            self.universities[university] = {
                "name": university,
                "courses": 0,
                "students": 0,
                "joined_at": datetime.now().isoformat()
            }

        self.universities[university]["courses"] += 1
        self._save_json(self.universities_file, self.universities)

    def register_university(
        self,
        name: str,
        department: str,
        contact_email: str,
        website: Optional[str] = None
    ) -> bool:
        """Register a university for academic program.

        Parameters
        ----------
        name : str
            University name
        department : str
            Department name
        contact_email : str
            Contact email
        website : str, optional
            University website

        Returns
        -------
        bool
            True if successful
        """
        if name in self.universities:
            return False

        self.universities[name] = {
            "name": name,
            "department": department,
            "contact_email": contact_email,
            "website": website,
            "courses": 0,
            "students": 0,
            "joined_at": datetime.now().isoformat(),
            "status": "active"
        }

        self._save_json(self.universities_file, self.universities)
        return True

    def get_courses(
        self,
        level: Optional[str] = None,
        topic: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get course materials.

        Parameters
        ----------
        level : str, optional
            Filter by level
        topic : str, optional
            Filter by topic
        limit : int
            Maximum results

        Returns
        -------
        List[Dict]
            List of courses
        """
        courses = self.courses

        if level:
            courses = [c for c in courses if c["level"] == level]

        if topic:
            courses = [c for c in courses if topic in c["topics"]]

        return courses[:limit]

    def get_assignments(
        self,
        difficulty: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get assignment templates.

        Parameters
        ----------
        difficulty : str, optional
            Filter by difficulty
        limit : int
            Maximum results

        Returns
        -------
        List[Dict]
            List of assignments
        """
        assignments = self.assignments

        if difficulty:
            assignments = [a for a in assignments if a["difficulty"] == difficulty]

        assignments.sort(key=lambda a: a["downloads"], reverse=True)

        return assignments[:limit]

    def get_tutorials(
        self,
        difficulty: Optional[str] = None,
        tag: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get tutorials.

        Parameters
        ----------
        difficulty : str, optional
            Filter by difficulty
        tag : str, optional
            Filter by tag
        limit : int
            Maximum results

        Returns
        -------
        List[Dict]
            List of tutorials
        """
        tutorials = self.tutorials

        if difficulty:
            tutorials = [t for t in tutorials if t["difficulty"] == difficulty]

        if tag:
            tutorials = [t for t in tutorials if tag in t["tags"]]

        tutorials.sort(key=lambda t: t["views"], reverse=True)

        return tutorials[:limit]

    def get_university_stats(self) -> Dict[str, Any]:
        """Get statistics about university adoption.

        Returns
        -------
        Dict
            Statistics
        """
        total_universities = len(self.universities)
        total_students = sum(u.get("students", 0) for u in self.universities.values())
        total_courses = sum(u.get("courses", 0) for u in self.universities.values())

        return {
            "total_universities": total_universities,
            "total_students": total_students,
            "total_courses": total_courses,
            "top_universities": self._get_top_universities(10)
        }

    def _get_top_universities(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top universities by usage."""
        universities = list(self.universities.values())
        universities.sort(key=lambda u: u.get("students", 0), reverse=True)

        return universities[:limit]

    def create_learning_path(
        self,
        title: str,
        description: str,
        modules: List[Dict[str, Any]],
        estimated_hours: int,
        target_audience: str
    ) -> str:
        """Create a structured learning path.

        Parameters
        ----------
        title : str
            Learning path title
        description : str
            Description
        modules : List[Dict]
            List of modules with tutorials/assignments
        estimated_hours : int
            Estimated completion time
        target_audience : str
            Target audience

        Returns
        -------
        str
            Learning path ID
        """
        learning_path_file = self.data_dir / "learning_paths.json"
        learning_paths = self._load_json(learning_path_file, [])

        path_id = f"path_{len(learning_paths) + 1}"

        learning_path = {
            "id": path_id,
            "title": title,
            "description": description,
            "modules": modules,
            "estimated_hours": estimated_hours,
            "target_audience": target_audience,
            "created_at": datetime.now().isoformat(),
            "enrollments": 0
        }

        learning_paths.append(learning_path)
        self._save_json(learning_path_file, learning_paths)

        return path_id


class UniversityLicenseManager:
    """Manage university licenses and academic programs."""

    def __init__(self, data_dir: str = "university_licenses"):
        """Initialize license manager.

        Parameters
        ----------
        data_dir : str
            Directory to store license data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)

        self.licenses_file = self.data_dir / "licenses.json"
        self._load_licenses()

    def _load_licenses(self):
        """Load licenses from disk."""
        if self.licenses_file.exists():
            with open(self.licenses_file, 'r') as f:
                self.licenses = json.load(f)
        else:
            self.licenses = {}
            self._save_licenses()

    def _save_licenses(self):
        """Save licenses to disk."""
        with open(self.licenses_file, 'w') as f:
            json.dump(self.licenses, f, indent=2)

    def issue_academic_license(
        self,
        university: str,
        department: str,
        instructor: str,
        email: str,
        student_count: int,
        duration_months: int = 12
    ) -> str:
        """Issue an academic license.

        Parameters
        ----------
        university : str
            University name
        department : str
            Department name
        instructor : str
            Instructor name
        email : str
            Contact email
        student_count : int
            Number of students
        duration_months : int
            License duration

        Returns
        -------
        str
            License key
        """
        from datetime import timedelta
        import hashlib

        license_key = hashlib.sha256(
            f"{university}{department}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16].upper()

        expiry_date = datetime.now() + timedelta(days=duration_months * 30)

        self.licenses[license_key] = {
            "university": university,
            "department": department,
            "instructor": instructor,
            "email": email,
            "student_count": student_count,
            "issued_at": datetime.now().isoformat(),
            "expires_at": expiry_date.isoformat(),
            "status": "active"
        }

        self._save_licenses()
        return license_key

    def verify_license(self, license_key: str) -> bool:
        """Verify if a license is valid.

        Parameters
        ----------
        license_key : str
            License key

        Returns
        -------
        bool
            True if valid
        """
        if license_key not in self.licenses:
            return False

        license_data = self.licenses[license_key]

        if license_data["status"] != "active":
            return False

        try:
            expiry = datetime.fromisoformat(license_data["expires_at"])
            if expiry < datetime.now():
                license_data["status"] = "expired"
                self._save_licenses()
                return False
        except (ValueError, KeyError):
            return False

        return True

    def get_license_info(self, license_key: str) -> Optional[Dict[str, Any]]:
        """Get license information.

        Parameters
        ----------
        license_key : str
            License key

        Returns
        -------
        Dict, optional
            License information
        """
        return self.licenses.get(license_key)
