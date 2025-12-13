"""
Learning Management System (LMS) integration.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class LMSConnector(ABC):
    """Base class for LMS integrations."""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = None
        if REQUESTS_AVAILABLE:
            self.session = requests.Session()
            self.session.headers.update(self._get_auth_headers())
    
    @abstractmethod
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for API requests."""
        pass
    
    @abstractmethod
    def get_courses(self) -> List[Dict[str, Any]]:
        """Fetch all courses from the LMS."""
        pass
    
    @abstractmethod
    def get_students(self, course_id: str) -> List[Dict[str, Any]]:
        """Fetch students enrolled in a course."""
        pass
    
    @abstractmethod
    def create_assignment(
        self,
        course_id: str,
        assignment_data: Dict[str, Any],
    ) -> Optional[str]:
        """Create an assignment in the LMS."""
        pass
    
    @abstractmethod
    def submit_grade(
        self,
        course_id: str,
        assignment_id: str,
        student_id: str,
        grade: float,
        feedback: Optional[str] = None,
    ) -> bool:
        """Submit a grade for a student."""
        pass
    
    @abstractmethod
    def get_submissions(
        self,
        course_id: str,
        assignment_id: str,
    ) -> List[Dict[str, Any]]:
        """Get all submissions for an assignment."""
        pass


class CanvasLMS(LMSConnector):
    """Canvas LMS integration."""
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get Canvas API authentication headers."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
    
    def get_courses(self) -> List[Dict[str, Any]]:
        """Fetch courses from Canvas."""
        if not REQUESTS_AVAILABLE or not self.session:
            return []
        
        try:
            response = self.session.get(f"{self.base_url}/api/v1/courses")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching courses: {e}")
            return []
    
    def get_students(self, course_id: str) -> List[Dict[str, Any]]:
        """Fetch students enrolled in a Canvas course."""
        if not REQUESTS_AVAILABLE or not self.session:
            return []
        
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/courses/{course_id}/users",
                params={"enrollment_type[]": "student"},
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching students: {e}")
            return []
    
    def create_assignment(
        self,
        course_id: str,
        assignment_data: Dict[str, Any],
    ) -> Optional[str]:
        """Create an assignment in Canvas."""
        if not REQUESTS_AVAILABLE or not self.session:
            return None
        
        try:
            payload = {
                "assignment": {
                    "name": assignment_data.get("title", "Neural DSL Assignment"),
                    "description": assignment_data.get("description", ""),
                    "due_at": assignment_data.get("due_date"),
                    "points_possible": assignment_data.get("points", 100),
                    "submission_types": ["online_text_entry", "online_upload"],
                }
            }
            
            response = self.session.post(
                f"{self.base_url}/api/v1/courses/{course_id}/assignments",
                json=payload,
            )
            response.raise_for_status()
            result = response.json()
            return str(result.get("id"))
        except Exception as e:
            print(f"Error creating assignment: {e}")
            return None
    
    def submit_grade(
        self,
        course_id: str,
        assignment_id: str,
        student_id: str,
        grade: float,
        feedback: Optional[str] = None,
    ) -> bool:
        """Submit a grade to Canvas."""
        if not REQUESTS_AVAILABLE or not self.session:
            return False
        
        try:
            payload = {
                "submission": {
                    "posted_grade": grade,
                }
            }
            
            if feedback:
                payload["comment"] = {
                    "text_comment": feedback,
                }
            
            response = self.session.put(
                f"{self.base_url}/api/v1/courses/{course_id}/assignments/{assignment_id}/submissions/{student_id}",
                json=payload,
            )
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"Error submitting grade: {e}")
            return False
    
    def get_submissions(
        self,
        course_id: str,
        assignment_id: str,
    ) -> List[Dict[str, Any]]:
        """Get submissions for a Canvas assignment."""
        if not REQUESTS_AVAILABLE or not self.session:
            return []
        
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/courses/{course_id}/assignments/{assignment_id}/submissions",
                params={"include[]": ["user", "submission_comments"]},
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching submissions: {e}")
            return []


class MoodleLMS(LMSConnector):
    """Moodle LMS integration."""
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get Moodle API authentication headers."""
        return {
            "Content-Type": "application/json",
        }
    
    def _call_api(self, function: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Call Moodle web service API."""
        if not REQUESTS_AVAILABLE or not self.session:
            return None
        
        try:
            data = {
                "wstoken": self.api_key,
                "wsfunction": function,
                "moodlewsrestformat": "json",
            }
            if params:
                data.update(params)
            
            response = self.session.post(
                f"{self.base_url}/webservice/rest/server.php",
                data=data,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error calling Moodle API: {e}")
            return None
    
    def get_courses(self) -> List[Dict[str, Any]]:
        """Fetch courses from Moodle."""
        result = self._call_api("core_course_get_courses")
        return result if isinstance(result, list) else []
    
    def get_students(self, course_id: str) -> List[Dict[str, Any]]:
        """Fetch students enrolled in a Moodle course."""
        result = self._call_api(
            "core_enrol_get_enrolled_users",
            {"courseid": course_id},
        )
        return result if isinstance(result, list) else []
    
    def create_assignment(
        self,
        course_id: str,
        assignment_data: Dict[str, Any],
    ) -> Optional[str]:
        """Create an assignment in Moodle."""
        params = {
            "courseid": course_id,
            "assignments[0][name]": assignment_data.get("title", "Neural DSL Assignment"),
            "assignments[0][intro]": assignment_data.get("description", ""),
            "assignments[0][duedate]": int(
                datetime.fromisoformat(assignment_data.get("due_date", "")).timestamp()
            ) if assignment_data.get("due_date") else 0,
            "assignments[0][grade]": assignment_data.get("points", 100),
        }
        
        result = self._call_api("mod_assign_create_assignments", params)
        
        if result and isinstance(result, list) and len(result) > 0:
            return str(result[0].get("id"))
        return None
    
    def submit_grade(
        self,
        course_id: str,
        assignment_id: str,
        student_id: str,
        grade: float,
        feedback: Optional[str] = None,
    ) -> bool:
        """Submit a grade to Moodle."""
        params = {
            "assignmentid": assignment_id,
            "userid": student_id,
            "grade": grade,
            "attemptnumber": -1,
        }
        
        if feedback:
            params["plugindata[assignfeedbackcomments_editor][text]"] = feedback
            params["plugindata[assignfeedbackcomments_editor][format]"] = 1
        
        result = self._call_api("mod_assign_save_grade", params)
        return result is not None
    
    def get_submissions(
        self,
        course_id: str,
        assignment_id: str,
    ) -> List[Dict[str, Any]]:
        """Get submissions for a Moodle assignment."""
        result = self._call_api(
            "mod_assign_get_submissions",
            {"assignmentids[]": assignment_id},
        )
        
        if result and isinstance(result, dict):
            return result.get("assignments", [{}])[0].get("submissions", [])
        return []


class BlackboardLMS(LMSConnector):
    """Blackboard Learn LMS integration."""
    
    def __init__(self, base_url: str, api_key: str, client_secret: str = ""):
        self.client_secret = client_secret
        self.access_token = None
        super().__init__(base_url, api_key)
        self._authenticate()
    
    def _authenticate(self) -> None:
        """Authenticate with Blackboard and get access token."""
        if not REQUESTS_AVAILABLE or not self.session:
            return
        
        try:
            response = self.session.post(
                f"{self.base_url}/learn/api/public/v1/oauth2/token",
                data={
                    "grant_type": "client_credentials",
                    "client_id": self.api_key,
                    "client_secret": self.client_secret,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            response.raise_for_status()
            result = response.json()
            self.access_token = result.get("access_token")
        except Exception as e:
            print(f"Error authenticating with Blackboard: {e}")
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get Blackboard API authentication headers."""
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }
    
    def get_courses(self) -> List[Dict[str, Any]]:
        """Fetch courses from Blackboard."""
        if not REQUESTS_AVAILABLE or not self.session:
            return []
        
        try:
            response = self.session.get(
                f"{self.base_url}/learn/api/public/v1/courses"
            )
            response.raise_for_status()
            result = response.json()
            return result.get("results", [])
        except Exception as e:
            print(f"Error fetching courses: {e}")
            return []
    
    def get_students(self, course_id: str) -> List[Dict[str, Any]]:
        """Fetch students enrolled in a Blackboard course."""
        if not REQUESTS_AVAILABLE or not self.session:
            return []
        
        try:
            response = self.session.get(
                f"{self.base_url}/learn/api/public/v1/courses/{course_id}/users"
            )
            response.raise_for_status()
            result = response.json()
            users = result.get("results", [])
            return [u for u in users if u.get("courseRoleId") == "Student"]
        except Exception as e:
            print(f"Error fetching students: {e}")
            return []
    
    def create_assignment(
        self,
        course_id: str,
        assignment_data: Dict[str, Any],
    ) -> Optional[str]:
        """Create an assignment in Blackboard."""
        if not REQUESTS_AVAILABLE or not self.session:
            return None
        
        try:
            payload = {
                "name": assignment_data.get("title", "Neural DSL Assignment"),
                "description": assignment_data.get("description", ""),
                "dueDate": assignment_data.get("due_date"),
                "score": {
                    "possible": assignment_data.get("points", 100),
                },
            }
            
            response = self.session.post(
                f"{self.base_url}/learn/api/public/v2/courses/{course_id}/gradebook/columns",
                json=payload,
            )
            response.raise_for_status()
            result = response.json()
            return result.get("id")
        except Exception as e:
            print(f"Error creating assignment: {e}")
            return None
    
    def submit_grade(
        self,
        course_id: str,
        assignment_id: str,
        student_id: str,
        grade: float,
        feedback: Optional[str] = None,
    ) -> bool:
        """Submit a grade to Blackboard."""
        if not REQUESTS_AVAILABLE or not self.session:
            return False
        
        try:
            payload = {
                "score": grade,
            }
            
            if feedback:
                payload["text"] = feedback
            
            response = self.session.patch(
                f"{self.base_url}/learn/api/public/v2/courses/{course_id}/gradebook/columns/{assignment_id}/users/{student_id}",
                json=payload,
            )
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"Error submitting grade: {e}")
            return False
    
    def get_submissions(
        self,
        course_id: str,
        assignment_id: str,
    ) -> List[Dict[str, Any]]:
        """Get submissions for a Blackboard assignment."""
        if not REQUESTS_AVAILABLE or not self.session:
            return []
        
        try:
            response = self.session.get(
                f"{self.base_url}/learn/api/public/v2/courses/{course_id}/gradebook/columns/{assignment_id}/attempts"
            )
            response.raise_for_status()
            result = response.json()
            return result.get("results", [])
        except Exception as e:
            print(f"Error fetching submissions: {e}")
            return []
