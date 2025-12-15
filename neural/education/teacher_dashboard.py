"""
Teacher dashboard for classroom management.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import dash
    from dash import Dash, Input, Output, State, callback, dcc, html, dash_table
    import dash_bootstrap_components as dbc
    import plotly.graph_objects as go
    import plotly.express as px
    from flask import Flask
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False

from .assignments import AssignmentManager
from .curriculum import Curriculum
from .grading import AutoGrader, GradingCriteria
from .models import ClassRoom, TeacherProfile
from .progress_tracker import ProgressTracker


class TeacherDashboard:
    """Web-based teacher dashboard for classroom management."""
    
    def __init__(
        self,
        teacher_id: str,
        storage_dir: str = "neural_education_data",
    ):
        self.teacher_id = teacher_id
        self.storage_dir = storage_dir
        
        self.progress_tracker = ProgressTracker(f"{storage_dir}/progress")
        self.assignment_manager = AssignmentManager(f"{storage_dir}/assignments")
        self.curriculum = Curriculum(f"{storage_dir}/curriculum")
        
        if not DASH_AVAILABLE:
            raise ImportError(
                "Dash is required for the teacher dashboard. "
                "Install with: pip install dash dash-bootstrap-components plotly"
            )
        
        self.server = Flask(__name__)
        self.app = dash.Dash(
            __name__,
            server=self.server,
            title="Neural DSL - Teacher Dashboard",
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            suppress_callback_exceptions=True,
        )
        
        self._setup_layout()
        self._setup_callbacks()
    
    def _setup_layout(self) -> None:
        """Setup dashboard layout."""
        self.app.layout = dbc.Container([
            dbc.NavbarSimple(
                brand="Neural DSL Teacher Dashboard",
                brand_href="#",
                color="primary",
                dark=True,
                className="mb-4",
            ),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Quick Stats"),
                        dbc.CardBody([
                            html.Div(id="quick-stats"),
                        ]),
                    ]),
                ], width=12),
            ], className="mb-4"),
            
            dbc.Tabs([
                dbc.Tab(label="Courses", tab_id="courses"),
                dbc.Tab(label="Assignments", tab_id="assignments"),
                dbc.Tab(label="Grading Queue", tab_id="grading"),
                dbc.Tab(label="Student Progress", tab_id="progress"),
                dbc.Tab(label="Analytics", tab_id="analytics"),
            ], id="tabs", active_tab="courses"),
            
            html.Div(id="tab-content", className="mt-4"),
            
            dcc.Interval(
                id="interval-component",
                interval=30*1000,
                n_intervals=0,
            ),
        ], fluid=True)
    
    def _setup_callbacks(self) -> None:
        """Setup dashboard callbacks."""
        
        @self.app.callback(
            Output("quick-stats", "children"),
            Input("interval-component", "n_intervals"),
        )
        def update_quick_stats(n):
            courses = self.curriculum.list_courses(teacher_id=self.teacher_id)
            assignments = self.assignment_manager.list_assignments(created_by=self.teacher_id)
            
            pending_grading = 0
            for assignment in assignments:
                subs = self.assignment_manager.get_assignment_submissions(
                    assignment.assignment_id
                )
                pending_grading += len([s for s in subs if s.score is None])
            
            return dbc.Row([
                dbc.Col([
                    html.H4(len(courses), className="text-primary"),
                    html.P("Active Courses"),
                ], width=3),
                dbc.Col([
                    html.H4(len(assignments), className="text-info"),
                    html.P("Total Assignments"),
                ], width=3),
                dbc.Col([
                    html.H4(pending_grading, className="text-warning"),
                    html.P("Pending Grading"),
                ], width=3),
                dbc.Col([
                    html.H4(sum(len(c.enrolled_students) for c in courses), className="text-success"),
                    html.P("Total Students"),
                ], width=3),
            ])
        
        @self.app.callback(
            Output("tab-content", "children"),
            Input("tabs", "active_tab"),
        )
        def render_tab_content(active_tab):
            if active_tab == "courses":
                return self._render_courses_tab()
            elif active_tab == "assignments":
                return self._render_assignments_tab()
            elif active_tab == "grading":
                return self._render_grading_tab()
            elif active_tab == "progress":
                return self._render_progress_tab()
            elif active_tab == "analytics":
                return self._render_analytics_tab()
            return html.Div("Select a tab")
    
    def _render_courses_tab(self) -> dbc.Container:
        """Render courses tab."""
        courses = self.curriculum.list_courses(teacher_id=self.teacher_id)
        
        course_cards = []
        for course in courses:
            lessons = self.curriculum.get_course_lessons(course.course_id)
            
            card = dbc.Card([
                dbc.CardHeader(html.H5(course.name)),
                dbc.CardBody([
                    html.P(course.description),
                    html.Hr(),
                    dbc.Row([
                        dbc.Col([
                            html.Strong("Students: "),
                            html.Span(len(course.enrolled_students)),
                        ], width=4),
                        dbc.Col([
                            html.Strong("Lessons: "),
                            html.Span(len(lessons)),
                        ], width=4),
                        dbc.Col([
                            html.Strong("Level: "),
                            html.Span(course.difficulty.value.title()),
                        ], width=4),
                    ]),
                ]),
                dbc.CardFooter([
                    dbc.Button("View Details", color="primary", size="sm", className="me-2"),
                    dbc.Button("Manage", color="secondary", size="sm"),
                ]),
            ], className="mb-3")
            
            course_cards.append(card)
        
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("My Courses"),
                    dbc.Button("Create New Course", color="success", className="mb-3"),
                ]),
            ]),
            dbc.Row([
                dbc.Col(course_cards if course_cards else html.P("No courses yet."), width=12),
            ]),
        ])
    
    def _render_assignments_tab(self) -> dbc.Container:
        """Render assignments tab."""
        assignments = self.assignment_manager.list_assignments(created_by=self.teacher_id)
        
        assignment_data = []
        for assignment in assignments[:20]:
            submissions = self.assignment_manager.get_assignment_submissions(
                assignment.assignment_id
            )
            graded = len([s for s in submissions if s.score is not None])
            
            assignment_data.append({
                "Title": assignment.title,
                "Course": assignment.course_id[:8] + "...",
                "Due Date": assignment.due_date.strftime("%Y-%m-%d") if assignment.due_date else "No deadline",
                "Points": assignment.points,
                "Submissions": len(submissions),
                "Graded": f"{graded}/{len(submissions)}",
            })
        
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("Assignments"),
                    dbc.Button("Create New Assignment", color="success", className="mb-3"),
                ]),
            ]),
            dbc.Row([
                dbc.Col([
                    dash_table.DataTable(
                        data=assignment_data,
                        columns=[{"name": col, "id": col} for col in assignment_data[0].keys()] if assignment_data else [],
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left'},
                        style_header={
                            'backgroundColor': 'rgb(230, 230, 230)',
                            'fontWeight': 'bold'
                        },
                        page_size=10,
                    ) if assignment_data else html.P("No assignments yet."),
                ], width=12),
            ]),
        ])
    
    def _render_grading_tab(self) -> dbc.Container:
        """Render grading queue tab."""
        assignments = self.assignment_manager.list_assignments(created_by=self.teacher_id)
        
        pending_submissions = []
        for assignment in assignments:
            submissions = self.assignment_manager.get_assignment_submissions(
                assignment.assignment_id
            )
            for submission in submissions:
                if submission.score is None:
                    student = self.progress_tracker.get_student(submission.student_id)
                    pending_submissions.append({
                        "submission": submission,
                        "assignment": assignment,
                        "student": student,
                    })
        
        submission_cards = []
        for item in pending_submissions[:10]:
            submission = item["submission"]
            assignment = item["assignment"]
            student = item["student"]
            
            card = dbc.Card([
                dbc.CardHeader([
                    html.H5(f"{assignment.title}"),
                    html.Small(f"Student: {student.name if student else submission.student_id}"),
                ]),
                dbc.CardBody([
                    html.P(f"Submitted: {submission.submitted_at.strftime('%Y-%m-%d %H:%M')}"),
                    html.P(f"Attempt: {submission.attempt_number}"),
                    html.Pre(
                        submission.code[:200] + "..." if len(submission.code) > 200 else submission.code,
                        style={'backgroundColor': '#f5f5f5', 'padding': '10px', 'fontSize': '12px'}
                    ),
                ]),
                dbc.CardFooter([
                    dbc.Button(
                        "Auto-Grade",
                        id={"type": "auto-grade-btn", "index": submission.submission_id},
                        color="primary",
                        size="sm",
                        className="me-2",
                    ),
                    dbc.Button("Manual Grade", color="secondary", size="sm"),
                ]),
            ], className="mb-3")
            
            submission_cards.append(card)
        
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("Grading Queue"),
                    html.P(f"{len(pending_submissions)} submissions pending"),
                ]),
            ]),
            dbc.Row([
                dbc.Col(
                    submission_cards if submission_cards else html.P("No submissions pending!"),
                    width=12
                ),
            ]),
        ])
    
    def _render_progress_tab(self) -> dbc.Container:
        """Render student progress tab."""
        courses = self.curriculum.list_courses(teacher_id=self.teacher_id)
        
        all_students = set()
        for course in courses:
            all_students.update(course.enrolled_students)
        
        student_data = []
        for student_id in list(all_students)[:50]:
            student = self.progress_tracker.get_student(student_id)
            if student:
                stats = self.progress_tracker.get_progress_stats(student_id)
                student_data.append({
                    "Name": student.name,
                    "Level": stats["level"],
                    "XP": stats["total_xp"],
                    "Tutorials": stats["tutorials_completed"],
                    "Assignments": stats["assignments_completed"],
                    "Badges": stats["badges_earned"],
                })
        
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("Student Progress"),
                ]),
            ]),
            dbc.Row([
                dbc.Col([
                    dash_table.DataTable(
                        data=student_data,
                        columns=[{"name": col, "id": col} for col in student_data[0].keys()] if student_data else [],
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left'},
                        style_header={
                            'backgroundColor': 'rgb(230, 230, 230)',
                            'fontWeight': 'bold'
                        },
                        sort_action="native",
                        filter_action="native",
                        page_size=20,
                    ) if student_data else html.P("No student data available."),
                ], width=12),
            ]),
        ])
    
    def _render_analytics_tab(self) -> dbc.Container:
        """Render analytics tab."""
        courses = self.curriculum.list_courses(teacher_id=self.teacher_id)
        
        course_stats = []
        for course in courses:
            stats = self.assignment_manager.get_course_statistics(course.course_id)
            course_stats.append({
                "course": course.name,
                "submissions": stats["total_submissions"],
                "avg_score": stats["average_score"],
                "students": len(course.enrolled_students),
            })
        
        if course_stats:
            fig_submissions = px.bar(
                course_stats,
                x="course",
                y="submissions",
                title="Submissions by Course",
            )
            
            fig_scores = px.bar(
                course_stats,
                x="course",
                y="avg_score",
                title="Average Scores by Course",
            )
        else:
            fig_submissions = go.Figure()
            fig_scores = go.Figure()
        
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("Analytics"),
                ]),
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=fig_submissions),
                ], width=6),
                dbc.Col([
                    dcc.Graph(figure=fig_scores),
                ], width=6),
            ]),
        ])
    
    def run(self, host: str = "0.0.0.0", port: int = 8052, debug: bool = False) -> None:
        """Run the dashboard server."""
        print(f"Starting Teacher Dashboard on http://{host}:{port}")
        print(f"Teacher ID: {self.teacher_id}")
        self.server.run(host=host, port=port, debug=debug)
