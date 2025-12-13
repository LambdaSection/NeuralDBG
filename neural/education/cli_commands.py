"""
CLI commands for the education module.
"""

from __future__ import annotations

import click
from pathlib import Path
from typing import Optional

from .assignments import AssignmentManager
from .curriculum import Curriculum, CurriculumTemplate
from .grading import AutoGrader, GradingCriteria
from .lms_integration import CanvasLMS, MoodleLMS, BlackboardLMS
from .notebook_tutorials import TutorialLibrary
from .progress_tracker import ProgressTracker
from .teacher_dashboard import TeacherDashboard


@click.group(name="education")
def education_cli():
    """Education module commands for teaching and learning Neural DSL."""
    pass


@education_cli.group(name="tutorial")
def tutorial_group():
    """Manage interactive tutorials."""
    pass


@tutorial_group.command(name="create")
@click.option("--title", required=True, help="Tutorial title")
@click.option("--difficulty", type=click.Choice(["beginner", "intermediate", "advanced", "expert"]), default="beginner")
@click.option("--time", type=int, default=30, help="Estimated time in minutes")
def create_tutorial(title: str, difficulty: str, time: int):
    """Create a new tutorial."""
    from .models import DifficultyLevel
    from .notebook_tutorials import NotebookTutorial
    import uuid
    
    tutorial = NotebookTutorial(
        tutorial_id=str(uuid.uuid4()),
        title=title,
        description=f"Tutorial: {title}",
        difficulty=DifficultyLevel(difficulty),
        estimated_time=time,
    )
    
    library = TutorialLibrary()
    library.add_tutorial(tutorial)
    
    click.echo(f"✓ Created tutorial: {tutorial.tutorial_id}")
    click.echo(f"  Saved to: neural_education_tutorials/{tutorial.tutorial_id}.ipynb")


@tutorial_group.command(name="list")
@click.option("--difficulty", type=click.Choice(["beginner", "intermediate", "advanced", "expert"]), help="Filter by difficulty")
def list_tutorials(difficulty: Optional[str]):
    """List all tutorials."""
    from .models import DifficultyLevel
    
    library = TutorialLibrary()
    
    diff_filter = DifficultyLevel(difficulty) if difficulty else None
    tutorials = library.list_tutorials(difficulty=diff_filter)
    
    if not tutorials:
        click.echo("No tutorials found.")
        return
    
    click.echo(f"\nFound {len(tutorials)} tutorial(s):\n")
    for tutorial in tutorials:
        click.echo(f"  {tutorial.tutorial_id}")
        click.echo(f"    Title: {tutorial.title}")
        click.echo(f"    Difficulty: {tutorial.difficulty.value}")
        click.echo(f"    Time: {tutorial.estimated_time} min")
        click.echo(f"    XP: {tutorial.xp_reward}")
        click.echo()


@education_cli.group(name="course")
def course_group():
    """Manage courses and curriculum."""
    pass


@course_group.command(name="create")
@click.option("--name", required=True, help="Course name")
@click.option("--teacher-id", required=True, help="Teacher ID")
@click.option("--template", type=click.Choice(["intro-nn", "dl-fundamentals", "nlp", "cv"]), help="Use a template")
def create_course(name: str, teacher_id: str, template: Optional[str]):
    """Create a new course."""
    curriculum = Curriculum()
    
    if template:
        if template == "intro-nn":
            course = CurriculumTemplate.intro_to_neural_networks()
        elif template == "dl-fundamentals":
            course = CurriculumTemplate.deep_learning_fundamentals()
        elif template == "nlp":
            course = CurriculumTemplate.advanced_nlp()
        elif template == "cv":
            course = CurriculumTemplate.computer_vision()
        
        course.teacher_id = teacher_id
        course.name = name
    else:
        course = curriculum.create_course(
            name=name,
            description=f"Course: {name}",
            teacher_id=teacher_id,
        )
    
    curriculum.courses[course.course_id] = course
    curriculum._save_data()
    
    click.echo(f"✓ Created course: {course.course_id}")
    click.echo(f"  Name: {course.name}")
    click.echo(f"  Teacher: {teacher_id}")


@course_group.command(name="list")
@click.option("--teacher-id", help="Filter by teacher ID")
def list_courses(teacher_id: Optional[str]):
    """List all courses."""
    curriculum = Curriculum()
    courses = curriculum.list_courses(teacher_id=teacher_id)
    
    if not courses:
        click.echo("No courses found.")
        return
    
    click.echo(f"\nFound {len(courses)} course(s):\n")
    for course in courses:
        click.echo(f"  {course.course_id}")
        click.echo(f"    Name: {course.name}")
        click.echo(f"    Teacher: {course.teacher_id}")
        click.echo(f"    Students: {len(course.enrolled_students)}")
        click.echo(f"    Lessons: {len(course.lessons)}")
        click.echo()


@education_cli.group(name="assignment")
def assignment_group():
    """Manage assignments."""
    pass


@assignment_group.command(name="create")
@click.option("--title", required=True, help="Assignment title")
@click.option("--course-id", required=True, help="Course ID")
@click.option("--teacher-id", required=True, help="Teacher ID")
@click.option("--points", type=int, default=100, help="Points possible")
def create_assignment(title: str, course_id: str, teacher_id: str, points: int):
    """Create a new assignment."""
    manager = AssignmentManager()
    
    assignment = manager.create_assignment(
        title=title,
        description=f"Assignment: {title}",
        course_id=course_id,
        created_by=teacher_id,
        points=points,
    )
    
    click.echo(f"✓ Created assignment: {assignment.assignment_id}")
    click.echo(f"  Title: {assignment.title}")
    click.echo(f"  Points: {assignment.points}")


@assignment_group.command(name="list")
@click.option("--course-id", help="Filter by course ID")
@click.option("--teacher-id", help="Filter by teacher ID")
def list_assignments(course_id: Optional[str], teacher_id: Optional[str]):
    """List all assignments."""
    manager = AssignmentManager()
    assignments = manager.list_assignments(course_id=course_id, created_by=teacher_id)
    
    if not assignments:
        click.echo("No assignments found.")
        return
    
    click.echo(f"\nFound {len(assignments)} assignment(s):\n")
    for assignment in assignments:
        submissions = manager.get_assignment_submissions(assignment.assignment_id)
        graded = len([s for s in submissions if s.score is not None])
        
        click.echo(f"  {assignment.assignment_id}")
        click.echo(f"    Title: {assignment.title}")
        click.echo(f"    Course: {assignment.course_id}")
        click.echo(f"    Points: {assignment.points}")
        click.echo(f"    Submissions: {len(submissions)} ({graded} graded)")
        click.echo()


@assignment_group.command(name="grade")
@click.option("--submission-id", required=True, help="Submission ID")
@click.option("--auto", is_flag=True, help="Use automated grading")
def grade_assignment(submission_id: str, auto: bool):
    """Grade a submission."""
    manager = AssignmentManager()
    submission = manager.get_submission(submission_id)
    
    if not submission:
        click.echo(f"✗ Submission not found: {submission_id}", err=True)
        return
    
    if auto:
        criteria = GradingCriteria(
            syntax_valid=10,
            architecture_valid=30,
            code_quality=20,
            documentation=15,
            creativity=15,
        )
        
        grader = AutoGrader(criteria)
        result = grader.grade(submission.code)
        
        click.echo("\n" + grader.generate_feedback_report(result))
        
        manager.grade_submission(
            submission_id=submission_id,
            score=result.percentage,
            feedback="\n".join(result.feedback),
            graded_by="auto-grader",
        )
        
        click.echo(f"\n✓ Graded submission: {submission_id}")
        click.echo(f"  Score: {result.percentage:.1f}%")
    else:
        click.echo(f"Submission: {submission_id}")
        click.echo(f"Student: {submission.student_id}")
        click.echo("\nCode:")
        click.echo(submission.code)
        click.echo("\nUse --auto for automated grading or grade manually.")


@education_cli.group(name="progress")
def progress_group():
    """Manage student progress."""
    pass


@progress_group.command(name="stats")
@click.option("--student-id", required=True, help="Student ID")
def show_progress(student_id: str):
    """Show student progress statistics."""
    tracker = ProgressTracker()
    stats = tracker.get_progress_stats(student_id)
    
    if not stats:
        click.echo(f"✗ Student not found: {student_id}", err=True)
        return
    
    click.echo(f"\nProgress for {stats['name']}:")
    click.echo(f"  Level: {stats['level']}")
    click.echo(f"  Total XP: {stats['total_xp']}")
    click.echo(f"  XP to next level: {stats['xp_to_next_level']}")
    click.echo(f"  Tutorials completed: {stats['tutorials_completed']}")
    click.echo(f"  Assignments completed: {stats['assignments_completed']}")
    click.echo(f"  Badges earned: {stats['badges_earned']}")
    click.echo()


@progress_group.command(name="leaderboard")
@click.option("--limit", type=int, default=10, help="Number of students to show")
def show_leaderboard(limit: int):
    """Show student leaderboard."""
    tracker = ProgressTracker()
    leaderboard = tracker.get_leaderboard(limit=limit)
    
    if not leaderboard:
        click.echo("No students found.")
        return
    
    click.echo(f"\nTop {len(leaderboard)} Students:\n")
    for i, student in enumerate(leaderboard, 1):
        click.echo(f"  {i}. {student.name}")
        click.echo(f"     Level {student.level} - {student.total_xp} XP")
        click.echo(f"     Badges: {len(student.badges)}")
        click.echo()


@education_cli.group(name="dashboard")
def dashboard_group():
    """Manage teacher dashboard."""
    pass


@dashboard_group.command(name="run")
@click.option("--teacher-id", required=True, help="Teacher ID")
@click.option("--host", default="0.0.0.0", help="Host address")
@click.option("--port", type=int, default=8052, help="Port number")
@click.option("--debug", is_flag=True, help="Enable debug mode")
def run_dashboard(teacher_id: str, host: str, port: int, debug: bool):
    """Run the teacher dashboard."""
    try:
        dashboard = TeacherDashboard(teacher_id=teacher_id)
        click.echo(f"Starting teacher dashboard for {teacher_id}")
        click.echo(f"Access at: http://{host}:{port}")
        dashboard.run(host=host, port=port, debug=debug)
    except ImportError as e:
        click.echo(f"✗ Error: {e}", err=True)
        click.echo("Install dashboard dependencies: pip install -e \".[education,dashboard]\"", err=True)


@education_cli.group(name="lms")
def lms_group():
    """LMS integration commands."""
    pass


@lms_group.command(name="sync-canvas")
@click.option("--url", required=True, help="Canvas base URL")
@click.option("--api-key", required=True, help="Canvas API key")
@click.option("--course-id", required=True, help="Course ID")
def sync_canvas(url: str, api_key: str, course_id: str):
    """Sync with Canvas LMS."""
    canvas = CanvasLMS(base_url=url, api_key=api_key)
    
    courses = canvas.get_courses()
    click.echo(f"Found {len(courses)} Canvas courses")
    
    students = canvas.get_students(course_id)
    click.echo(f"Found {len(students)} students in course {course_id}")
    
    click.echo("\n✓ Canvas sync complete")


@lms_group.command(name="sync-moodle")
@click.option("--url", required=True, help="Moodle base URL")
@click.option("--token", required=True, help="Moodle web service token")
@click.option("--course-id", required=True, help="Course ID")
def sync_moodle(url: str, token: str, course_id: str):
    """Sync with Moodle LMS."""
    moodle = MoodleLMS(base_url=url, api_key=token)
    
    courses = moodle.get_courses()
    click.echo(f"Found {len(courses)} Moodle courses")
    
    students = moodle.get_students(course_id)
    click.echo(f"Found {len(students)} students in course {course_id}")
    
    click.echo("\n✓ Moodle sync complete")


def register_education_commands(cli):
    """Register education commands with the main CLI."""
    cli.add_command(education_cli)
