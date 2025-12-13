"""
Example usage of the education module.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from .assignments import AssignmentManager
from .curriculum import Curriculum, CurriculumTemplate
from .grading import AutoGrader, GradingCriteria
from .lms_integration import CanvasLMS
from .models import DifficultyLevel
from .notebook_tutorials import NotebookTutorial, TutorialLibrary
from .progress_tracker import ProgressTracker
from .teacher_dashboard import TeacherDashboard


def example_create_tutorial_library():
    """Create a tutorial library with example tutorials."""
    library = TutorialLibrary.create_default_tutorials()
    print(f"Created tutorial library with {len(library.tutorials)} tutorials")
    
    for tutorial in library.list_tutorials():
        print(f"  - {tutorial.title} ({tutorial.difficulty.value}, {tutorial.estimated_time} min)")
    
    return library


def example_create_course():
    """Create a course with lessons."""
    curriculum = Curriculum()
    
    course = curriculum.create_course(
        name="Introduction to Neural Networks with Neural DSL",
        description="Learn neural network fundamentals using Neural DSL",
        teacher_id="teacher001",
        duration_weeks=8,
        credits=3.0,
    )
    
    lessons = CurriculumTemplate.create_lessons_for_intro_course()
    for lesson in lessons:
        curriculum.lessons[lesson.lesson_id] = lesson
        curriculum.add_lesson_to_course(course.course_id, lesson.lesson_id)
    
    print(f"Created course: {course.name}")
    print(f"  Course ID: {course.course_id}")
    print(f"  Lessons: {len(curriculum.get_course_lessons(course.course_id))}")
    
    return course


def example_create_assignment():
    """Create an assignment with grading criteria."""
    manager = AssignmentManager()
    
    assignment = manager.create_assignment(
        title="Build a Convolutional Neural Network",
        description="""
        Create a CNN for image classification using Neural DSL.
        
        Requirements:
        - Use at least 2 convolutional layers
        - Include pooling layers
        - Add batch normalization
        - Use dropout for regularization
        - End with a fully connected classifier
        
        Your network should achieve reasonable performance on MNIST or CIFAR-10.
        """,
        course_id="course123",
        created_by="teacher001",
        due_date=datetime.now() + timedelta(days=7),
        points=100,
        requirements={
            "min_conv_layers": 2,
            "min_total_layers": 5,
            "must_include": ["Conv2D", "MaxPool2D", "Dense"],
        },
        starter_code="""
network ImageClassifier {
    input: [32, 32, 3]
    
    # Your layers here
    
    flow: # Your flow here
}
        """.strip(),
    )
    
    print(f"Created assignment: {assignment.title}")
    print(f"  Assignment ID: {assignment.assignment_id}")
    print(f"  Points: {assignment.points}")
    print(f"  Due: {assignment.due_date.strftime('%Y-%m-%d')}")
    
    return assignment


def example_auto_grade_submission():
    """Example of automated grading."""
    student_code = """
network MNIST_CNN {
    input: [28, 28, 1]
    
    # First convolutional block
    layer conv1: Conv2D(filters=32, kernel_size=3, activation='relu')
    layer bn1: BatchNorm()
    layer pool1: MaxPool2D(pool_size=2)
    layer dropout1: Dropout(rate=0.25)
    
    # Second convolutional block
    layer conv2: Conv2D(filters=64, kernel_size=3, activation='relu')
    layer bn2: BatchNorm()
    layer pool2: MaxPool2D(pool_size=2)
    layer dropout2: Dropout(rate=0.25)
    
    # Classifier
    layer flatten: Flatten()
    layer dense1: Dense(units=128, activation='relu')
    layer dropout3: Dropout(rate=0.5)
    layer output: Dense(units=10, activation='softmax')
    
    flow: input -> conv1 -> bn1 -> pool1 -> dropout1 
          -> conv2 -> bn2 -> pool2 -> dropout2 
          -> flatten -> dense1 -> dropout3 -> output
}
    """
    
    criteria = GradingCriteria(
        syntax_valid=10,
        architecture_valid=30,
        layer_count_min=5,
        code_quality=20,
        documentation=15,
        creativity=15,
        custom_tests=[
            {
                "name": "Uses CNN layers",
                "type": "contains",
                "required": ["Conv2D", "MaxPool2D"],
                "points": 5,
            },
            {
                "name": "Uses regularization",
                "type": "contains",
                "required": ["Dropout", "BatchNorm"],
                "points": 5,
            },
        ],
    )
    
    grader = AutoGrader(criteria)
    result = grader.grade(student_code)
    
    print("\n" + "=" * 60)
    print("AUTOMATED GRADING RESULT")
    print("=" * 60)
    print(grader.generate_feedback_report(result))
    
    return result


def example_track_student_progress():
    """Track student progress with gamification."""
    tracker = ProgressTracker()
    
    student = tracker.register_student(
        student_id="student001",
        name="Alice Johnson",
        email="alice@example.com",
    )
    
    print(f"Registered student: {student.name}")
    
    badges = tracker.complete_tutorial("student001", "intro-neural-dsl", completion_time=800)
    tracker.complete_tutorial("student001", "building-cnns")
    tracker.complete_assignment("student001", "assignment001", score=95.0)
    
    print(f"\nEarned {len(badges)} badge(s)")
    for badge in badges:
        achievement = tracker.achievements.get(badge.achievement_id)
        if achievement:
            print(f"  {achievement.icon} {achievement.name}: {achievement.description}")
    
    stats = tracker.get_progress_stats("student001")
    print(f"\nProgress Stats:")
    print(f"  Level: {stats['level']}")
    print(f"  Total XP: {stats['total_xp']}")
    print(f"  Tutorials: {stats['tutorials_completed']}")
    print(f"  Assignments: {stats['assignments_completed']}")
    print(f"  Badges: {stats['badges_earned']}")
    
    return student


def example_lms_integration():
    """Example LMS integration (requires valid credentials)."""
    print("Canvas LMS Integration Example")
    print("Note: Requires valid Canvas credentials")
    
    example_config = {
        "base_url": "https://canvas.example.com",
        "api_key": "your_api_key_here",
        "course_id": "12345",
    }
    
    print(f"\nConfiguration:")
    print(f"  Base URL: {example_config['base_url']}")
    print(f"  Course ID: {example_config['course_id']}")
    print("\nTo use:")
    print("  canvas = CanvasLMS(base_url, api_key)")
    print("  courses = canvas.get_courses()")
    print("  students = canvas.get_students(course_id)")
    
    return example_config


def example_teacher_dashboard():
    """Example teacher dashboard setup."""
    print("Teacher Dashboard Example")
    print("\nTo run the dashboard:")
    print("  from neural.education import TeacherDashboard")
    print("  dashboard = TeacherDashboard(teacher_id='teacher001')")
    print("  dashboard.run(port=8052)")
    print("\nThen open: http://localhost:8052")
    
    print("\nDashboard features:")
    print("  - Course management")
    print("  - Assignment creation and grading")
    print("  - Student progress tracking")
    print("  - Analytics and visualizations")
    print("  - Real-time statistics")


def run_all_examples():
    """Run all examples."""
    print("=" * 60)
    print("Neural DSL Education Module - Examples")
    print("=" * 60)
    print()
    
    print("1. Creating Tutorial Library")
    print("-" * 60)
    example_create_tutorial_library()
    print()
    
    print("2. Creating Course")
    print("-" * 60)
    example_create_course()
    print()
    
    print("3. Creating Assignment")
    print("-" * 60)
    example_create_assignment()
    print()
    
    print("4. Automated Grading")
    print("-" * 60)
    example_auto_grade_submission()
    print()
    
    print("5. Tracking Student Progress")
    print("-" * 60)
    example_track_student_progress()
    print()
    
    print("6. LMS Integration")
    print("-" * 60)
    example_lms_integration()
    print()
    
    print("7. Teacher Dashboard")
    print("-" * 60)
    example_teacher_dashboard()
    print()
    
    print("=" * 60)
    print("Examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_examples()
