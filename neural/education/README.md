# Neural DSL - Education Module

The Education module provides comprehensive features for teaching and learning Neural DSL in academic and professional settings.

## Features

### Interactive Jupyter Notebook Tutorials
- Step-by-step guided tutorials with executable code cells
- Progressive difficulty levels (Beginner, Intermediate, Advanced, Expert)
- Built-in validation and feedback
- Pre-built tutorial library covering core concepts
- Support for exercises with hints and solutions

### Progress Tracking and Gamification
- XP (Experience Points) system with student leveling
- Student achievements and milestones
- Badges for different accomplishments:
  - Completion badges (first tutorial, first assignment)
  - Speed badges (fast completion)
  - Accuracy badges (perfect scores)
  - Mastery badges (advanced achievements)
  - Consistency badges (learning streaks)
- Progress dashboards and analytics
- Leaderboards for competitive learning

### Assignment System
- Flexible assignment creation with due dates
- Student submission tracking with version history
- Support for multiple submission attempts
- Late submission handling with configurable penalties
- File upload support
- Rubric-based grading criteria

### Automated Grading
- Architecture validation (syntax, layer count, structure)
- Performance benchmarking
- Code quality checks (indentation, best practices)
- Documentation requirements
- Creativity assessment
- Custom test cases
- Detailed feedback generation
- Breakdown of scores by category

### Curriculum Templates
- Pre-built course structures for common topics:
  - Introduction to Neural Networks
  - Deep Learning Fundamentals
  - Advanced Natural Language Processing
  - Computer Vision with Deep Learning
- Lesson planning tools with learning objectives
- Prerequisite tracking and dependencies
- Customizable course templates

### LMS Integration
- **Canvas LMS**: Full integration with course sync, grading, and assignments
- **Moodle**: Web service API integration
- **Blackboard Learn**: REST API integration
- Automatic grade synchronization
- Assignment import/export
- Student roster management
- Submission retrieval

### Teacher Dashboard
- Web-based classroom management interface
- Real-time statistics and metrics
- Course overview with enrollment data
- Assignment creation and management
- Automated grading queue
- Student progress monitoring
- Analytics and visualization
- Performance insights

## Installation

```bash
pip install -e ".[education]"
```

For full functionality including dashboard:

```bash
pip install -e ".[education,dashboard,visualization]"
```

## Quick Start

### Creating a Course

```python
from neural.education import Curriculum, CurriculumTemplate

curriculum = Curriculum()

course = curriculum.create_course(
    name="Introduction to Neural Networks",
    description="Learn the fundamentals of neural networks using Neural DSL",
    teacher_id="teacher123",
    duration_weeks=8,
)

lessons = CurriculumTemplate.create_lessons_for_intro_course()
for lesson in lessons:
    curriculum.lessons[lesson.lesson_id] = lesson
    curriculum.add_lesson_to_course(course.course_id, lesson.lesson_id)
```

### Creating Interactive Tutorials

```python
from neural.education import NotebookTutorial, TutorialLibrary
from neural.education.models import DifficultyLevel

library = TutorialLibrary()

tutorial = NotebookTutorial(
    tutorial_id="my-first-cnn",
    title="Building Your First CNN",
    description="Learn to create convolutional neural networks",
    difficulty=DifficultyLevel.BEGINNER,
    estimated_time=45,
)

tutorial.add_markdown("## Introduction to CNNs")
tutorial.add_code("""
network SimpleCNN {
    input: [28, 28, 1]
    
    layer conv1: Conv2D(filters=32, kernel_size=3, activation='relu')
    layer pool1: MaxPool2D(pool_size=2)
    layer flatten: Flatten()
    layer output: Dense(units=10, activation='softmax')
    
    flow: input -> conv1 -> pool1 -> flatten -> output
}
""")

tutorial.add_exercise(
    instruction="Add batch normalization after the conv layer",
    starter_code="layer bn1: BatchNorm()",
    hints=["Insert it between conv1 and pool1"],
)

library.add_tutorial(tutorial)
tutorial.save("my_first_cnn.ipynb")
```

### Creating Assignments

```python
from neural.education import AssignmentManager, Assignment
from datetime import datetime, timedelta

manager = AssignmentManager()

assignment = manager.create_assignment(
    title="Build an Image Classifier",
    description="Create a CNN for CIFAR-10 classification",
    course_id="course123",
    created_by="teacher123",
    due_date=datetime.now() + timedelta(days=7),
    points=100,
    requirements={
        "min_layers": 5,
        "must_include": ["Conv2D", "MaxPool2D", "Dense"],
    },
)
```

### Automated Grading

```python
from neural.education import AutoGrader, GradingCriteria

criteria = GradingCriteria(
    syntax_valid=10,
    architecture_valid=30,
    layer_count_min=5,
    layer_count_max=15,
    code_quality=20,
    documentation=15,
    creativity=15,
    custom_tests=[
        {
            "name": "Uses CNN layers",
            "type": "contains",
            "required": ["Conv2D", "MaxPool2D"],
            "points": 10,
        }
    ],
)

grader = AutoGrader(criteria)
result = grader.grade(student_code)

print(grader.generate_feedback_report(result))
```

### Tracking Student Progress

```python
from neural.education import ProgressTracker

tracker = ProgressTracker()

student = tracker.register_student(
    student_id="student123",
    name="Jane Doe",
    email="jane@example.com",
)

new_badges = tracker.complete_tutorial("student123", "intro-neural-dsl")
tracker.complete_assignment("student123", "assignment1", score=95.0)

stats = tracker.get_progress_stats("student123")
print(f"Level: {stats['level']}, XP: {stats['total_xp']}")
print(f"Badges earned: {stats['badges_earned']}")

leaderboard = tracker.get_leaderboard(limit=10)
```

### LMS Integration

```python
from neural.education import CanvasLMS

canvas = CanvasLMS(
    base_url="https://canvas.example.com",
    api_key="your_api_key_here",
)

courses = canvas.get_courses()
students = canvas.get_students(course_id="12345")

assignment_id = canvas.create_assignment(
    course_id="12345",
    assignment_data={
        "title": "Neural DSL Assignment",
        "description": "Build a neural network",
        "points": 100,
        "due_date": "2024-12-31T23:59:59Z",
    },
)

canvas.submit_grade(
    course_id="12345",
    assignment_id=assignment_id,
    student_id="67890",
    grade=95.0,
    feedback="Excellent work!",
)
```

### Running the Teacher Dashboard

```python
from neural.education import TeacherDashboard

dashboard = TeacherDashboard(teacher_id="teacher123")
dashboard.run(host="0.0.0.0", port=8052, debug=True)
```

Then open your browser to `http://localhost:8052` to access the dashboard.

## Architecture

### Components

- **`models.py`**: Core data models (StudentProfile, TeacherProfile, ClassRoom, enums)
- **`notebook_tutorials.py`**: Interactive Jupyter tutorial system with TutorialLibrary
- **`progress_tracker.py`**: Student progress tracking, XP, levels, achievements, badges
- **`assignments.py`**: Assignment creation, submission management, versioning
- **`grading.py`**: Automated grading engine with multiple criteria types
- **`curriculum.py`**: Course and lesson management with templates
- **`lms_integration.py`**: LMS connectors (Canvas, Moodle, Blackboard)
- **`teacher_dashboard.py`**: Web-based teacher interface built with Dash

### Data Storage

All data is stored in JSON files organized by module:
- `neural_education_progress/`: Student profiles, badges
- `neural_education_assignments/`: Assignments, submissions
- `neural_education_curriculum/`: Courses, lessons
- `neural_education_tutorials/`: Tutorial library

### Gamification System

Students earn XP and level up:
- Level = 1 + (total_xp // 1000)
- Tutorial completion: 100-1000 XP based on difficulty
- Assignment completion: Based on score and assignment points
- Badges provide bonus XP: 100-500 XP per badge

Badge Types:
- **Completion**: First milestones (first tutorial, first assignment)
- **Speed**: Fast completion times
- **Accuracy**: Perfect or high scores
- **Creativity**: Advanced techniques and approaches
- **Consistency**: Learning streaks
- **Mastery**: Advanced achievements (e.g., 10+ tutorials)

## Dependencies

Required:
- `nbformat`: Jupyter notebook format handling
- `jupyter`: For interactive tutorials

Optional:
- `dash`, `dash-bootstrap-components`: Teacher dashboard
- `plotly`: Progress visualization and analytics
- `requests`: LMS API integration
- `flask`: Dashboard server

Install all with:

```bash
pip install -e ".[education,dashboard,visualization]"
```

## Use Cases

### Academic Institutions
- Teach deep learning courses with structured curriculum
- Track student progress and performance
- Automate grading for programming assignments
- Integrate with existing LMS infrastructure

### Corporate Training
- Onboard ML engineers with guided tutorials
- Gamify learning with achievements and leaderboards
- Monitor team progress and completion rates
- Create custom curriculum for specific domains

### Self-Paced Learning
- Interactive tutorials with immediate feedback
- Clear progression path with difficulty levels
- Motivating gamification elements
- Practice assignments with automated grading

### MOOCs and Online Courses
- Scale education with automated grading
- Provide consistent feedback to all students
- Track engagement and completion metrics
- Export data to popular LMS platforms

## Advanced Features

### Custom Grading Criteria

```python
from neural.education import GradingCriteria

criteria = GradingCriteria(
    syntax_valid=10,
    architecture_valid=20,
    layer_count_min=3,
    layer_count_max=10,
    code_quality=25,
    documentation=15,
    creativity=20,
    custom_tests=[
        {
            "name": "Uses attention mechanism",
            "type": "regex",
            "pattern": r"Attention|MultiHeadAttention",
            "points": 10,
        },
        {
            "name": "Has proper documentation",
            "type": "regex",
            "pattern": r"#.*(?:layer|network|architecture)",
            "points": 5,
        },
    ],
)
```

### Course Templates

```python
from neural.education import CurriculumTemplate

intro_course = CurriculumTemplate.intro_to_neural_networks()
dl_course = CurriculumTemplate.deep_learning_fundamentals()
nlp_course = CurriculumTemplate.advanced_nlp()
cv_course = CurriculumTemplate.computer_vision()
```

### Progress Analytics

```python
tracker = ProgressTracker()

stats = tracker.get_progress_stats("student123")
leaderboard = tracker.get_leaderboard(limit=10)

for student in leaderboard:
    print(f"{student.name}: Level {student.level} ({student.total_xp} XP)")
```

## API Reference

See individual module docstrings for detailed API documentation:
- `NotebookTutorial`: Tutorial creation and management
- `ProgressTracker`: Progress tracking and gamification
- `AssignmentManager`: Assignment lifecycle management
- `AutoGrader`: Automated grading engine
- `Curriculum`: Course and lesson management
- `LMSConnector`: Base class for LMS integrations
- `TeacherDashboard`: Web interface for teachers

## Contributing

To add support for additional LMS platforms:
1. Extend the `LMSConnector` abstract class
2. Implement all required methods
3. Add authentication and API-specific logic
4. Submit a pull request with tests

## License

MIT License - see LICENSE file for details
