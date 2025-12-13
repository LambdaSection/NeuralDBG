# Neural DSL Education Module - Quick Start Guide

Get started with teaching and learning Neural DSL in minutes!

## Installation

```bash
pip install -e ".[education,dashboard,visualization]"
```

## 5-Minute Quick Start

### 1. Create Your First Tutorial (30 seconds)

```python
from neural.education import NotebookTutorial, TutorialLibrary
from neural.education.models import DifficultyLevel

tutorial = NotebookTutorial(
    tutorial_id="my-first-tutorial",
    title="Introduction to Neural Networks",
    description="Learn the basics of neural networks",
    difficulty=DifficultyLevel.BEGINNER,
    estimated_time=30,
)

tutorial.add_markdown("## Welcome to Neural DSL!")
tutorial.add_code("""
network SimpleNet {
    input: [784]
    layer hidden: Dense(units=128, activation='relu')
    layer output: Dense(units=10, activation='softmax')
    flow: input -> hidden -> output
}
""")

library = TutorialLibrary()
library.add_tutorial(tutorial)
```

### 2. Track Student Progress (30 seconds)

```python
from neural.education import ProgressTracker

tracker = ProgressTracker()

student = tracker.register_student(
    student_id="student001",
    name="John Doe",
    email="john@example.com"
)

badges = tracker.complete_tutorial("student001", "my-first-tutorial")
print(f"Earned {len(badges)} badge(s)!")

stats = tracker.get_progress_stats("student001")
print(f"Level {stats['level']}, {stats['total_xp']} XP")
```

### 3. Create an Assignment (1 minute)

```python
from neural.education import AssignmentManager
from datetime import datetime, timedelta

manager = AssignmentManager()

assignment = manager.create_assignment(
    title="Build a Simple Network",
    description="Create a feedforward neural network with at least 3 layers",
    course_id="intro-course",
    created_by="teacher001",
    due_date=datetime.now() + timedelta(days=7),
    points=100,
)

print(f"Assignment created: {assignment.assignment_id}")
```

### 4. Auto-Grade a Submission (1 minute)

```python
from neural.education import AutoGrader, GradingCriteria

student_code = """
network StudentNet {
    input: [784]
    layer hidden1: Dense(units=128, activation='relu')
    layer hidden2: Dense(units=64, activation='relu')
    layer output: Dense(units=10, activation='softmax')
    flow: input -> hidden1 -> hidden2 -> output
}
"""

criteria = GradingCriteria(
    syntax_valid=20,
    architecture_valid=40,
    layer_count_min=3,
    code_quality=20,
    documentation=20,
)

grader = AutoGrader(criteria)
result = grader.grade(student_code)

print(f"Score: {result.percentage:.1f}%")
print(grader.generate_feedback_report(result))
```

### 5. Launch Teacher Dashboard (1 minute)

```python
from neural.education import TeacherDashboard

dashboard = TeacherDashboard(teacher_id="teacher001")
dashboard.run(port=8052)
```

Then open: http://localhost:8052

## Common Use Cases

### For Teachers

#### Setup a New Course

```python
from neural.education import Curriculum, CurriculumTemplate

curriculum = Curriculum()

course = CurriculumTemplate.intro_to_neural_networks()
course.teacher_id = "your-teacher-id"

curriculum.courses[course.course_id] = course
curriculum._save_data()

print(f"Course ready: {course.course_id}")
```

#### Bulk Import Students

```python
from neural.education import ProgressTracker

tracker = ProgressTracker()

students = [
    {"student_id": "s001", "name": "Alice", "email": "alice@example.com"},
    {"student_id": "s002", "name": "Bob", "email": "bob@example.com"},
    {"student_id": "s003", "name": "Carol", "email": "carol@example.com"},
]

for student_data in students:
    tracker.register_student(**student_data)

print(f"Registered {len(students)} students")
```

#### Grade All Pending Submissions

```python
from neural.education import AssignmentManager, AutoGrader, GradingCriteria

manager = AssignmentManager()
grader = AutoGrader(GradingCriteria())

assignment_id = "your-assignment-id"
submissions = manager.get_assignment_submissions(assignment_id)

for submission in submissions:
    if submission.score is None:
        result = grader.grade(submission.code)
        manager.grade_submission(
            submission_id=submission.submission_id,
            score=result.percentage,
            feedback="\n".join(result.feedback),
            graded_by="auto-grader",
        )
        print(f"Graded {submission.student_id}: {result.percentage:.1f}%")
```

### For Students

#### View Your Progress

```python
from neural.education import ProgressTracker

tracker = ProgressTracker()
stats = tracker.get_progress_stats("your-student-id")

print(f"Level: {stats['level']}")
print(f"XP: {stats['total_xp']} / {stats['total_xp'] + stats['xp_to_next_level']}")
print(f"Tutorials: {stats['tutorials_completed']}")
print(f"Assignments: {stats['assignments_completed']}")
print(f"Badges: {stats['badges_earned']}")
```

#### Submit an Assignment

```python
from neural.education import AssignmentManager

manager = AssignmentManager()

my_code = """
network MyNetwork {
    input: [784]
    # Your implementation here
}
"""

submission = manager.submit_assignment(
    assignment_id="assignment-id",
    student_id="your-student-id",
    code=my_code,
)

print(f"Submitted! Submission ID: {submission.submission_id}")
```

### For Administrators

#### Integrate with Canvas LMS

```python
from neural.education import CanvasLMS, AssignmentManager

canvas = CanvasLMS(
    base_url="https://your-canvas-instance.com",
    api_key="your-api-key"
)

course_id = "12345"
students = canvas.get_students(course_id)

manager = AssignmentManager()
for assignment in manager.list_assignments(course_id=course_id):
    submissions = manager.get_assignment_submissions(assignment.assignment_id)
    
    for submission in submissions:
        if submission.score is not None:
            canvas.submit_grade(
                course_id=course_id,
                assignment_id=assignment.assignment_id,
                student_id=submission.student_id,
                grade=submission.score,
                feedback=submission.feedback,
            )

print("Grades synced to Canvas!")
```

## CLI Usage

### Tutorial Management

```bash
neural education tutorial create --title "My Tutorial" --difficulty beginner
neural education tutorial list
```

### Course Management

```bash
neural education course create --name "Intro to NNs" --teacher-id teacher001
neural education course list --teacher-id teacher001
```

### Assignment Management

```bash
neural education assignment create \
    --title "Build a CNN" \
    --course-id course123 \
    --teacher-id teacher001 \
    --points 100

neural education assignment list --course-id course123
neural education assignment grade --submission-id sub123 --auto
```

### Progress Tracking

```bash
neural education progress stats --student-id student001
neural education progress leaderboard --limit 10
```

### Teacher Dashboard

```bash
neural education dashboard run --teacher-id teacher001 --port 8052
```

### LMS Integration

```bash
neural education lms sync-canvas \
    --url https://canvas.example.com \
    --api-key YOUR_KEY \
    --course-id 12345

neural education lms sync-moodle \
    --url https://moodle.example.com \
    --token YOUR_TOKEN \
    --course-id 67890
```

## Advanced Examples

### Custom Grading Criteria

```python
from neural.education import GradingCriteria

criteria = GradingCriteria(
    syntax_valid=10,
    architecture_valid=20,
    layer_count_min=5,
    layer_count_max=15,
    code_quality=20,
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
            "name": "Proper documentation",
            "type": "regex",
            "pattern": r"#.*architecture",
            "points": 5,
        },
        {
            "name": "Layer count check",
            "type": "layer_count",
            "min": 5,
            "max": 10,
            "points": 10,
        },
    ],
)
```

### Interactive Exercise with Validation

```python
from neural.education import NotebookTutorial

tutorial = NotebookTutorial(
    tutorial_id="cnn-tutorial",
    title="Building CNNs",
    description="Learn convolutional neural networks",
    difficulty=DifficultyLevel.INTERMEDIATE,
)

tutorial.add_exercise(
    instruction="Add a convolutional layer with 32 filters",
    starter_code="layer conv1: Conv2D(filters=?, kernel_size=3)",
    solution="layer conv1: Conv2D(filters=32, kernel_size=3, activation='relu')",
    hints=[
        "Filters define the number of feature maps",
        "ReLU is a common activation for CNNs",
    ],
)
```

## Tips & Best Practices

1. **Start Simple**: Begin with beginner tutorials and basic assignments
2. **Use Templates**: Leverage pre-built course templates
3. **Enable Auto-Grading**: Save time with automated grading for standard assignments
4. **Track Progress**: Regularly check student progress and engagement
5. **Gamify Learning**: Use badges and leaderboards to motivate students
6. **Integrate with LMS**: Sync with your existing learning platform
7. **Provide Feedback**: Include detailed feedback in grading reports
8. **Monitor Analytics**: Use the dashboard to identify struggling students

## Troubleshooting

### Dashboard won't start
```bash
pip install -e ".[dashboard]"
```

### Notebooks not generating
```bash
pip install nbformat jupyter
```

### LMS integration fails
- Check API credentials
- Verify base URL format
- Ensure proper permissions

## Next Steps

- Explore the full [README](README.md) for comprehensive documentation
- Check out [examples.py](examples.py) for more code samples
- Read the API documentation in module docstrings
- Join the community for support and best practices

## Support

For issues, questions, or contributions:
- GitHub Issues: https://github.com/Lemniscate-world/Neural/issues
- Documentation: See README.md
- Examples: Run `python -m neural.education.examples`
