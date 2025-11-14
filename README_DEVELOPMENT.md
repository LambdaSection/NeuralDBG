# Neural DSL Development Guide

## 🎯 Quick Navigation

- **Automation:** `AUTOMATION_GUIDE.md` or `README_AUTOMATION.md`
- **AI Integration:** `docs/ai_integration_guide.md`
- **Roadmap:** `ROADMAP.md` (internal)
- **Vision:** `VISION.md` (internal)
- **Checklist:** `CHECKLIST.md`
- **Summary:** `SUMMARY.md` or `FINAL_SUMMARY.md`

---

## 🚀 Getting Started

### For Development
1. Read `ROADMAP.md` for priorities
2. Check `CHECKLIST.md` for tasks
3. Review `VISION.md` for goals

### For Automation
1. Read `AUTOMATION_GUIDE.md`
2. Try: `python scripts/automation/master_automation.py --blog`
3. Set up GitHub Secrets for publishing

### For AI Features
1. Read `docs/ai_integration_guide.md`
2. Try: `from neural.ai.ai_assistant import NeuralAIAssistant`
3. Test with natural language commands

---

## 📁 Key Directories

- `scripts/automation/` - All automation scripts
- `.github/workflows/` - GitHub Actions
- `neural/ai/` - AI integration
- `docs/blog/` - Generated blog posts
- `docs/social/` - Social media posts

---

## 🔧 Common Tasks

### Generate Blog Posts
```bash
python scripts/automation/master_automation.py --blog
```

### Run Tests
```bash
python scripts/automation/master_automation.py --test
```

### Release New Version
```bash
python scripts/automation/master_automation.py --release --version-type patch
```

---

**For detailed information, see the specific guides listed above.**

