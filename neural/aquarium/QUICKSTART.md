# Neural Aquarium - Quick Start Guide

Get up and running with Neural Aquarium in 5 minutes!

## Prerequisites

- **Node.js** 16+ and npm ([Download](https://nodejs.org/))
- **Python** 3.8+ ([Download](https://python.org/))
- **Neural DSL** installed (from repo root: `pip install -e .`)

## Installation

### Step 1: Install Frontend Dependencies

```bash
cd neural/aquarium
npm install
```

### Step 2: Install Backend Dependencies

```bash
cd neural/aquarium/backend
pip install -r requirements.txt
```

## Running the Application

### Option A: Automated Start (Recommended)

**Linux/Mac:**
```bash
cd neural/aquarium
chmod +x start-dev.sh
./start-dev.sh
```

**Windows:**
```cmd
cd neural\aquarium
start-dev.bat
```

This will start both frontend and backend automatically.

### Option B: Manual Start

**Terminal 1 - Backend:**
```bash
cd neural/aquarium/backend
python api.py
```

**Terminal 2 - Frontend:**
```bash
cd neural/aquarium
npm start
```

## Access the Application

Open your browser and navigate to:
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:5000

## First Steps

1. **Open the AI Assistant**: Look for the sidebar on the right side of the screen

2. **Try a Simple Command**: Type in the chat:
   ```
   Create a CNN for MNIST classification with 10 classes
   ```

3. **Review the Generated DSL**: The DSL code will appear in the code viewer below the chat

4. **Edit if Needed**: Click the "Edit" button to modify the code

5. **Apply to Workspace**: Click "Apply" to use the model

6. **Continue Refining**: Add more layers or modify settings through conversation:
   ```
   Add a dense layer with 256 units
   Add dropout with rate 0.5
   Set optimizer to Adam with learning rate 0.001
   ```

## Example Session

```
You: Create a CNN for MNIST
Assistant: I've created a CNN model for you...

[DSL code appears]

You: Add dropout with rate 0.3
Assistant: Added layer: Dropout(0.3)

You: Download the code
[Click Download button in DSL viewer]

You: Apply the model
[Click Apply button]
```

## Common Commands

### Create Models
- "Create a CNN for image classification"
- "Create an LSTM for text processing"
- "Create a model for CIFAR-10"

### Add Layers
- "Add a convolutional layer with 64 filters"
- "Add a dense layer with 128 units"
- "Add dropout with rate 0.5"
- "Add max pooling with pool size 2"
- "Add batch normalization"

### Configure Training
- "Set optimizer to Adam"
- "Set learning rate to 0.001"
- "Set loss to categorical crossentropy"

### Other Commands
- "Reset conversation" - Start over
- "Show current model" - Display full DSL
- "Visualize architecture" - (coming soon)

## Language Support

1. Click the language dropdown at the top of the sidebar
2. Select your preferred language (12+ supported)
3. Type in your language - responses will be translated
4. DSL code remains in standard syntax

## Troubleshooting

### Backend won't start
```bash
# Check Python installation
python --version

# Install dependencies again
cd neural/aquarium/backend
pip install -r requirements.txt

# Check if port 5000 is available
netstat -an | grep 5000
```

### Frontend won't start
```bash
# Check Node.js installation
node --version
npm --version

# Clear cache and reinstall
cd neural/aquarium
rm -rf node_modules package-lock.json
npm install

# Try starting again
npm start
```

### Connection Error
- Ensure backend is running on port 5000
- Check CORS settings if using different ports
- Verify firewall isn't blocking connections

### "I didn't understand that"
- Be more specific in your prompts
- Use technical terms (e.g., "convolutional layer" not just "layer")
- Try example prompts from the suggestions

## Next Steps

- Read [README.md](./README.md) for detailed features
- Check [EXAMPLES.md](./EXAMPLES.md) for more usage examples
- See [ARCHITECTURE.md](./ARCHITECTURE.md) to understand the system
- Review [DEPLOYMENT.md](./DEPLOYMENT.md) for production setup

## Getting Help

- Check the console for error messages
- Review backend logs for API errors
- Consult the main Neural DSL documentation
- Report issues on GitHub

## Quick Reference

| Action | Command |
|--------|---------|
| Start backend | `python neural/aquarium/backend/api.py` |
| Start frontend | `npm start` (from neural/aquarium) |
| Build frontend | `npm run build` |
| Run tests | `npm test` |
| Install deps | `npm install` |
| Backend health | `curl http://localhost:5000/health` |

## Development Tips

1. **Keep both terminals open** to see logs in real-time
2. **Use browser DevTools** to inspect network requests
3. **Check backend logs** for AI processing details
4. **Save your DSL code** frequently using the Download button
5. **Experiment with prompts** to find what works best

## What's Next?

Now that you're up and running:

1. Experiment with different model architectures
2. Try multi-language support
3. Export your models and use with Neural CLI
4. Build more complex networks iteratively
5. Share your DSL files with your team

Happy building! 🚀
