import React, { useState } from 'react';
import AIAssistantSidebar from './components/ai/AIAssistantSidebar';
import './App.css';

function App() {
  const [currentDSL, setCurrentDSL] = useState<string>('');
  const [appliedDSL, setAppliedDSL] = useState<string>('');

  const handleDSLGenerated = (dslCode: string) => {
    setCurrentDSL(dslCode);
    console.log('DSL Generated:', dslCode);
  };

  const handleDSLApplied = (dslCode: string) => {
    setAppliedDSL(dslCode);
    console.log('DSL Applied:', dslCode);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Neural Aquarium</h1>
        <p>AI-Powered Neural DSL Builder</p>
      </header>

      <main className="App-main">
        <div className="content-area">
          <div className="model-workspace">
            <h2>Model Workspace</h2>
            {appliedDSL ? (
              <div className="applied-model">
                <h3>Applied Model</h3>
                <pre className="model-code">{appliedDSL}</pre>
              </div>
            ) : (
              <div className="placeholder">
                <p>Use the AI Assistant to create a neural network model.</p>
                <p>Your generated DSL code will appear here once applied.</p>
              </div>
            )}
          </div>
        </div>
      </main>

      <AIAssistantSidebar
        onDSLGenerated={handleDSLGenerated}
        onDSLApplied={handleDSLApplied}
      />
    </div>
  );
}

export default App;
