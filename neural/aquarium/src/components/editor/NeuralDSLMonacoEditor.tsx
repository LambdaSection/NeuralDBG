import React, { useEffect, useRef, useState } from 'react';
import * as monaco from 'monaco-editor';
import { NeuralDSLLanguageConfig } from './languageConfig';
import { NeuralDSLTheme } from './theme';
import { DiagnosticsProvider } from './diagnosticsProvider';
import { CompletionProvider } from './completionProvider';

interface NeuralDSLMonacoEditorProps {
  value?: string;
  onChange?: (value: string) => void;
  onValidation?: (errors: monaco.editor.IMarker[]) => void;
  height?: string;
  theme?: 'light' | 'dark';
  readOnly?: boolean;
  parserEndpoint?: string;
}

export const NeuralDSLMonacoEditor: React.FC<NeuralDSLMonacoEditorProps> = ({
  value = '',
  onChange,
  onValidation,
  height = '600px',
  theme = 'dark',
  readOnly = false,
  parserEndpoint = '/api/parse'
}) => {
  const editorRef = useRef<monaco.editor.IStandaloneCodeEditor | null>(null);
  const monacoRef = useRef<typeof monaco | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const diagnosticsProviderRef = useRef<DiagnosticsProvider | null>(null);
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    if (!containerRef.current) return;

    monacoRef.current = monaco;

    const languageId = 'neural-dsl';
    
    if (!monaco.languages.getLanguages().some(lang => lang.id === languageId)) {
      monaco.languages.register({ id: languageId });
      
      monaco.languages.setMonarchTokensProvider(
        languageId,
        NeuralDSLLanguageConfig.monarchLanguage
      );
      
      monaco.languages.setLanguageConfiguration(
        languageId,
        NeuralDSLLanguageConfig.languageConfiguration
      );
      
      monaco.editor.defineTheme('neural-dsl-dark', NeuralDSLTheme.dark);
      monaco.editor.defineTheme('neural-dsl-light', NeuralDSLTheme.light);
      
      const completionProvider = new CompletionProvider();
      monaco.languages.registerCompletionItemProvider(
        languageId,
        completionProvider
      );
      
      monaco.languages.registerDocumentFormattingEditProvider(languageId, {
        provideDocumentFormattingEdits: (model) => {
          const formatted = formatNeuralDSL(model.getValue());
          return [{
            range: model.getFullModelRange(),
            text: formatted
          }];
        }
      });
    }

    const editor = monaco.editor.create(containerRef.current, {
      value,
      language: languageId,
      theme: theme === 'dark' ? 'neural-dsl-dark' : 'neural-dsl-light',
      automaticLayout: true,
      minimap: { enabled: true },
      scrollBeyondLastLine: false,
      fontSize: 14,
      lineNumbers: 'on',
      roundedSelection: false,
      readOnly,
      cursorStyle: 'line',
      wordWrap: 'off',
      folding: true,
      foldingStrategy: 'indentation',
      showFoldingControls: 'always',
      matchBrackets: 'always',
      bracketPairColorization: {
        enabled: true
      },
      suggest: {
        showKeywords: true,
        showSnippets: true,
        showClasses: true,
        showFunctions: true
      },
      quickSuggestions: {
        other: true,
        comments: false,
        strings: false
      },
      parameterHints: {
        enabled: true
      }
    });

    editorRef.current = editor;
    diagnosticsProviderRef.current = new DiagnosticsProvider(editor, parserEndpoint);

    const model = editor.getModel();
    if (model) {
      diagnosticsProviderRef.current.validateModel(model).then((markers) => {
        if (onValidation) {
          onValidation(markers);
        }
      });
    }

    editor.onDidChangeModelContent(() => {
      const currentValue = editor.getValue();
      if (onChange) {
        onChange(currentValue);
      }
      
      const currentModel = editor.getModel();
      if (currentModel && diagnosticsProviderRef.current) {
        diagnosticsProviderRef.current.validateModel(currentModel).then((markers) => {
          if (onValidation) {
            onValidation(markers);
          }
        });
      }
    });

    setIsReady(true);

    return () => {
      if (diagnosticsProviderRef.current) {
        diagnosticsProviderRef.current.dispose();
      }
      editor.dispose();
    };
  }, []);

  useEffect(() => {
    if (editorRef.current && isReady) {
      const currentValue = editorRef.current.getValue();
      if (value !== currentValue) {
        editorRef.current.setValue(value);
      }
    }
  }, [value, isReady]);

  useEffect(() => {
    if (editorRef.current && monacoRef.current) {
      const themeName = theme === 'dark' ? 'neural-dsl-dark' : 'neural-dsl-light';
      monacoRef.current.editor.setTheme(themeName);
    }
  }, [theme]);

  useEffect(() => {
    if (editorRef.current) {
      editorRef.current.updateOptions({ readOnly });
    }
  }, [readOnly]);

  return (
    <div 
      ref={containerRef} 
      style={{ 
        height, 
        width: '100%',
        border: '1px solid #444'
      }} 
    />
  );
};

function formatNeuralDSL(code: string): string {
  const lines = code.split('\n');
  let indentLevel = 0;
  const indentSize = 2;
  const formattedLines: string[] = [];

  for (let line of lines) {
    const trimmed = line.trim();
    
    if (trimmed.length === 0) {
      formattedLines.push('');
      continue;
    }

    if (trimmed.startsWith('//') || trimmed.startsWith('/*') || trimmed.startsWith('*')) {
      formattedLines.push(' '.repeat(indentLevel * indentSize) + trimmed);
      continue;
    }

    if (trimmed === '}') {
      indentLevel = Math.max(0, indentLevel - 1);
    }

    formattedLines.push(' '.repeat(indentLevel * indentSize) + trimmed);

    if (trimmed.endsWith('{')) {
      indentLevel++;
    } else if (trimmed === '}') {
      // Already decremented above
    }
  }

  return formattedLines.join('\n');
}
