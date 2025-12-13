import * as monaco from 'monaco-editor';

interface ParseError {
  line: number;
  column: number;
  message: string;
  severity: 'error' | 'warning' | 'info';
  endLine?: number;
  endColumn?: number;
}

interface ParseResult {
  success: boolean;
  errors: ParseError[];
  warnings?: ParseError[];
}

export class DiagnosticsProvider {
  private editor: monaco.editor.IStandaloneCodeEditor;
  private parserEndpoint: string;
  private validationTimeout: NodeJS.Timeout | null = null;
  private decorationsCollection: monaco.editor.IEditorDecorationsCollection;

  constructor(editor: monaco.editor.IStandaloneCodeEditor, parserEndpoint: string) {
    this.editor = editor;
    this.parserEndpoint = parserEndpoint;
    this.decorationsCollection = editor.createDecorationsCollection();
  }

  async validateModel(model: monaco.editor.ITextModel): Promise<monaco.editor.IMarker[]> {
    if (this.validationTimeout) {
      clearTimeout(this.validationTimeout);
    }

    return new Promise((resolve) => {
      this.validationTimeout = setTimeout(async () => {
        try {
          const code = model.getValue();
          
          const basicErrors = this.performBasicValidation(code);
          
          let parserErrors: ParseError[] = [];
          try {
            const parseResult = await this.callParser(code);
            if (!parseResult.success && parseResult.errors) {
              parserErrors = parseResult.errors;
            }
            if (parseResult.warnings) {
              parserErrors = parserErrors.concat(parseResult.warnings);
            }
          } catch (error) {
            console.warn('Parser endpoint not available, using basic validation only:', error);
          }

          const allErrors = [...basicErrors, ...parserErrors];
          const markers = this.convertErrorsToMarkers(allErrors, model);
          
          monaco.editor.setModelMarkers(model, 'neural-dsl', markers);
          this.updateErrorDecorations(markers);
          
          resolve(markers);
        } catch (error) {
          console.error('Validation error:', error);
          resolve([]);
        }
      }, 500);
    });
  }

  private performBasicValidation(code: string): ParseError[] {
    const errors: ParseError[] = [];
    const lines = code.split('\n');

    let inMultilineComment = false;
    let networkDepth = 0;
    let hasNetwork = false;
    let hasInput = false;
    let hasLayers = false;

    lines.forEach((line, index) => {
      const trimmed = line.trim();
      const lineNum = index + 1;

      if (trimmed.includes('/*')) inMultilineComment = true;
      if (trimmed.includes('*/')) {
        inMultilineComment = false;
        return;
      }
      if (inMultilineComment || trimmed.startsWith('//')) return;

      if (trimmed.startsWith('network ')) {
        hasNetwork = true;
        if (!trimmed.match(/network\s+[a-zA-Z_]\w*\s*\{/)) {
          errors.push({
            line: lineNum,
            column: 1,
            message: 'Invalid network declaration. Expected: network NetworkName {',
            severity: 'error'
          });
        }
      }

      if (trimmed.startsWith('input:')) {
        hasInput = true;
        if (!trimmed.match(/input:\s*(\(|\[|\{)/)) {
          errors.push({
            line: lineNum,
            column: 1,
            message: 'Invalid input declaration. Expected tuple, list, or dict.',
            severity: 'error'
          });
        }
      }

      if (trimmed.startsWith('layers:')) {
        hasLayers = true;
      }

      const openBraces = (trimmed.match(/{/g) || []).length;
      const closeBraces = (trimmed.match(/}/g) || []).length;
      networkDepth += openBraces - closeBraces;

      if (networkDepth < 0) {
        errors.push({
          line: lineNum,
          column: trimmed.indexOf('}') + 1,
          message: 'Unexpected closing brace',
          severity: 'error'
        });
        networkDepth = 0;
      }

      const unclosedString = this.checkUnclosedStrings(trimmed);
      if (unclosedString) {
        errors.push({
          line: lineNum,
          column: unclosedString.column,
          message: 'Unclosed string literal',
          severity: 'error'
        });
      }

      const mismatchedParens = this.checkMismatchedParentheses(trimmed);
      if (mismatchedParens) {
        errors.push({
          line: lineNum,
          column: 1,
          message: mismatchedParens,
          severity: 'error'
        });
      }

      if (trimmed.match(/\b(Dense|Conv2D|LSTM|GRU|Output)\s*\(\s*\)/i)) {
        errors.push({
          line: lineNum,
          column: 1,
          message: 'Layer missing required parameters',
          severity: 'warning'
        });
      }
    });

    if (networkDepth > 0) {
      errors.push({
        line: lines.length,
        column: 1,
        message: 'Unclosed brace in network definition',
        severity: 'error'
      });
    }

    if (hasNetwork && !hasInput) {
      errors.push({
        line: 1,
        column: 1,
        message: 'Network missing input definition',
        severity: 'warning'
      });
    }

    if (hasNetwork && !hasLayers) {
      errors.push({
        line: 1,
        column: 1,
        message: 'Network missing layers definition',
        severity: 'warning'
      });
    }

    return errors;
  }

  private checkUnclosedStrings(line: string): { column: number } | null {
    let inString = false;
    let stringChar = '';
    let escaped = false;

    for (let i = 0; i < line.length; i++) {
      const char = line[i];
      
      if (escaped) {
        escaped = false;
        continue;
      }

      if (char === '\\') {
        escaped = true;
        continue;
      }

      if ((char === '"' || char === "'") && !inString) {
        inString = true;
        stringChar = char;
      } else if (char === stringChar && inString) {
        inString = false;
        stringChar = '';
      }
    }

    return inString ? { column: line.indexOf(stringChar) + 1 } : null;
  }

  private checkMismatchedParentheses(line: string): string | null {
    const stack: string[] = [];
    const pairs: Record<string, string> = { '(': ')', '[': ']', '{': '}' };
    const opening = Object.keys(pairs);
    const closing = Object.values(pairs);

    for (const char of line) {
      if (opening.includes(char)) {
        stack.push(char);
      } else if (closing.includes(char)) {
        const last = stack.pop();
        if (!last || pairs[last] !== char) {
          return `Mismatched bracket: expected ${last ? pairs[last] : 'nothing'}, found ${char}`;
        }
      }
    }

    if (stack.length > 0) {
      return `Unclosed bracket: ${stack[stack.length - 1]}`;
    }

    return null;
  }

  private async callParser(code: string): Promise<ParseResult> {
    try {
      const response = await fetch(this.parserEndpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ code }),
      });

      if (!response.ok) {
        throw new Error(`Parser request failed: ${response.statusText}`);
      }

      const result = await response.json();
      return result;
    } catch (error) {
      throw error;
    }
  }

  private convertErrorsToMarkers(
    errors: ParseError[],
    model: monaco.editor.ITextModel
  ): monaco.editor.IMarker[] {
    return errors.map(error => {
      const startLine = Math.max(1, error.line);
      const startColumn = Math.max(1, error.column);
      const endLine = error.endLine || startLine;
      const endColumn = error.endColumn || model.getLineMaxColumn(endLine);

      return {
        severity: this.getSeverity(error.severity),
        message: error.message,
        startLineNumber: startLine,
        startColumn: startColumn,
        endLineNumber: endLine,
        endColumn: endColumn,
      };
    });
  }

  private getSeverity(severity: string): monaco.MarkerSeverity {
    switch (severity) {
      case 'error':
        return monaco.MarkerSeverity.Error;
      case 'warning':
        return monaco.MarkerSeverity.Warning;
      case 'info':
        return monaco.MarkerSeverity.Info;
      default:
        return monaco.MarkerSeverity.Error;
    }
  }

  private updateErrorDecorations(markers: monaco.editor.IMarker[]): void {
    const decorations = markers
      .filter(marker => marker.severity === monaco.MarkerSeverity.Error)
      .map(marker => ({
        range: new monaco.Range(
          marker.startLineNumber,
          marker.startColumn,
          marker.endLineNumber,
          marker.endColumn
        ),
        options: {
          inlineClassName: 'neural-dsl-error-inline',
          className: 'neural-dsl-error-line',
          glyphMarginClassName: 'neural-dsl-error-glyph',
          hoverMessage: { value: marker.message },
          minimap: {
            color: '#ff0000',
            position: monaco.editor.MinimapPosition.Inline
          }
        }
      }));

    this.decorationsCollection.set(decorations);
  }

  dispose(): void {
    if (this.validationTimeout) {
      clearTimeout(this.validationTimeout);
    }
    this.decorationsCollection.clear();
  }
}
