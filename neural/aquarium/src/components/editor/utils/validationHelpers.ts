export interface ValidationRule {
  pattern: RegExp;
  message: string;
  severity: 'error' | 'warning' | 'info';
}

export const validationRules: ValidationRule[] = [
  {
    pattern: /network\s+\w+\s*\{[^}]*\}/g,
    message: 'Network definition should contain input and layers sections',
    severity: 'warning'
  },
  {
    pattern: /Dense\([^)]*\)/gi,
    message: 'Dense layer should specify units parameter',
    severity: 'warning'
  },
  {
    pattern: /Conv2D\([^)]*\)/gi,
    message: 'Conv2D layer should specify filters and kernel_size',
    severity: 'warning'
  },
  {
    pattern: /Dropout\(rate\s*=\s*([0-9.]+)\)/gi,
    message: 'Dropout rate should be between 0 and 1',
    severity: 'error'
  },
];

export function validateDropoutRate(value: number): boolean {
  return value >= 0 && value <= 1;
}

export function validateUnits(value: number): boolean {
  return Number.isInteger(value) && value > 0;
}

export function validateKernelSize(value: number | [number, number] | [number, number, number]): boolean {
  if (typeof value === 'number') {
    return value > 0;
  }
  if (Array.isArray(value)) {
    return value.every(v => v > 0);
  }
  return false;
}

export function validateLearningRate(value: number): boolean {
  return value > 0 && value < 1;
}

export function validateEpochs(value: number): boolean {
  return Number.isInteger(value) && value > 0;
}

export function validateBatchSize(value: number): boolean {
  return Number.isInteger(value) && value > 0 && value <= 1024;
}

export function extractLayerParameters(layerText: string): Record<string, any> {
  const params: Record<string, any> = {};
  const paramPattern = /(\w+)\s*[=:]\s*([^,)]+)/g;
  let match;
  
  while ((match = paramPattern.exec(layerText)) !== null) {
    const [, key, value] = match;
    params[key] = value.trim();
  }
  
  return params;
}

export function validateLayerDefinition(layerType: string, parameters: Record<string, any>): string[] {
  const errors: string[] = [];
  
  switch (layerType.toLowerCase()) {
    case 'dense':
      if (!parameters.units) {
        errors.push('Dense layer requires "units" parameter');
      } else if (!validateUnits(parseInt(parameters.units))) {
        errors.push('Dense layer "units" must be a positive integer');
      }
      break;
      
    case 'conv2d':
      if (!parameters.filters) {
        errors.push('Conv2D layer requires "filters" parameter');
      }
      if (!parameters.kernel_size) {
        errors.push('Conv2D layer requires "kernel_size" parameter');
      }
      break;
      
    case 'dropout':
      if (!parameters.rate) {
        errors.push('Dropout layer requires "rate" parameter');
      } else {
        const rate = parseFloat(parameters.rate);
        if (!validateDropoutRate(rate)) {
          errors.push('Dropout "rate" must be between 0 and 1');
        }
      }
      break;
      
    case 'lstm':
    case 'gru':
      if (!parameters.units) {
        errors.push(`${layerType} layer requires "units" parameter`);
      }
      break;
      
    case 'output':
      if (!parameters.units) {
        errors.push('Output layer requires "units" parameter');
      }
      if (!parameters.activation) {
        errors.push('Output layer should specify activation function');
      }
      break;
  }
  
  return errors;
}

export function findUnmatchedBrackets(code: string): Array<{ line: number; column: number; bracket: string }> {
  const stack: Array<{ bracket: string; line: number; column: number }> = [];
  const unmatched: Array<{ line: number; column: number; bracket: string }> = [];
  const pairs: Record<string, string> = { '(': ')', '[': ']', '{': '}' };
  const opening = Object.keys(pairs);
  const closing = Object.values(pairs);
  
  const lines = code.split('\n');
  
  for (let lineIdx = 0; lineIdx < lines.length; lineIdx++) {
    const line = lines[lineIdx];
    let inString = false;
    let stringChar = '';
    
    for (let colIdx = 0; colIdx < line.length; colIdx++) {
      const char = line[colIdx];
      
      if ((char === '"' || char === "'") && (colIdx === 0 || line[colIdx - 1] !== '\\')) {
        if (!inString) {
          inString = true;
          stringChar = char;
        } else if (char === stringChar) {
          inString = false;
        }
        continue;
      }
      
      if (inString) continue;
      
      if (opening.includes(char)) {
        stack.push({ bracket: char, line: lineIdx + 1, column: colIdx + 1 });
      } else if (closing.includes(char)) {
        if (stack.length === 0) {
          unmatched.push({ line: lineIdx + 1, column: colIdx + 1, bracket: char });
        } else {
          const last = stack[stack.length - 1];
          if (pairs[last.bracket] === char) {
            stack.pop();
          } else {
            unmatched.push({ line: lineIdx + 1, column: colIdx + 1, bracket: char });
          }
        }
      }
    }
  }
  
  return [...unmatched, ...stack.map(item => ({ 
    line: item.line, 
    column: item.column, 
    bracket: item.bracket 
  }))];
}

export function checkIndentation(code: string): Array<{ line: number; expected: number; actual: number }> {
  const lines = code.split('\n');
  const issues: Array<{ line: number; expected: number; actual: number }> = [];
  let expectedIndent = 0;
  const indentSize = 2;
  
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmed = line.trim();
    
    if (trimmed.length === 0 || trimmed.startsWith('//')) {
      continue;
    }
    
    const actualIndent = line.length - line.trimStart().length;
    
    if (trimmed === '}') {
      expectedIndent = Math.max(0, expectedIndent - indentSize);
    }
    
    if (actualIndent !== expectedIndent) {
      issues.push({
        line: i + 1,
        expected: expectedIndent,
        actual: actualIndent
      });
    }
    
    if (trimmed.endsWith('{')) {
      expectedIndent += indentSize;
    } else if (trimmed === '}') {
      // Already decremented above
    }
  }
  
  return issues;
}
