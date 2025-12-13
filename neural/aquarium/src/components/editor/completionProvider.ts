import * as monaco from 'monaco-editor';

export class CompletionProvider implements monaco.languages.CompletionItemProvider {
  triggerCharacters = ['.', '(', ':', ' '];

  provideCompletionItems(
    model: monaco.editor.ITextModel,
    position: monaco.Position,
    context: monaco.languages.CompletionContext
  ): monaco.languages.ProviderResult<monaco.languages.CompletionList> {
    const textUntilPosition = model.getValueInRange({
      startLineNumber: position.lineNumber,
      startColumn: 1,
      endLineNumber: position.lineNumber,
      endColumn: position.column,
    });

    const word = model.getWordUntilPosition(position);
    const range = {
      startLineNumber: position.lineNumber,
      endLineNumber: position.lineNumber,
      startColumn: word.startColumn,
      endColumn: word.endColumn,
    };

    let suggestions: monaco.languages.CompletionItem[] = [];

    if (this.isInLayersSection(model, position)) {
      suggestions = suggestions.concat(this.getLayerSuggestions(range));
    }

    if (this.isAfterLayerType(textUntilPosition)) {
      suggestions = suggestions.concat(this.getLayerParameterSuggestions(textUntilPosition, range));
    }

    if (this.isInTopLevelContext(model, position)) {
      suggestions = suggestions.concat(this.getTopLevelKeywordSuggestions(range));
    }

    if (textUntilPosition.includes('activation')) {
      suggestions = suggestions.concat(this.getActivationSuggestions(range));
    }

    if (textUntilPosition.includes('optimizer')) {
      suggestions = suggestions.concat(this.getOptimizerSuggestions(range));
    }

    if (textUntilPosition.includes('loss')) {
      suggestions = suggestions.concat(this.getLossSuggestions(range));
    }

    if (textUntilPosition.includes('HPO(')) {
      suggestions = suggestions.concat(this.getHPOSuggestions(range));
    }

    return { suggestions };
  }

  private isInLayersSection(model: monaco.editor.ITextModel, position: monaco.Position): boolean {
    const text = model.getValue();
    const offset = model.getOffsetAt(position);
    const beforeText = text.substring(0, offset);
    
    const layersMatch = beforeText.lastIndexOf('layers:');
    if (layersMatch === -1) return false;
    
    const afterLayersText = beforeText.substring(layersMatch);
    const openBraces = (afterLayersText.match(/{/g) || []).length;
    const closeBraces = (afterLayersText.match(/}/g) || []).length;
    
    return openBraces > closeBraces;
  }

  private isAfterLayerType(textUntilPosition: string): boolean {
    const layerPattern = /(Dense|Conv1D|Conv2D|Conv3D|MaxPooling2D|LSTM|GRU|Dropout|Output)\s*\(\s*$/i;
    return layerPattern.test(textUntilPosition);
  }

  private isInTopLevelContext(model: monaco.editor.ITextModel, position: monaco.Position): boolean {
    const textUntilPosition = model.getValueInRange({
      startLineNumber: 1,
      startColumn: 1,
      endLineNumber: position.lineNumber,
      endColumn: position.column,
    });

    const networkMatch = textUntilPosition.match(/network\s+\w+\s*{/);
    if (!networkMatch) return false;

    const afterNetwork = textUntilPosition.substring(networkMatch.index! + networkMatch[0].length);
    const openBraces = (afterNetwork.match(/{/g) || []).length;
    const closeBraces = (afterNetwork.match(/}/g) || []).length;

    return openBraces === closeBraces;
  }

  private getTopLevelKeywordSuggestions(range: monaco.IRange): monaco.languages.CompletionItem[] {
    const keywords = [
      { label: 'network', detail: 'Define a neural network', insertText: 'network ${1:ModelName} {\n  $0\n}' },
      { label: 'input', detail: 'Define input shape', insertText: 'input: ${1:(None, 28, 28)}' },
      { label: 'layers', detail: 'Define network layers', insertText: 'layers:\n  $0' },
      { label: 'optimizer', detail: 'Define optimizer', insertText: 'optimizer: ${1:Adam}' },
      { label: 'loss', detail: 'Define loss function', insertText: 'loss: "${1:categorical_crossentropy}"' },
      { label: 'metrics', detail: 'Define metrics', insertText: 'metrics: ["${1:accuracy}"]' },
      { label: 'training', detail: 'Define training parameters', insertText: 'training: {\n  epochs: ${1:10},\n  batch_size: ${2:32}\n}' },
      { label: 'hpo', detail: 'Define hyperparameter optimization', insertText: 'hpo: {\n  $0\n}' },
    ];

    return keywords.map(kw => ({
      label: kw.label,
      kind: monaco.languages.CompletionItemKind.Keyword,
      detail: kw.detail,
      insertText: kw.insertText,
      insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
      range
    }));
  }

  private getLayerSuggestions(range: monaco.IRange): monaco.languages.CompletionItem[] {
    const layers = [
      { label: 'Dense', detail: 'Fully connected layer', params: 'units=${1:128}, activation="${2:relu}"' },
      { label: 'Conv2D', detail: '2D convolutional layer', params: 'filters=${1:32}, kernel_size=(${2:3}, ${3:3}), activation="${4:relu}"' },
      { label: 'Conv1D', detail: '1D convolutional layer', params: 'filters=${1:32}, kernel_size=${2:3}, activation="${3:relu}"' },
      { label: 'Conv3D', detail: '3D convolutional layer', params: 'filters=${1:32}, kernel_size=(${2:3}, ${3:3}, ${4:3}), activation="${5:relu}"' },
      { label: 'MaxPooling2D', detail: '2D max pooling layer', params: 'pool_size=(${1:2}, ${2:2})' },
      { label: 'MaxPooling1D', detail: '1D max pooling layer', params: 'pool_size=${1:2}' },
      { label: 'MaxPooling3D', detail: '3D max pooling layer', params: 'pool_size=(${1:2}, ${2:2}, ${3:2})' },
      { label: 'GlobalAveragePooling2D', detail: 'Global average pooling for 2D', params: '' },
      { label: 'GlobalAveragePooling1D', detail: 'Global average pooling for 1D', params: '' },
      { label: 'Dropout', detail: 'Dropout regularization', params: 'rate=${1:0.5}' },
      { label: 'Flatten', detail: 'Flatten input', params: '' },
      { label: 'LSTM', detail: 'Long Short-Term Memory layer', params: 'units=${1:128}, return_sequences=${2:false}' },
      { label: 'GRU', detail: 'Gated Recurrent Unit layer', params: 'units=${1:128}, return_sequences=${2:false}' },
      { label: 'SimpleRNN', detail: 'Simple recurrent layer', params: 'units=${1:128}' },
      { label: 'BatchNormalization', detail: 'Batch normalization', params: '' },
      { label: 'LayerNormalization', detail: 'Layer normalization', params: '' },
      { label: 'Activation', detail: 'Activation layer', params: 'activation="${1:relu}"' },
      { label: 'Output', detail: 'Output layer', params: 'units=${1:10}, activation="${2:softmax}"' },
      { label: 'Transformer', detail: 'Transformer layer', params: 'num_heads=${1:8}, d_model=${2:512}' },
      { label: 'TransformerEncoder', detail: 'Transformer encoder', params: 'num_heads=${1:8}, d_model=${2:512}' },
      { label: 'TransformerDecoder', detail: 'Transformer decoder', params: 'num_heads=${1:8}, d_model=${2:512}' },
      { label: 'Add', detail: 'Add layer outputs', params: '' },
      { label: 'Concatenate', detail: 'Concatenate layer outputs', params: 'axis=${1:-1}' },
      { label: 'Multiply', detail: 'Multiply layer outputs', params: '' },
    ];

    return layers.map(layer => ({
      label: layer.label,
      kind: monaco.languages.CompletionItemKind.Class,
      detail: layer.detail,
      insertText: layer.params ? `${layer.label}(${layer.params})` : `${layer.label}()`,
      insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
      range
    }));
  }

  private getLayerParameterSuggestions(textUntilPosition: string, range: monaco.IRange): monaco.languages.CompletionItem[] {
    const layerTypeMatch = textUntilPosition.match(/(Dense|Conv2D|Conv1D|LSTM|GRU|Dropout|Output)\s*\(\s*$/i);
    if (!layerTypeMatch) return [];

    const layerType = layerTypeMatch[1].toLowerCase();
    const parameterMap: Record<string, Array<{ label: string; detail: string; insertText: string }>> = {
      dense: [
        { label: 'units', detail: 'Number of neurons', insertText: 'units=${1:128}' },
        { label: 'activation', detail: 'Activation function', insertText: 'activation="${1:relu}"' },
        { label: 'use_bias', detail: 'Use bias', insertText: 'use_bias=${1:true}' },
        { label: 'kernel_initializer', detail: 'Weight initializer', insertText: 'kernel_initializer="${1:glorot_uniform}"' },
      ],
      conv2d: [
        { label: 'filters', detail: 'Number of filters', insertText: 'filters=${1:32}' },
        { label: 'kernel_size', detail: 'Size of convolution kernel', insertText: 'kernel_size=(${1:3}, ${2:3})' },
        { label: 'activation', detail: 'Activation function', insertText: 'activation="${1:relu}"' },
        { label: 'padding', detail: 'Padding mode', insertText: 'padding="${1:valid}"' },
        { label: 'strides', detail: 'Stride length', insertText: 'strides=(${1:1}, ${2:1})' },
      ],
      conv1d: [
        { label: 'filters', detail: 'Number of filters', insertText: 'filters=${1:32}' },
        { label: 'kernel_size', detail: 'Size of convolution kernel', insertText: 'kernel_size=${1:3}' },
        { label: 'activation', detail: 'Activation function', insertText: 'activation="${1:relu}"' },
        { label: 'padding', detail: 'Padding mode', insertText: 'padding="${1:valid}"' },
        { label: 'strides', detail: 'Stride length', insertText: 'strides=${1:1}' },
      ],
      lstm: [
        { label: 'units', detail: 'Number of LSTM units', insertText: 'units=${1:128}' },
        { label: 'return_sequences', detail: 'Return full sequence', insertText: 'return_sequences=${1:false}' },
        { label: 'dropout', detail: 'Dropout rate', insertText: 'dropout=${1:0.0}' },
        { label: 'recurrent_dropout', detail: 'Recurrent dropout', insertText: 'recurrent_dropout=${1:0.0}' },
      ],
      gru: [
        { label: 'units', detail: 'Number of GRU units', insertText: 'units=${1:128}' },
        { label: 'return_sequences', detail: 'Return full sequence', insertText: 'return_sequences=${1:false}' },
        { label: 'dropout', detail: 'Dropout rate', insertText: 'dropout=${1:0.0}' },
        { label: 'recurrent_dropout', detail: 'Recurrent dropout', insertText: 'recurrent_dropout=${1:0.0}' },
      ],
      dropout: [
        { label: 'rate', detail: 'Dropout rate (0-1)', insertText: 'rate=${1:0.5}' },
      ],
      output: [
        { label: 'units', detail: 'Number of output units', insertText: 'units=${1:10}' },
        { label: 'activation', detail: 'Activation function', insertText: 'activation="${1:softmax}"' },
      ],
    };

    const params = parameterMap[layerType] || [];
    return params.map(param => ({
      label: param.label,
      kind: monaco.languages.CompletionItemKind.Property,
      detail: param.detail,
      insertText: param.insertText,
      insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
      range
    }));
  }

  private getActivationSuggestions(range: monaco.IRange): monaco.languages.CompletionItem[] {
    const activations = [
      'relu', 'sigmoid', 'tanh', 'softmax', 'softplus', 'softsign',
      'selu', 'elu', 'exponential', 'leaky_relu', 'relu6', 'silu',
      'swish', 'gelu', 'hard_sigmoid', 'linear', 'mish'
    ];

    return activations.map(activation => ({
      label: activation,
      kind: monaco.languages.CompletionItemKind.Value,
      detail: `${activation} activation function`,
      insertText: `"${activation}"`,
      range
    }));
  }

  private getOptimizerSuggestions(range: monaco.IRange): monaco.languages.CompletionItem[] {
    const optimizers = [
      { label: 'Adam', params: 'learning_rate=${1:0.001}' },
      { label: 'SGD', params: 'learning_rate=${1:0.01}, momentum=${2:0.9}' },
      { label: 'RMSprop', params: 'learning_rate=${1:0.001}' },
      { label: 'Adagrad', params: 'learning_rate=${1:0.01}' },
      { label: 'Adadelta', params: 'learning_rate=${1:1.0}' },
      { label: 'Adamax', params: 'learning_rate=${1:0.002}' },
      { label: 'Nadam', params: 'learning_rate=${1:0.001}' },
    ];

    return optimizers.map(opt => ({
      label: opt.label,
      kind: monaco.languages.CompletionItemKind.Class,
      detail: `${opt.label} optimizer`,
      insertText: `${opt.label}(${opt.params})`,
      insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
      range
    }));
  }

  private getLossSuggestions(range: monaco.IRange): monaco.languages.CompletionItem[] {
    const losses = [
      'categorical_crossentropy',
      'sparse_categorical_crossentropy',
      'binary_crossentropy',
      'mean_squared_error',
      'mean_absolute_error',
      'mean_absolute_percentage_error',
      'cosine_similarity',
      'huber',
      'log_cosh',
      'kl_divergence',
      'poisson',
      'hinge',
      'squared_hinge',
    ];

    return losses.map(loss => ({
      label: loss,
      kind: monaco.languages.CompletionItemKind.Value,
      detail: `${loss} loss function`,
      insertText: `"${loss}"`,
      range
    }));
  }

  private getHPOSuggestions(range: monaco.IRange): monaco.languages.CompletionItem[] {
    const hpoTypes = [
      { label: 'range', detail: 'Continuous range', insertText: 'range(${1:0.001}, ${2:0.1})' },
      { label: 'log_range', detail: 'Logarithmic range', insertText: 'log_range(${1:0.001}, ${2:0.1})' },
      { label: 'choice', detail: 'Discrete choices', insertText: 'choice(${1:32}, ${2:64}, ${3:128})' },
      { label: 'categorical', detail: 'Categorical choices', insertText: 'categorical("${1:relu}", "${2:tanh}")' },
    ];

    return hpoTypes.map(hpo => ({
      label: hpo.label,
      kind: monaco.languages.CompletionItemKind.Function,
      detail: hpo.detail,
      insertText: hpo.insertText,
      insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
      range
    }));
  }
}
