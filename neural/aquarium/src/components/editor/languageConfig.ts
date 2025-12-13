import * as monaco from 'monaco-editor';

export const NeuralDSLLanguageConfig = {
  monarchLanguage: {
    defaultToken: '',
    tokenPostfix: '.neural',

    keywords: [
      'network', 'input', 'layers', 'optimizer', 'loss', 'metrics', 
      'training', 'train', 'hpo', 'execution', 'HPO'
    ],

    layerTypes: [
      'Dense', 'Conv1D', 'Conv2D', 'Conv3D', 'Conv2DTranspose',
      'MaxPooling1D', 'MaxPooling2D', 'MaxPooling3D',
      'GlobalAveragePooling1D', 'GlobalAveragePooling2D',
      'Dropout', 'Flatten', 'LSTM', 'GRU', 'SimpleRNN',
      'SimpleRNNDropoutWrapper', 'LSTMCell', 'GRUCell',
      'Output', 'Transformer', 'TransformerEncoder', 'TransformerDecoder',
      'BatchNormalization', 'LayerNormalization', 'InstanceNormalization',
      'GroupNormalization', 'GaussianNoise',
      'Activation', 'Add', 'Subtract', 'Multiply', 'Average', 
      'Maximum', 'Concatenate', 'Dot',
      'TimeDistributed', 'ResidualConnection'
    ],

    activationFunctions: [
      'relu', 'sigmoid', 'tanh', 'softmax', 'softplus', 'softsign',
      'selu', 'elu', 'exponential', 'leaky_relu', 'relu6', 'silu',
      'swish', 'gelu', 'hard_sigmoid', 'linear', 'mish'
    ],

    optimizers: [
      'Adam', 'SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adamax', 'Nadam', 'Ftrl'
    ],

    lossFunctions: [
      'categorical_crossentropy', 'sparse_categorical_crossentropy',
      'binary_crossentropy', 'mean_squared_error', 'mean_absolute_error',
      'mean_absolute_percentage_error', 'mean_squared_logarithmic_error',
      'cosine_similarity', 'huber', 'log_cosh', 'kl_divergence',
      'poisson', 'hinge', 'squared_hinge', 'categorical_hinge'
    ],

    hpoTypes: [
      'range', 'log_range', 'choice', 'categorical'
    ],

    learningRateSchedules: [
      'ExponentialDecay', 'StepDecay', 'PolynomialDecay', 
      'CosineDecay', 'WarmupCosineDecay'
    ],

    booleans: ['true', 'false'],
    nullValue: ['none', 'None', 'null'],

    operators: ['=', ':', ',', '*', '@'],

    symbols: /[=><!~?:&|+\-*\/\^%@]+/,
    escapes: /\\(?:[abfnrtv\\"']|x[0-9A-Fa-f]{1,4}|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8})/,

    tokenizer: {
      root: [
        // Comments
        [/\/\/.*$/, 'comment'],
        [/\/\*/, 'comment', '@comment'],

        // Layer types (case-insensitive)
        [/\b(Dense|Conv1D|Conv2D|Conv3D|Conv2DTranspose|MaxPooling1D|MaxPooling2D|MaxPooling3D|GlobalAveragePooling1D|GlobalAveragePooling2D|Dropout|Flatten|LSTM|GRU|SimpleRNN|SimpleRNNDropoutWrapper|LSTMCell|GRUCell|Output|Transformer|TransformerEncoder|TransformerDecoder|BatchNormalization|LayerNormalization|InstanceNormalization|GroupNormalization|GaussianNoise|Activation|Add|Subtract|Multiply|Average|Maximum|Concatenate|Dot|TimeDistributed|ResidualConnection)\b/i, 'type.layer'],

        // Keywords
        [/\b(network|input|layers|optimizer|loss|metrics|training|train|hpo|execution|HPO)\b/, 'keyword'],

        // Optimizers
        [/\b(Adam|SGD|RMSprop|Adagrad|Adadelta|Adamax|Nadam|Ftrl)\b/, 'type.optimizer'],

        // Learning rate schedules
        [/\b(ExponentialDecay|StepDecay|PolynomialDecay|CosineDecay|WarmupCosineDecay)\b/, 'type.schedule'],

        // HPO types
        [/\b(range|log_range|choice|categorical)\b/, 'keyword.hpo'],

        // Booleans and null
        [/\b(true|false)\b/i, 'keyword.boolean'],
        [/\b(none|None|null)\b/, 'keyword.null'],

        // Identifiers for layer names and parameters
        [/[a-zA-Z_]\w*_layer\b/, 'identifier.layer-name'],
        [/[a-zA-Z_]\w*(?=\s*:)/, 'identifier.parameter'],
        [/[a-zA-Z_]\w*/, 'identifier'],

        // Strings
        [/"([^"\\]|\\.)*$/, 'string.invalid'],
        [/'([^'\\]|\\.)*$/, 'string.invalid'],
        [/"/, 'string', '@string_double'],
        [/'/, 'string', '@string_single'],

        // Numbers
        [/[+-]?\d+\.\d+([eE][+-]?\d+)?/, 'number.float'],
        [/[+-]?\d+[eE][+-]?\d+/, 'number.float'],
        [/[+-]?\d+/, 'number'],

        // Delimiters and operators
        [/[{}()\[\]]/, '@brackets'],
        [/@/, 'operator.device'],
        [/[,:]/, 'delimiter'],
        [/\*(?=\s*\d+)/, 'operator.multiply'],
        [/=/, 'operator'],
      ],

      comment: [
        [/[^\/*]+/, 'comment'],
        [/\*\//, 'comment', '@pop'],
        [/[\/*]/, 'comment']
      ],

      string_double: [
        [/[^\\"]+/, 'string'],
        [/@escapes/, 'string.escape'],
        [/\\./, 'string.escape.invalid'],
        [/"/, 'string', '@pop']
      ],

      string_single: [
        [/[^\\']+/, 'string'],
        [/@escapes/, 'string.escape'],
        [/\\./, 'string.escape.invalid'],
        [/'/, 'string', '@pop']
      ],
    },
  } as monaco.languages.IMonarchLanguage,

  languageConfiguration: {
    comments: {
      lineComment: '//',
      blockComment: ['/*', '*/']
    },
    brackets: [
      ['{', '}'],
      ['[', ']'],
      ['(', ')']
    ],
    autoClosingPairs: [
      { open: '{', close: '}' },
      { open: '[', close: ']' },
      { open: '(', close: ')' },
      { open: '"', close: '"', notIn: ['string'] },
      { open: "'", close: "'", notIn: ['string', 'comment'] }
    ],
    surroundingPairs: [
      { open: '{', close: '}' },
      { open: '[', close: ']' },
      { open: '(', close: ')' },
      { open: '"', close: '"' },
      { open: "'", close: "'" }
    ],
    folding: {
      markers: {
        start: new RegExp('^\\s*//\\s*#?region\\b'),
        end: new RegExp('^\\s*//\\s*#?endregion\\b')
      },
      offSide: false
    },
    wordPattern: /[a-zA-Z_]\w*/,
    indentationRules: {
      increaseIndentPattern: new RegExp('^.*\\{[^}]*$'),
      decreaseIndentPattern: new RegExp('^\\s*\\}.*$')
    }
  } as monaco.languages.LanguageConfiguration
};
