export interface Snippet {
  label: string;
  description: string;
  insertText: string;
  category: 'network' | 'layer' | 'optimizer' | 'hpo' | 'training';
}

export const neuralDSLSnippets: Snippet[] = [
  {
    label: 'network-basic',
    description: 'Basic neural network template',
    category: 'network',
    insertText: `network \${1:ModelName} {
  input: (\${2:None}, \${3:28}, \${4:28})
  layers:
    \${5:Dense(units=128, activation="relu")}
    \${6:Output(units=10, activation="softmax")}
  loss: "\${7:categorical_crossentropy}"
  optimizer: \${8:Adam}
}`
  },
  {
    label: 'network-cnn',
    description: 'Convolutional neural network template',
    category: 'network',
    insertText: `network \${1:CNNModel} {
  input: (\${2:28}, \${3:28}, \${4:1})
  layers:
    Conv2D(filters=\${5:32}, kernel_size=(\${6:3}, \${7:3}), activation="relu")
    MaxPooling2D(pool_size=(\${8:2}, \${9:2}))
    Conv2D(filters=\${10:64}, kernel_size=(\${11:3}, \${12:3}), activation="relu")
    MaxPooling2D(pool_size=(\${13:2}, \${14:2}))
    Flatten()
    Dense(units=\${15:128}, activation="relu")
    Dropout(rate=\${16:0.5})
    Output(units=\${17:10}, activation="softmax")
  loss: "sparse_categorical_crossentropy"
  optimizer: Adam(learning_rate=0.001)
  metrics: ["accuracy"]
}`
  },
  {
    label: 'network-rnn',
    description: 'Recurrent neural network template',
    category: 'network',
    insertText: `network \${1:RNNModel} {
  input: (\${2:None}, \${3:100}, \${4:50})
  layers:
    LSTM(units=\${5:128}, return_sequences=\${6:true})
    LSTM(units=\${7:64})
    Dense(units=\${8:64}, activation="relu")
    Dropout(rate=\${9:0.3})
    Output(units=\${10:1}, activation="sigmoid")
  loss: "binary_crossentropy"
  optimizer: Adam(learning_rate=0.001)
  metrics: ["accuracy"]
}`
  },
  {
    label: 'network-transformer',
    description: 'Transformer network template',
    category: 'network',
    insertText: `network \${1:TransformerModel} {
  input: (\${2:None}, \${3:512})
  layers:
    Transformer(num_heads=\${4:8}, d_model=\${5:512}, num_layers=\${6:6})
    Dense(units=\${7:256}, activation="relu")
    Dropout(rate=\${8:0.1})
    Output(units=\${9:1000}, activation="softmax")
  loss: "categorical_crossentropy"
  optimizer: Adam(learning_rate=0.0001)
  metrics: ["accuracy", "top_k_categorical_accuracy"]
}`
  },
  {
    label: 'network-with-hpo',
    description: 'Network with hyperparameter optimization',
    category: 'hpo',
    insertText: `network \${1:HPOModel} {
  input: (\${2:28}, \${3:28}, \${4:1})
  layers:
    Conv2D(
      filters=HPO(choice(\${5:32}, \${6:64}, \${7:128})),
      kernel_size=(\${8:3}, \${9:3}),
      activation="relu"
    )
    MaxPooling2D(pool_size=(\${10:2}, \${11:2}))
    Flatten()
    Dense(
      units=HPO(range(\${12:64}, \${13:256})),
      activation=HPO(categorical("relu", "tanh", "selu"))
    )
    Dropout(rate=HPO(range(\${14:0.2}, \${15:0.5})))
    Output(units=\${16:10}, activation="softmax")
  loss: "categorical_crossentropy"
  optimizer: Adam(learning_rate=HPO(log_range(\${17:0.0001}, \${18:0.01})))
  metrics: ["accuracy"]
  hpo: {
    n_trials: \${19:100},
    direction: "\${20:maximize}",
    metric: "\${21:val_accuracy}"
  }
}`
  },
  {
    label: 'resnet-block',
    description: 'Residual connection block',
    category: 'layer',
    insertText: `ResidualConnection: {
  Conv2D(filters=\${1:64}, kernel_size=(\${2:3}, \${3:3}), activation="relu")
  BatchNormalization()
  Conv2D(filters=\${4:64}, kernel_size=(\${5:3}, \${6:3}), activation="relu")
  BatchNormalization()
}`
  },
  {
    label: 'inception-module',
    description: 'Inception module pattern',
    category: 'layer',
    insertText: `branch1: {
  Conv2D(filters=\${1:64}, kernel_size=(1, 1), activation="relu")
}
branch2: {
  Conv2D(filters=\${2:96}, kernel_size=(1, 1), activation="relu")
  Conv2D(filters=\${3:128}, kernel_size=(3, 3), activation="relu")
}
branch3: {
  Conv2D(filters=\${4:16}, kernel_size=(1, 1), activation="relu")
  Conv2D(filters=\${5:32}, kernel_size=(5, 5), activation="relu")
}
branch4: {
  MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")
  Conv2D(filters=\${6:32}, kernel_size=(1, 1), activation="relu")
}
Concatenate(axis=-1)`
  },
  {
    label: 'training-config',
    description: 'Training configuration',
    category: 'training',
    insertText: `training: {
  epochs: \${1:10},
  batch_size: \${2:32},
  validation_split: \${3:0.2},
  shuffle: \${4:true}
}`
  },
  {
    label: 'optimizer-adam',
    description: 'Adam optimizer with parameters',
    category: 'optimizer',
    insertText: `Adam(
  learning_rate=\${1:0.001},
  beta_1=\${2:0.9},
  beta_2=\${3:0.999},
  epsilon=\${4:1e-7}
)`
  },
  {
    label: 'optimizer-sgd',
    description: 'SGD optimizer with momentum',
    category: 'optimizer',
    insertText: `SGD(
  learning_rate=\${1:0.01},
  momentum=\${2:0.9},
  nesterov=\${3:true}
)`
  },
  {
    label: 'hpo-range',
    description: 'HPO continuous range',
    category: 'hpo',
    insertText: `HPO(range(\${1:min}, \${2:max}))`
  },
  {
    label: 'hpo-log-range',
    description: 'HPO logarithmic range',
    category: 'hpo',
    insertText: `HPO(log_range(\${1:0.0001}, \${2:0.1}))`
  },
  {
    label: 'hpo-choice',
    description: 'HPO discrete choices',
    category: 'hpo',
    insertText: `HPO(choice(\${1:32}, \${2:64}, \${3:128}))`
  },
  {
    label: 'hpo-categorical',
    description: 'HPO categorical choices',
    category: 'hpo',
    insertText: `HPO(categorical("\${1:relu}", "\${2:tanh}", "\${3:selu}"))`
  },
  {
    label: 'layer-conv2d',
    description: 'Convolutional 2D layer',
    category: 'layer',
    insertText: `Conv2D(filters=\${1:32}, kernel_size=(\${2:3}, \${3:3}), activation="\${4:relu}", padding="\${5:valid}")`
  },
  {
    label: 'layer-lstm',
    description: 'LSTM layer',
    category: 'layer',
    insertText: `LSTM(units=\${1:128}, return_sequences=\${2:false}, dropout=\${3:0.0}, recurrent_dropout=\${4:0.0})`
  },
  {
    label: 'layer-dense',
    description: 'Dense layer',
    category: 'layer',
    insertText: `Dense(units=\${1:128}, activation="\${2:relu}", use_bias=\${3:true})`
  },
  {
    label: 'layer-batch-norm',
    description: 'Batch normalization layer',
    category: 'layer',
    insertText: `BatchNormalization()`
  },
  {
    label: 'layer-dropout',
    description: 'Dropout layer',
    category: 'layer',
    insertText: `Dropout(rate=\${1:0.5})`
  },
  {
    label: 'device-specification',
    description: 'Specify device for layer',
    category: 'layer',
    insertText: `\${1:Dense(units=128)} @"\${2:GPU:0}"`
  },
  {
    label: 'layer-multiplication',
    description: 'Repeat layer multiple times',
    category: 'layer',
    insertText: `\${1:Dense(units=128, activation="relu")} * \${2:3}`
  },
];

export function getSnippetsByCategory(category: Snippet['category']): Snippet[] {
  return neuralDSLSnippets.filter(snippet => snippet.category === category);
}

export function getSnippetByLabel(label: string): Snippet | undefined {
  return neuralDSLSnippets.find(snippet => snippet.label === label);
}

export function getAllSnippets(): Snippet[] {
  return neuralDSLSnippets;
}
