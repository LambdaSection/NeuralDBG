export type HPOParameterType = 'range' | 'log_range' | 'choice' | 'categorical';

export interface HPOParameter {
  id: string;
  name: string;
  type: HPOParameterType;
  layerType: string;
  paramName: string;
  min?: number;
  max?: number;
  step?: number;
  values?: any[];
  logScale?: boolean;
}

export interface HPOConfig {
  parameters: HPOParameter[];
  nTrials: number;
  dataset: string;
  backend: string;
  device: string;
  dslCode: string;
}

export interface HPOTrial {
  number: number;
  values: Record<string, any>;
  value: number | number[];
  state: 'RUNNING' | 'COMPLETE' | 'PRUNED' | 'FAIL';
  datetime_start?: Date;
  datetime_complete?: Date;
  duration?: number;
}

export interface HPOStudyResults {
  best_trial: HPOTrial;
  best_params: Record<string, any>;
  trials: HPOTrial[];
}
