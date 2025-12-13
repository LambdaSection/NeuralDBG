import axios from 'axios';
import { HPOConfig, HPOStudyResults } from '../types/hpo';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5003';

export class HPOService {
  static async executeHPO(config: HPOConfig): Promise<HPOStudyResults> {
    try {
      const response = await axios.post(`${API_BASE_URL}/api/hpo/execute`, config);
      return response.data;
    } catch (error) {
      console.error('HPO execution failed:', error);
      throw error;
    }
  }

  static async streamTrials(
    config: HPOConfig,
    onTrialUpdate: (trial: any) => void,
    onComplete: (results: HPOStudyResults) => void,
    onError: (error: Error) => void
  ): Promise<void> {
    try {
      const eventSource = new EventSource(
        `${API_BASE_URL}/api/hpo/stream?config=${encodeURIComponent(JSON.stringify(config))}`
      );

      eventSource.addEventListener('trial', (event) => {
        const trial = JSON.parse(event.data);
        onTrialUpdate(trial);
      });

      eventSource.addEventListener('complete', (event) => {
        const results = JSON.parse(event.data);
        onComplete(results);
        eventSource.close();
      });

      eventSource.addEventListener('error', (event) => {
        const error = new Error('Stream error');
        onError(error);
        eventSource.close();
      });
    } catch (error) {
      onError(error as Error);
    }
  }

  static async getStudyStatus(studyId: string): Promise<any> {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/hpo/study/${studyId}`);
      return response.data;
    } catch (error) {
      console.error('Failed to get study status:', error);
      throw error;
    }
  }

  static async stopStudy(studyId: string): Promise<void> {
    try {
      await axios.post(`${API_BASE_URL}/api/hpo/study/${studyId}/stop`);
    } catch (error) {
      console.error('Failed to stop study:', error);
      throw error;
    }
  }
}

export default HPOService;
