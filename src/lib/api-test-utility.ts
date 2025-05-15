import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://13.202.208.115:8000';

/**
 * Utility function to test the connection to the backend API
 */
export async function testApiConnection(): Promise<{ 
  success: boolean; 
  message: string;
  serverInfo?: any;
}> {
  try {
    console.log(`Testing connection to API at: ${API_BASE_URL}`);
    
    const response = await axios.get(`${API_BASE_URL}/`, { timeout: 5000 });
    
    if (response.status === 200) {
      console.log('API connection successful:', response.data);
      return {
        success: true,
        message: 'Successfully connected to the backend server',
        serverInfo: response.data
      };
    } else {
      console.warn('API returned non-200 response:', response.status);
      return {
        success: false,
        message: `API responded with status: ${response.status}`
      };
    }
  } catch (error) {
    console.error('API connection failed:', error);
    
    // Get more specific error information
    let errorMessage = 'Failed to connect to the backend';
    if (axios.isAxiosError(error)) {
      if (error.code === 'ECONNREFUSED') {
        errorMessage = 'Connection refused. Make sure the backend server is running.';
      } else if (error.code === 'ECONNABORTED') {
        errorMessage = 'Connection timed out. Check your network or server status.';
      } else if (error.response) {
        errorMessage = `Server responded with error: ${error.response.status} ${error.response.statusText}`;
      }
    }
    
    return {
      success: false,
      message: errorMessage
    };
  }
}

/**
 * Test the specific files endpoint
 */
export async function testFilesEndpoint(): Promise<{
  success: boolean;
  message: string;
  files?: any[];
}> {
  try {
    console.log(`Testing files endpoint at: ${API_BASE_URL}/files`);
    
    const response = await axios.get(`${API_BASE_URL}/files`, { timeout: 5000 });
    
    if (response.status === 200) {
      const files = response.data.files || response.data;
      console.log('Files endpoint successful:', files);
      return {
        success: true,
        message: `Retrieved ${Array.isArray(files) ? files.length : 0} files`,
        files: Array.isArray(files) ? files : []
      };
    } else {
      console.warn('Files endpoint returned non-200 response:', response.status);
      return {
        success: false,
        message: `Files endpoint responded with status: ${response.status}`
      };
    }
  } catch (error) {
    console.error('Files endpoint request failed:', error);
    
    let errorMessage = 'Failed to fetch files from backend';
    if (axios.isAxiosError(error)) {
      if (error.response) {
        errorMessage = `Server responded with error: ${error.response.status} ${error.response.statusText}`;
      } else if (error.request) {
        errorMessage = 'No response received from server';
      }
    }
    
    return {
      success: false,
      message: errorMessage
    };
  }
}
