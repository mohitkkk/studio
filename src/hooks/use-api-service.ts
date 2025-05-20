import { useEffect, useRef } from 'react';
import type { ChatMessage, ChatSession } from '@/types';
import axios from 'axios';

// Connect to the FastAPI backend
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://13.202.208.115:3000';

// Track API errors by type for better error reporting
const ERROR_TRACKING = {
  ollama_errors: 0,
  network_errors: 0,
  other_errors: 0
};

// Caching and rate-limiting settings
const CACHE_DURATION = 30000; // 30 seconds
const REQUEST_THROTTLE = 2000; // 2 seconds between identical requests

// Create a singleton instance of the API service to avoid hook-related issues
let apiServiceInstance: ReturnType<typeof createApiService> | null = null;

interface ApiServiceProps {
  baseUrl?: string;
  headers?: Record<string, string>;
}

interface FileMetadata {
  filename: string;
  description: string;
  tags: string[];
  added_date: string;
  updated_date: string;
  suggested_questions?: Array<{question: string}>;
}

type ProgressCallback = (progress: number) => void;

// Create a non-hook version of the API service for use outside of components
function createApiService(props?: ApiServiceProps) {
  console.log("Creating API service instance");
  
  const baseUrl = props?.baseUrl || API_BASE_URL;
  
  // Configure axios defaults
  axios.defaults.baseURL = baseUrl;
  axios.defaults.headers.common['Content-Type'] = 'application/json';
  axios.defaults.headers.common['Accept'] = 'application/json';
  
  // Add any additional headers
  if (props?.headers) {
    Object.entries(props.headers).forEach(([key, value]) => {
      axios.defaults.headers.common[key] = value;
    });
  }
  
  // Add global error handler if not already added
  // Use a module-level variable to track if we've added the interceptor
  let responseInterceptorAdded = false;
  
  if (!responseInterceptorAdded) {
    axios.interceptors.response.use(
      response => response, 
      error => {
        // Log the error but let calling code handle it
        if (error.response) {
          console.error(`API Error ${error.response.status}: ${error.response.statusText}`, error.response.data);
        } else if (error.request) {
          console.error('API Error: No response received', error.request);
        } else {
          console.error('API Error:', error.message);
        }
        return Promise.reject(error);
      }
    );
    responseInterceptorAdded = true;
  }
  
  // Cache and request tracking - using simple objects instead of refs
  const cache: {[key: string]: {data: any, timestamp: number}} = {};
  const pendingRequests: {[key: string]: boolean} = {};
  const lastRequestTime: {[key: string]: number} = {};

  // Helper for making requests with caching and rate limiting
  const makeRequest = async <T>(
    endpoint: string, 
    method: 'get' | 'post' | 'put' | 'delete' = 'get',
    data?: any,
    options: { 
      cacheKey?: string;
      bypassCache?: boolean;
      timeoutMs?: number;
    } = {}
  ): Promise<T> => {
    const {
      cacheKey = endpoint,
      bypassCache = false,
      timeoutMs = 10000,
    } = options;
    
    const now = Date.now();
    
    // Check if request is already pending
    if (pendingRequests[cacheKey]) {
      console.log(`Request to ${endpoint} already pending, waiting...`);
      await new Promise(resolve => setTimeout(resolve, 300));
      return makeRequest(endpoint, method, data, options);
    }
    
    // Check cache for GET requests
    if (method === 'get' && !bypassCache && cache[cacheKey]) {
      const cachedData = cache[cacheKey];
      if (now - cachedData.timestamp < CACHE_DURATION) {
        console.log(`Using cached response for ${endpoint}`);
        return cachedData.data as T;
      }
    }
    
    // Check rate limiting
    const lastTime = lastRequestTime[cacheKey] || 0;
    if (now - lastTime < REQUEST_THROTTLE) {
      const delay = REQUEST_THROTTLE - (now - lastTime);
      console.log(`Rate limiting request to ${endpoint}, waiting ${delay}ms`);
      await new Promise(resolve => setTimeout(resolve, delay));
    }
    
    // Mark request as pending and update last request time
    pendingRequests[cacheKey] = true;
    lastRequestTime[cacheKey] = now;
    
    try {
      // Make the actual request
      let response;
      
      const requestConfig = {
        timeout: timeoutMs,
        headers: { 'Cache-Control': 'no-cache' }
      };
      
      if (method === 'get') {
        response = await axios.get(endpoint, requestConfig);
      } else if (method === 'post') {
        response = await axios.post(endpoint, data, requestConfig);
      } else if (method === 'put') {
        response = await axios.put(endpoint, data, requestConfig);
      } else {
        response = await axios.delete(endpoint, { ...requestConfig, data });
      }
      
      // Cache GET responses
      if (method === 'get' && !bypassCache) {
        cache[cacheKey] = {
          data: response.data,
          timestamp: now
        };
      }
      
      return response.data as T;
    } catch (error) {
      console.error(`Error in ${method.toUpperCase()} request to ${endpoint}:`, error);
      throw error;
    } finally {
      // Clear pending flag
      pendingRequests[cacheKey] = false;
    }
  };

  // Status check
  const checkStatus = async () => {
    try {
      const response = await axios.get('/', { timeout: 5000 });
      return {
        version: response.data?.version || 'unknown',
        status: 'connected'
      };
    } catch (error) {
      console.error("Error checking API status:", error);
      return {
        version: 'unknown',
        status: 'disconnected'
      };
    }
  };

  // Get configuration
  const getConfig = async () => {
    try {
      return await makeRequest<any>('/', 'get');
    } catch (error) {
      console.error("Unable to fetch server config:", error);
      return { version: 'unknown', status: 'offline' };
    }
  };

  // Get files from the /files endpoint
  const getFiles = async (tag?: string) => {
    const endpoint = tag ? `/files?tag=${encodeURIComponent(tag)}` : '/files';
    return makeRequest<any>(endpoint, 'get', undefined, {
      cacheKey: `files-${tag || 'all'}`,
      timeoutMs: 15000 // Longer timeout for files loading
    });
  };

  // Send a message to chat with better error handling for empty messages
  const sendChatMessage = async (params: {
    sessionId?: string;
    message: string;
    selected_files?: string[];
  }): Promise<any> => {
    try {
      // Ensure message is never empty by using a space if it's blank
      const messageToSend = params.message.trim() || " ";
      
      console.log(`Sending chat message to backend: ${messageToSend.substring(0, 50)}...`);
      
      const response = await axios.post('/chat', {
        message: messageToSend,
        client_id: params.sessionId,
        selected_files: params.selected_files || []
      });
      
      if (ERROR_TRACKING.ollama_errors > 0 || ERROR_TRACKING.network_errors > 0) {
        // Reset error counters on successful request
        ERROR_TRACKING.ollama_errors = 0;
        ERROR_TRACKING.network_errors = 0;
        console.log("API connection restored - resetting error counters");
      }
      
      return response.data;
    } catch (error: any) {
      console.error("Error sending chat message:", error);
      
      // Check for known Ollama errors
      let errorMessage = error?.response?.data?.detail || error.message || "Unknown error";
      let errorType = 'other';
      
      if (errorMessage.includes('ollama') || errorMessage.includes('Ollama') || 
          errorMessage.includes('completions') || errorMessage.includes('LLM')) {
        errorType = 'ollama';
        ERROR_TRACKING.ollama_errors++;
        
        throw new Error(`Ollama LLM error (${ERROR_TRACKING.ollama_errors}): ${errorMessage}`);
      } else if (error.code === 'ECONNABORTED' || error.code === 'ECONNREFUSED' || !error.response) {
        errorType = 'network';
        ERROR_TRACKING.network_errors++;
        
        throw new Error(`Network error (${ERROR_TRACKING.network_errors}): Could not connect to API server`);
      } else {
        ERROR_TRACKING.other_errors++;
        throw new Error(`API error: ${errorMessage}`);
      }
    }
  };

  // Upload a file
  const uploadFile = async (file: File, onProgress: ProgressCallback) => {
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('description', `Uploaded on ${new Date().toLocaleString()}`);
      
      const response = await axios.post('/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / (progressEvent.total || 100));
          onProgress(percentCompleted);
        },
        timeout: 60000 // 60 seconds timeout for uploads
      });
      
      return {
        fileId: response.data.filename || file.name,
        fileName: file.name,
        fileSize: file.size,
        fileType: file.type
      };
    } catch (error) {
      console.error("Error uploading file:", error);
      throw error;
    }
  };

  // Process document - combines the document processing step with upload
  const processDocument = async (params: {fileId: string; fileName: string}) => {
    return makeRequest<any>(
      `/process/${params.fileId}`, 'post', { fileName: params.fileName },
      { bypassCache: true }
    );
  };

  // Get chat sessions from the /chats endpoint
  const getChatSessions = async (): Promise<ChatSession[]> => {
    try {
      const response = await axios.get('/chats');
      
      if (!response.data) {
        return [];
      }
      
      const sessions = Array.isArray(response.data) ? response.data : [];
      
      return sessions.map((chat: any) => ({
        id: chat.chat_id || chat.id || `chat_${Date.now()}`,
        name: chat.chat_name || chat.name || "Untitled Chat",
        messages: chat.messages || [],
        lastModified: new Date(chat.updated_at || chat.lastModified || Date.now()).getTime(),
        selectedFiles: chat.selected_files || chat.selectedFiles || []
      }));
    } catch (error) {
      console.error("Error fetching chat sessions:", error);
      return []; // Return empty array on error
    }
  };

  // Create a new chat session
  const createChatSession = async (session: ChatSession): Promise<ChatSession> => {
    // The backend doesn't have a direct create session endpoint,
    // it creates a session when first message is sent
    return session;
  };

  // Update a chat session
  const updateChatSession = async (session: ChatSession): Promise<ChatSession> => {
    try {
      // The backend might not have explicit chat session management
      // Just return the session object - we'll rely on sending the data with each message
      console.log("Chat session updates are handled implicitly through messages");
      return session;
    } catch (error) {
      console.error("Error updating chat session:", error);
      return session; // Return the session anyway to allow local updates
    }
  };

  // Remove a file - delete file from server
  const removeFile = async (fileId: string): Promise<void> => {
    await makeRequest<any>(`/files/${fileId}`, 'delete', undefined, { bypassCache: true });
  };

  // Clear API cache - useful for debugging or forcing fresh data
  const clearCache = () => {
    // ...existing code...
  };

  // Update file selection that works with or without the endpoint
  const updateFileSelection = async (params: {
    sessionId: string;
    files: string[];
  }): Promise<any> => {
    // ...existing code...
  };

  // Debug log the returned object
  const service = {
    checkStatus,
    getConfig,
    getFiles,
    uploadFile,
    processDocument,
    getChatSessions,
    createChatSession,
    updateChatSession,
    sendChatMessage,
    removeFile,
    clearCache,
    updateFileSelection,
  };
  
  console.log("API service methods:", Object.keys(service));
  return service;
}

// Create the service without hooks - modify this to not depend on hooks during initialization
export const api = (() => {
  if (typeof window === 'undefined') {
    // Server-side rendering handling
    return {} as ReturnType<typeof createApiService>;
  }
  
  if (!apiServiceInstance) {
    apiServiceInstance = createApiService();
  }
  return apiServiceInstance;
})();  

// Improved hook implementation
export function useApiService(props?: ApiServiceProps) {
  const isMounted = useRef(false);
  
  // Initialize or update the service on component mount
  useEffect(() => {
    if (!apiServiceInstance) {
      apiServiceInstance = createApiService(props);
    }
    
    isMounted.current = true;
    
    return () => {
      isMounted.current = false;
    };
  }, [props?.baseUrl, props?.headers]);
  
  // Return the singleton instance
  return apiServiceInstance || createApiService(props);
}
