import { useState, useEffect, useRef } from 'react';
import type { ChatMessage, ChatSession } from '@/types';
import axios from 'axios';

// Connect to the FastAPI backend
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://13.202.208.115:8000';

// Track API errors by type for better error reporting
const ERROR_TRACKING = {
  ollama_errors: 0,
  network_errors: 0,
  other_errors: 0
};

// Caching and rate-limiting settings
const CACHE_DURATION = 30000; // 30 seconds
const REQUEST_THROTTLE = 2000; // 2 seconds between identical requests

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

export function useApiService(props?: ApiServiceProps) {
  const baseUrl = props?.baseUrl || API_BASE_URL;
  const [isInitialized, setIsInitialized] = useState(false);
  
  // Cache and request tracking
  const cache = useRef<{[key: string]: {data: any, timestamp: number}}>({});
  const pendingRequests = useRef<{[key: string]: boolean}>({});
  const lastRequestTime = useRef<{[key: string]: number}>({});
  
  useEffect(() => {
    console.log(`API service initializing with baseUrl: ${baseUrl}`);
    axios.defaults.baseURL = baseUrl;
    
    // Add default headers
    axios.defaults.headers.common['Content-Type'] = 'application/json';
    axios.defaults.headers.common['Accept'] = 'application/json';
    
    // Add any additional headers
    if (props?.headers) {
      Object.entries(props.headers).forEach(([key, value]) => {
        axios.defaults.headers.common[key] = value;
      });
    }
    
    // Add global error handler
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
    
    setIsInitialized(true);
    
    return () => {
      // Clean up interceptors on unmount
      axios.interceptors.response.eject(0);
    };
  }, [baseUrl, props?.headers]);

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
    if (pendingRequests.current[cacheKey]) {
      console.log(`Request to ${endpoint} already pending, waiting...`);
      await new Promise(resolve => setTimeout(resolve, 300));
      return makeRequest(endpoint, method, data, options);
    }
    
    // Check cache for GET requests
    if (method === 'get' && !bypassCache && cache.current[cacheKey]) {
      const cachedData = cache.current[cacheKey];
      if (now - cachedData.timestamp < CACHE_DURATION) {
        console.log(`Using cached response for ${endpoint}`);
        return cachedData.data as T;
      }
    }
    
    // Check rate limiting
    const lastTime = lastRequestTime.current[cacheKey] || 0;
    if (now - lastTime < REQUEST_THROTTLE) {
      const delay = REQUEST_THROTTLE - (now - lastTime);
      console.log(`Rate limiting request to ${endpoint}, waiting ${delay}ms`);
      await new Promise(resolve => setTimeout(resolve, delay));
    }
    
    // Mark request as pending and update last request time
    pendingRequests.current[cacheKey] = true;
    lastRequestTime.current[cacheKey] = now;
    
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
        cache.current[cacheKey] = {
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
      pendingRequests.current[cacheKey] = false;
    }
  };

  // Status check
  const checkStatus = async () => {
    return makeRequest<{version: string, status: string}>(
      '/', 'get', undefined, { timeoutMs: 5000 }
    );
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
      const response = await makeRequest<any[]>('/chats', 'get');
      
      return response.map((chat: any) => ({
        id: chat.chat_id,
        name: chat.chat_name,
        messages: [],
        lastModified: new Date(chat.updated_at).getTime(),
        selectedFiles: chat.selected_files || []
      }));
    } catch (error) {
      console.error("Error fetching chat sessions:", error);
      throw error;
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
    cache.current = {};
    console.log('API cache cleared');
  };

  // Update file selection that works with or without the endpoint
  const updateFileSelection = async (params: {
    sessionId: string;
    files: string[];
  }): Promise<any> => {
    try {
      console.log(`Sending file selection update: ${params.files.length} files for session ${params.sessionId}`);
      
      // First try to use chat endpoint with empty message - most reliable method
      try {
        const response = await axios.post('/chat', {
          message: " ", // Space character to avoid empty message errors
          client_id: params.sessionId,
          selected_files: params.files
        }, {
          timeout: 5000 // Short timeout
        });
        
        console.log("File selection via chat endpoint succeeded");
        return { success: true, method: "chat" };
      } catch (chatError) {
        console.warn("Chat endpoint for file selection failed, trying dedicated endpoint:", chatError);
        
        // If chat method fails, try dedicated endpoint
        try {
          const response = await axios.post('/file-selection', {
            session_id: params.sessionId,
            selected_files: params.files,
            timestamp: Date.now()
          }, {
            timeout: 5000 // Short timeout
          });
          
          console.log("Dedicated file-selection endpoint succeeded");
          return { success: true, method: "file-selection" };
        } catch (endpointError) {
          // Both methods failed - just return success anyway to not block UI
          console.warn("All file selection methods failed, using local-only mode");
          return { 
            success: true, 
            method: "local-only",
            warning: "Could not sync file selection with server"
          };
        }
      }
    } catch (error: any) {
      // Return a successful result anyway to avoid disrupting the UI
      // The important part is that files will be sent with each chat message
      console.error("Error in file selection update:", error);
      return { success: true, method: "error-handled", warning: "Error in update, but proceeding" };
    }
  };

  return {
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
}
