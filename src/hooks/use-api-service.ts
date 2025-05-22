import axios, { AxiosProgressEvent } from 'axios';
import { useRef, useState, useEffect } from 'react';

// Define API service types
export interface ApiServiceProps {
  baseUrl?: string;
  headers?: Record<string, string>;
}

// Export this type for better type safety
export type ApiServiceType = {
  checkStatus: () => Promise<any>;
  getFiles: () => Promise<any>;
  getChatHistory: (chatId: string) => Promise<any>;
  createChatSession: () => Promise<any>;
  sendChatMessage: (params: any) => Promise<any>;
  updateFileSelection: (params: any) => Promise<any>;
  clearCache: () => Promise<any>;
  uploadFile: (file: File, onProgress?: (progress: number) => void) => Promise<any>;
  processDocument: (params: any) => Promise<any>;
};

export interface FileUploadResponse {
  fileId: string;
  fileName: string;
  status?: string;
}

export interface ProcessDocumentResponse {
  summary: string | null;
  fileId: string;
  status: string;
}

export interface ChatResponse {
  message: string;
  all_messages?: any[];
  chat_name?: string;
  client_id: string;
  status: string;
}

// Setup default base URL with better fallback options
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 
                      'http://13.202.208.115:3000'; // Updated port to match server config port 3000

// Store attempted URLs to avoid repetitive failures

// Use singleton pattern for API service with props tracking
let apiServiceInstance: ApiServiceType | null = null;
let lastProps: ApiServiceProps | undefined;

// Add immediate fallback methods to guarantee availability
const FALLBACK_METHODS = {
  checkStatus: async () => ({ status: 'error', message: 'Method not implemented' }),
  getFiles: async () => ({ files: [], count: 0 }),
  getChatHistory: async () => ({ messages: [] }),
  createChatSession: async () => ({ chat_id: `fallback_${Date.now()}` }),
  sendChatMessage: async (params: any) => ({
    message: 'API service is initializing. Please try again in a moment.',
    status: 'error',
    client_id: params?.sessionId || 'fallback'
  }),
  updateFileSelection: async () => ({ status: 'error', message: 'Using global fallback' }),
  clearCache: async () => ({ success: false }),
  uploadFile: async () => ({ 
    fileId: 'error', 
    fileName: 'error', 
    status: 'error' 
  }),
  processDocument: async () => ({ 
    summary: null, 
    fileId: 'error', 
    status: 'error' 
  })
};

// Create and initialize a global fallback instance immediately to ensure it's always available
const GLOBAL_FALLBACK_INSTANCE: ApiServiceType = {
  checkStatus: async () => ({ status: 'error', message: 'Using global fallback' }),
  getFiles: async () => ({ files: [], count: 0 }),
  getChatHistory: async () => ({ messages: [] }),
  createChatSession: async () => ({ chat_id: `fallback_${Date.now()}` }),
  sendChatMessage: async (params: any) => ({
    message: 'API service is initializing. Please try again in a moment.',
    status: 'error',
    client_id: params?.sessionId || 'fallback'
  }),
  updateFileSelection: async () => ({ status: 'error', message: 'Using global fallback' }),
  clearCache: async () => ({ success: false }),
  uploadFile: async () => ({ 
    fileId: 'error', 
    fileName: 'error', 
    status: 'error' 
  }),
  processDocument: async () => ({ 
    summary: null, 
    fileId: 'error', 
    status: 'error' 
  })
};

// Initialize the singleton immediately
apiServiceInstance = GLOBAL_FALLBACK_INSTANCE;

/**
 * Helper function to ensure all methods exist
 * This creates fallbacks for any missing methods
 */
export function ensureApiServiceMethods(service: any): ApiServiceType {
  if (!service) {
    console.warn("Null service provided to ensureApiServiceMethods, using global fallback");
    return GLOBAL_FALLBACK_INSTANCE;
  }
  
  // Make a copy to avoid modifying original object
  const completedService = { ...service };
  
  // CRITICAL: Check for sendChatMessage first and most explicitly
  if (typeof completedService.sendChatMessage !== 'function') {
    console.warn("🔴 Critical method sendChatMessage missing - adding fallback implementation");
    completedService.sendChatMessage = FALLBACK_METHODS.sendChatMessage;
  }
  
  // Then check other methods
  for (const [key, func] of Object.entries(FALLBACK_METHODS)) {
    if (typeof completedService[key] !== 'function') {
      console.warn(`Adding missing API method: ${key}`);
      completedService[key] = func;
    }
  }
  
  // Double-check that sendChatMessage is properly set
  if (typeof completedService.sendChatMessage !== 'function') {
    console.error("🚨 CRITICAL: sendChatMessage still not available after fixes!");
    // Force add it again as a last resort
    completedService.sendChatMessage = FALLBACK_METHODS.sendChatMessage;
  }
  
  return completedService as ApiServiceType;
}

/**
 * Create API service with standard methods
 */
export function createApiService(props?: ApiServiceProps): ApiServiceType {
  try {
    const baseUrl = props?.baseUrl || API_BASE_URL;
    console.log("Creating API service with base URL:", baseUrl);
    
    // CRITICAL: Start with a complete set of fallback methods
    const service = { ...FALLBACK_METHODS };
    
    // Now override with real implementations, keeping fallbacks if something fails
    service.sendChatMessage = async (data: {
      message: string;
      selected_files: string[];
      sessionId: string;
      format?: string;
    }): Promise<ChatResponse> => {
      try {
        console.log(`Sending chat message to ${baseUrl}/chat with data:`, {
          message: data.message.substring(0, 50) + (data.message.length > 50 ? '...' : ''),
          selected_files_count: data.selected_files.length,
          sessionId: data.sessionId,
          format: data.format || "html"
        });
        
        // Add safety checks
        if (!data.message) {
          throw new Error("Message cannot be empty");
        }
        
        if (!data.sessionId) {
          throw new Error("Session ID is required");
        }
        
        const response = await axios.post(`${baseUrl}/chat`, {
          message: data.message,
          selected_files: data.selected_files,
          client_id: data.sessionId,
          format: data.format || "html"
        }, {
          timeout: 30000,
          headers: {
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache',
            'Expires': '0',
          }
        });
        
        console.log("Chat response received:", response.status, response.statusText);
        return response.data;
      } catch (error) {
        console.error("Error sending chat message:", error);
        // Rethrow with more details
        if (axios.isAxiosError(error)) {
          if (error.response) {
            throw new Error(`Server error: ${error.response.status} ${error.response.statusText}`);
          } else if (error.request) {
            throw new Error(`No response from server. Check your network connection.`);
          } else {
            throw new Error(`Request failed: ${error.message}`);
          }
        }
        throw error;
      }
    };
    
    // Add other methods after the critical one
    service.checkStatus = async () => {
      try {
        console.log(`Checking backend status at ${baseUrl}/files`);
        const response = await axios.get(`${baseUrl}/files`, { 
          timeout: 5000,
          // Add cache busting parameter to avoid cached responses
          params: { _t: Date.now() }
        });
        console.log("API connection successful, files found:", response.data.count || 0);
        return {
          status: 'connected',
          filesCount: response.data.count || 0,
          message: 'Successfully connected to backend'
        };
      } catch (error: any) {
        console.error(`API connection failed to ${baseUrl}:`, error.message || error);
        
        // Provide more detailed error
        if (error.code === 'ECONNREFUSED') {
          throw new Error(`API server not running at ${baseUrl}`);
        } else if (error.response) {
          throw new Error(`API responded with status ${error.response.status}: ${error.response.statusText}`);
        } else if (error.request) {
          throw new Error(`No response from API server at ${baseUrl}`);
        }
        throw new Error(`API connection failed: ${error.message || "Unknown error"}`);
      }
    };
    
    service.getFiles = async (tag?: string) => {
      try {
        const url = tag ? `${baseUrl}/files?tag=${encodeURIComponent(tag)}` : `${baseUrl}/files`;
        console.log("Fetching files from:", url);
        const response = await axios.get(url, { timeout: 8000 }); // Increased timeout
        
        // Enhanced debug logging
        console.log("Files API raw response status:", response.status);
        console.log("Files API response data:", response.data);
        
        // Handle different API response formats and normalize to consistent format
        let files = [];
        let count = 0;
        
        // Case 1: Response is {files: [...], count: number}
        if (response.data && typeof response.data === 'object' && Array.isArray(response.data.files)) {
          files = response.data.files;
          count = response.data.count || files.length;
          console.log("Got files from response.data.files array:", files.length);
        }
        // Case 2: Response is an array directly
        else if (Array.isArray(response.data)) {
          files = response.data;
          count = files.length;
          console.log("Response data is a direct array:", files.length);
        }
        // Case 3: Response has files nested deeper
        else if (response.data && typeof response.data === 'object') {
          // Try to find files array in any property of the response
          for (const key in response.data) {
            if (Array.isArray(response.data[key]) && 
                response.data[key].length > 0 && 
                typeof response.data[key][0] === 'object' &&
                response.data[key][0].filename) {
              files = response.data[key];
              count = files.length;
              console.log(`Found files array in property '${key}':`, files.length);
              break;
            }
          }
          
          if (files.length === 0) {
            console.warn("Couldn't find files array in response object:", response.data);
          }
        }
        
        // Return both the original response (for debugging) and the normalized structure
        return {
          files: files,
          count: count,
          originalResponse: response.data // Include original for debugging
        };
      } catch (error: any) {
        console.error("Error fetching files:", error.message || error);
        // Return empty array on error instead of throwing
        return { files: [], count: 0 };
      }
    };
    
    // Get chat history with improved error handling
    service.getChatHistory = async (chatId: string) => {
      if (!chatId) {
        console.warn("getChatHistory called with empty chatId");
        return { messages: [] }; // Return empty result instead of throwing
      }
      
      try {
        console.log(`Fetching chat history for ${chatId}`);
        const response = await axios.get(`${baseUrl}/chat/${chatId}`, {
          timeout: 8000, // Increased timeout for potentially larger payload
          headers: {
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache'
          }
        });
        
        // Validate response data before returning
        if (response.data && typeof response.data === 'object') {
          return response.data;
        } else {
          console.warn(`Invalid chat history response format for ${chatId}`);
          return { messages: [] }; // Return standardized empty result
        }
      } catch (error) {
        console.error("Error fetching chat history:", error);
        // Return empty result instead of throwing to avoid breaking UI
        return { 
          messages: [],
          error: error instanceof Error ? error.message : "Unknown error" 
        };
      }
    };
    
    // Create new chat session with improved error handling
    service.createChatSession = async () => {
      try {
        console.log("Creating new chat session");
        const response = await axios.post(`${baseUrl}/new-chat`, {}, {
          timeout: 5000,
          headers: {
            'Content-Type': 'application/json',
            'Cache-Control': 'no-cache'
          }
        });
        
        // Validate the response contains at least a chat_id
        if (response.data && response.data.chat_id) {
          console.log(`New chat session created: ${response.data.chat_id}`);
          return response.data;
        } else {
          console.warn("Invalid response from new-chat endpoint:", response.data);
          // Return a fallback response with a client-side generated ID
          return { 
            chat_id: `local_${Date.now()}`, 
            chat_name: "New Chat",
            status: "created_locally" 
          };
        }
      } catch (error) {
        console.error("Error creating chat session:", error);
        // Return a fallback response instead of throwing
        return { 
          chat_id: `local_${Date.now()}`, 
          chat_name: "Offline Chat",
          status: "error",
          error: error instanceof Error ? error.message : "Unknown error"
        };
      }
    };
    
    // Update file selection
    service.updateFileSelection = async (params: {
      sessionId: string;
      files: string[];
    }) => {
      try {
        const response = await axios.post(`${baseUrl}/file-selection`, {
          session_id: params.sessionId,
          selected_files: params.files,
          timestamp: Date.now()
        });
        return response.data;
      } catch (error) {
        console.error("Error updating file selection:", error);
        throw error;
      }
    };
    
    // Clear cache (utility function)
    service.clearCache = async () => {
      console.log("Cache cleared");
      return { success: true };
    };
    
    // File upload function
    service.uploadFile = async (file: File, progressCallback?: (progress: number) => void): Promise<FileUploadResponse> => {
      try {
        // Create FormData object
        const formData = new FormData();
        formData.append('file', file);
        
        // Configure axios with progress tracking
        const config = {
          onUploadProgress: (progressEvent: AxiosProgressEvent) => {
            if (progressCallback && progressEvent.total) {
              const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
              progressCallback(percentCompleted);
            }
          }
        };
        
        // Send the file to backend
        const response = await axios.post(`${baseUrl}/upload`, formData, config);
        
        return {
          fileId: response.data.filename, // Backend returns the stored filename 
          fileName: response.data.filename,
          status: response.data.status
        };
      } catch (error) {
        console.error("Error uploading file:", error);
        throw error;
      }
    };
    
    // Process document function
    service.processDocument = async (params: { fileId: string, fileName: string }): Promise<ProcessDocumentResponse> => {
      try {
        const response = await axios.post(`${baseUrl}/process`, {
          filename: params.fileName
        });
        
        return {
          summary: response.data.summary || null,
          fileId: params.fileId,
          status: response.data.status
        };
      } catch (error) {
        console.error("Error processing document:", error);
        throw error;
      }
    };
    
    // Double-check sendChatMessage exists before returning
    if (typeof service.sendChatMessage !== 'function') {
      console.error("🚨 CRITICAL: sendChatMessage still missing after service creation!");
      service.sendChatMessage = FALLBACK_METHODS.sendChatMessage;
    }
    
    // Log the methods that exist in the service
    console.log('API service created with methods:', Object.keys(service));
    console.log('sendChatMessage exists:', typeof service.sendChatMessage === 'function');

    // Ensure we never return a service without the critical methods
    return ensureApiServiceMethods(service);
  } catch (e) {
    console.error('Error creating API service:', e);
    // Return fallback service if creation fails
    return GLOBAL_FALLBACK_INSTANCE;
  }
}

/**
 * Hook to get the API service
 */
export function useApiService(props?: ApiServiceProps): ApiServiceType {
  // Initialize with the global fallback first to ensure it's never null
  const [service, setService] = useState<ApiServiceType>(GLOBAL_FALLBACK_INSTANCE);
  
  // Keep a ref to avoid stale closures in the useEffect cleanup
  const serviceRef = useRef<ApiServiceType>(GLOBAL_FALLBACK_INSTANCE);
  
  // Initialize the service only once on component mount
  useEffect(() => {
    let isMounted = true;
    let initializationTimeout: NodeJS.Timeout | null = null;
    
    const initializeService = async () => {
      try {
        let newService: ApiServiceType;
        
        // If we already have a service instance with the critical method, use it
        if (apiServiceInstance && typeof apiServiceInstance.sendChatMessage === 'function') {
          console.log("Using existing API service instance with verified sendChatMessage");
          newService = apiServiceInstance;
        } else {
          // Otherwise create a new one
          console.log("Creating new API service instance in hook");
          newService = createApiService(props);
          
          // Verify it has the critical method
          if (typeof newService.sendChatMessage !== 'function') {
            console.error("Created service missing sendChatMessage method, using fallback!");
            newService = ensureApiServiceMethods(newService);
          }
          
          // Store for future components
          apiServiceInstance = newService;
        }
        
        // Keep local ref updated
        serviceRef.current = newService;
        
        // Update state with the verified service only if component still mounted
        if (isMounted) {
          setService(newService);
        }
      } catch (e) {
        console.error("Error initializing API service:", e);
        // Keep using fallback on error (already set in initial state)
        
        // Try again after a delay (only if mounted)
        if (isMounted) {
          initializationTimeout = setTimeout(initializeService, 2000);
        }
      }
    };

    // Start initialization
    initializeService();
    
    // Cleanup function
    return () => {
      isMounted = false;
      if (initializationTimeout) {
        clearTimeout(initializationTimeout);
      }
    };
  }, [props]); // Only recreate if props change

  // Return the reference value to avoid stale closures
  return serviceRef.current;
}

/**
 * Export a standalone function to get the API service instance
 */
export function getApiService(props?: ApiServiceProps): ApiServiceType {
  if (!apiServiceInstance || typeof apiServiceInstance.sendChatMessage !== 'function') {
    console.log("Creating new API service instance in getter");
    apiServiceInstance = createApiService(props);
  }
  return apiServiceInstance;
}

// Set global API service for non-hook usage
let globalApiService: ApiServiceType = GLOBAL_FALLBACK_INSTANCE;

export function setGlobalApiService(service: ApiServiceType): void {
  globalApiService = ensureApiServiceMethods(service);
  console.log("Global API service set with methods:", Object.keys(globalApiService));
}

export function getGlobalApiService(): ApiServiceType {
  return globalApiService;
}

// Try to initialize the API service immediately
try {
  if (!apiServiceInstance || typeof apiServiceInstance.sendChatMessage !== 'function') {
    console.log("Pre-initializing API service");
    apiServiceInstance = createApiService();
  }
} catch (e) {
  console.error("Failed to pre-initialize API service:", e);
  // We already have the fallback set, so no further action needed
}
