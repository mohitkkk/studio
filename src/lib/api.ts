import axios from 'axios';
import { ApiServiceType } from '@/hooks/use-api-service';

// Configure axios with longer timeout
const axiosInstance = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL || 'http://13.202.208.115:3000',
  timeout: 15000, // Increase timeout to 15 seconds
  headers: {
    'Content-Type': 'application/json',
  },
});

// Cache for API responses
const apiCache = new Map();

// Define types for the API
export type CacheKey = 'chatHistory' | 'files' | 'all';
export interface ChatData { id: string; title: string; messages: any[]; lastUpdated: string; }
export interface FileMetadata { filename: string; description: string; tags: string[]; }

// Track ongoing requests to prevent duplicate calls
const pendingRequests = new Map<string, Promise<any>>();

// Local chat history storage
const LOCAL_CHAT_HISTORY_KEY = "nebulaChatHistory";

// Expanded API interface to include missing methods
export const api = {
    // Add a checkStatus method
    checkStatus: async (): Promise<{ status: string; version: string }> => {
        try {
            const response = await axiosInstance.get('/files');
            return response.data;
        } catch (error) {
            console.error('Error checking API status:', error);
            throw new Error('API connection failed');
        }
    },
    
    // Updated to prioritize local storage and avoid backend calls
    getChatHistory: async (chatId?: string): Promise<ChatData[]> => {
        // Always try local storage first
        try {
            const localHistory = getLocalChatHistory();
            
            // If specific chat ID requested, filter for just that chat
            if (chatId && chatId !== 'all') {
                const specificChat = localHistory.find(chat => chat.id === chatId);
                return specificChat ? [specificChat] : [];
            }
            
            // Return all local chats
            return localHistory;
        } catch (error) {
            console.error('Error fetching local chat history:', error);
            return [];
        }
    },
    
    // Add a stable getFiles method with caching, tag filtering, and proper error handling
    getFiles: async (tag?: string): Promise<FileMetadata[]> => {
        // Create a cache key that includes the tag parameter
        const cacheKey = `files_${tag || 'all'}`;
        
        // Use cached response if available (with short TTL)
        if (apiCache.has(cacheKey)) {
            return apiCache.get(cacheKey);
        }
        
        // Check if there's already a pending request for this tag
        if (pendingRequests.has(cacheKey)) {
            return pendingRequests.get(cacheKey);
        }
        
        try {
            // Build the URL with optional tag filter
            const url = tag ? `/files?tag=${encodeURIComponent(tag)}` : '/files';
            
            // Create a promise for this request and store it
            const requestPromise = axiosInstance.get(url)
                .then(response => {
                    let files: FileMetadata[] = [];
                    
                    // Handle different response formats
                    if (response.data && Array.isArray(response.data)) {
                        files = response.data;
                    } else if (response.data && response.data.files && Array.isArray(response.data.files)) {
                        files = response.data.files;
                    }
                    
                    // Cache the response (for 30 seconds)
                    apiCache.set(cacheKey, files);
                    setTimeout(() => apiCache.delete(cacheKey), 30000);
                    
                    // Remove from pending requests
                    pendingRequests.delete(cacheKey);
                    return files;
                })
                .catch(error => {
                    console.error('Error fetching files:', error);
                    // Remove from pending requests
                    pendingRequests.delete(cacheKey);
                    // Return empty array as fallback
                    return [];
                });
            
            // Store the pending request
            pendingRequests.set(cacheKey, requestPromise);
            return requestPromise;
        } catch (error) {
            console.error('Error creating files request:', error);
            return [];
        }
    },
    
    // Method to clear cache when needed (e.g., after mutations)
    clearCache: (key?: CacheKey): void => {
        if (key) {
            apiCache.delete(key);
            pendingRequests.delete(key);
        } else {
            apiCache.clear();
            pendingRequests.clear();
        }
    },
    
    // Add the critical missing methods
    createNewChat: async () => {
      try {
        // Generate local ID first
        const localChatId = `chat_${Math.random().toString(36).substring(2, 10)}_${Date.now()}`;
        
        // Create a new chat session in local storage
        const newChat = {
          id: localChatId,
          title: "New Chat",
          messages: [],
          lastUpdated: new Date().toISOString()
        };
        
        // Save to local storage
        const localHistory = getLocalChatHistory();
        localHistory.unshift(newChat);
        saveLocalChatHistory(localHistory);
        
        // Try backend creation but don't block on it
        try {
          const baseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://13.202.208.115:3000';
          const response = await fetch(`${baseUrl}/new-chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
          });
          
          if (response.ok) {
            const data = await response.json();
            // Update local chat with backend ID if successful
            if (data && data.chat_id) {
              const updatedHistory = getLocalChatHistory();
              const index = updatedHistory.findIndex(chat => chat.id === localChatId);
              if (index >= 0) {
                updatedHistory[index].id = data.chat_id;
                saveLocalChatHistory(updatedHistory);
              }
              return data;
            }
          }
        } catch (backendError) {
          console.warn('Backend chat creation failed, using local only:', backendError);
        }
        
        return { chat_id: localChatId, status: 'local' };
      } catch (error) {
        console.error('Error creating new chat:', error);
        return { chat_id: `chat_${Date.now()}`, status: 'local' };
      }
    },
    
    sendChatMessage: async (data: any) => {
      try {
        // Update local chat history first
        updateLocalChatWithMessage(data.sessionId, {
          role: "user",
          content: data.message,
          timestamp: new Date().toISOString()
        });
        
        // Then try to send to backend
        const baseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://13.202.208.115:3000';
        const response = await fetch(`${baseUrl}/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            message: data.message,
            selected_files: data.selected_files,
            client_id: data.sessionId,
            format: data.format || "html"
          })
        });
        
        if (!response.ok) throw new Error(`API responded with ${response.status}`);
        const responseData = await response.json();
        
        // Update local chat with AI response
        if (responseData.message) {
          updateLocalChatWithMessage(data.sessionId, {
            role: "assistant",
            content: responseData.message,
            timestamp: new Date().toISOString()
          });
          
          // Update chat name if provided
          if (responseData.chat_name) {
            updateLocalChatName(data.sessionId, responseData.chat_name);
          }
        }
        
        return responseData;
      } catch (error) {
        console.error('Error sending chat message:', error);
        return { 
          message: 'Failed to send message. Please try again.',
          client_id: data.sessionId,
          status: 'error'
        };
      }
    }
};

// Helper functions for local chat storage
function getLocalChatHistory(): ChatData[] {
  try {
    if (typeof window === 'undefined') return [];
    const historyJson = localStorage.getItem(LOCAL_CHAT_HISTORY_KEY);
    return historyJson ? JSON.parse(historyJson) : [];
  } catch (error) {
    console.error("Error loading chat history from localStorage:", error);
    return [];
  }
}

function saveLocalChatHistory(history: ChatData[]): void {
  try {
    if (typeof window === 'undefined') return;
    localStorage.setItem(LOCAL_CHAT_HISTORY_KEY, JSON.stringify(history));
  } catch (error) {
    console.error("Error saving chat history to localStorage:", error);
  }
}

function updateLocalChatWithMessage(chatId: string, message: any): void {
  try {
    const history = getLocalChatHistory();
    const chatIndex = history.findIndex(chat => chat.id === chatId);
    
    if (chatIndex >= 0) {
      // Add message to existing chat
      history[chatIndex].messages.push(message);
      history[chatIndex].lastUpdated = new Date().toISOString();
    } else {
      // Create new chat with this message
      history.unshift({
        id: chatId,
        title: "New Chat",
        messages: [message],
        lastUpdated: new Date().toISOString()
      });
    }
    
    saveLocalChatHistory(history);
  } catch (error) {
    console.error("Error updating local chat with message:", error);
  }
}

function updateLocalChatName(chatId: string, newName: string): void {
  try {
    const history = getLocalChatHistory();
    const chatIndex = history.findIndex(chat => chat.id === chatId);
    
    if (chatIndex >= 0) {
      history[chatIndex].title = newName;
      saveLocalChatHistory(history);
    }
  } catch (error) {
    console.error("Error updating chat name:", error);
  }
}

// Initialize the API integration
export function initializeApiIntegration(apiService: ApiServiceType): void {
  // Attach the API service methods to the api object
  Object.entries(apiService).forEach(([key, method]) => {
    if (typeof method === 'function' && !api[key as keyof typeof api]) {
      (api as any)[key] = method;
    }
  });
  
  console.log('API integration initialized with methods:', Object.keys(api));
}

// Alternative approach: Re-export the axios instance in case other modules need direct access
export { axiosInstance };