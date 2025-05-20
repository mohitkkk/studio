import axios from 'axios';

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
interface ChatData {
    // ChatData structure - adjust properties based on actual data structure
    id: string;
    title?: string;
    messages?: any[];
    createdAt?: string;
    updatedAt?: string;
    [key: string]: any; // For other properties that might exist
}

// File metadata interface
interface FileMetadata {
    filename: string;
    description?: string;
    tags?: string[];
    added_date?: string;
    updated_date?: string;
    suggested_questions?: Array<{question: string}>;
    [key: string]: any;
}

type CacheKey = string;

// Track ongoing requests to prevent duplicate calls
const pendingRequests = new Map<string, Promise<any>>();

export const api = {
    // Add a checkStatus method
    checkStatus: async (): Promise<{ status: string; version: string }> => {
        try {
            const response = await axiosInstance.get('/status');
            return response.data;
        } catch (error) {
            console.error('Error checking API status:', error);
            throw new Error('API connection failed');
        }
    },
    
    // Keep only one method for getting chat history to avoid confusion
    getChatHistory: async (): Promise<ChatData[]> => {
        // Use cached response if available
        if (apiCache.has('chatHistory')) {
            return apiCache.get('chatHistory');
        }
        
        // Check if there's already a pending request
        if (pendingRequests.has('chatHistory')) {
            return pendingRequests.get('chatHistory');
        }
        
        try {
            // Create a promise for this request and store it
            const requestPromise = axiosInstance.get('/chats')
                .then(response => {
                    const data: ChatData[] = response.data;
                    // Cache the response
                    apiCache.set('chatHistory', data);
                    // Remove from pending requests
                    pendingRequests.delete('chatHistory');
                    return data;
                })
                .catch(error => {
                    console.error('Error fetching chat history:', error);
                    // Remove from pending requests
                    pendingRequests.delete('chatHistory');
                    // Return empty array as fallback
                    return [];
                });
            
            // Store the pending request
            pendingRequests.set('chatHistory', requestPromise);
            return requestPromise;
        } catch (error) {
            console.error('Error creating chat history request:', error);
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
    
    // Other API methods with improved error handling
    // ...existing code...
    
    // Method to clear cache when needed (e.g., after mutations)
    clearCache: (key?: CacheKey): void => {
        if (key) {
            apiCache.delete(key);
            pendingRequests.delete(key);
        } else {
            apiCache.clear();
            pendingRequests.clear();
        }
    }
};

// Export types for use in other modules
export type { ChatData, FileMetadata, CacheKey };

// Alternative approach: Re-export the axios instance in case other modules need direct access
export { axiosInstance };