/**
 * Utility for accessing API service outside of React components
 * or in places where hooks cannot be used directly
 */

// Import types only
import type { ApiServiceType } from "@/hooks/use-api-service";
import { api } from '@/lib/api';

// This variable will store the instance when it's provided by the component
let globalApiServiceInstance: ApiServiceType | null = null;

/**
 * Get a reference to the API service
 * This is NOT a hook - can be called anywhere
 */
export function getApiService(): ApiServiceType {
  if (!globalApiServiceInstance) {
    // Create a bridge between the standard API and the API service
    globalApiServiceInstance = {
      checkStatus: api.checkStatus || (async () => ({ status: 'error', message: 'API service not available' })),
      getFiles: api.getFiles || (async () => ({ files: [], count: 0 })),
      getChatHistory: async (chatId: string) => {
        try {
          return await api.getChatHistory();
        } catch (error) {
          console.error('Error in getChatHistory:', error);
          return { messages: [] };
        }
      },
      createChatSession: async () => {
        try {
          return await api.createNewChat();
        } catch (error) {
          console.error('Error in createChatSession:', error);
          return { chat_id: `error_${Date.now()}`, status: 'error' };
        }
      },
      // Critical missing method that must always be defined
      sendChatMessage: async (params: any) => {
        console.log('Using API bridge sendChatMessage with params:', params);
        try {
          // Call the actual implementation if it exists
          if (api.sendChatMessage) {
            return await api.sendChatMessage(params);
          }
          
          // Otherwise, use direct fetch fallback
          const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://13.202.208.115:3000'}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              message: params.message,
              selected_files: params.selected_files,
              client_id: params.sessionId,
              format: params.format || "html"
            })
          });
          
          if (!response.ok) {
            throw new Error(`Server responded with ${response.status}`);
          }
          
          return await response.json();
        } catch (error) {
          console.error('Error in sendChatMessage:', error);
          return {
            message: 'API service error. Please try again later.',
            client_id: params.sessionId || 'error',
            status: 'error'
          };
        }
      },
      updateFileSelection: async (params: any) => {
        try {
          if (api.updateFileSelection) {
            return await api.updateFileSelection(params);
          }
          
          // Fallback direct implementation
          const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://13.202.208.115:3000'}/file-selection`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              session_id: params.sessionId,
              selected_files: params.files
            })
          });
          
          if (!response.ok) {
            throw new Error(`Server responded with ${response.status}`);
          }
          
          return await response.json();
        } catch (error) {
          console.error('Error in updateFileSelection:', error);
          return { status: 'error' };
        }
      },
      clearCache: api.clearCache ? async () => {
        api.clearCache();
        return { success: true };
      } : async () => ({ success: false }),
      uploadFile: async (file: File, onProgress) => {
        try {
          if (api.uploadFile) {
            return await api.uploadFile(file, onProgress);
          }
          
          // Fallback implementation
          const formData = new FormData();
          formData.append('file', file);
          
          const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://13.202.208.115:3000'}/upload`, {
            method: 'POST',
            body: formData
          });
          
          if (!response.ok) {
            throw new Error(`Server responded with ${response.status}`);
          }
          
          const data = await response.json();
          return {
            fileId: data.filename,
            fileName: data.filename,
            status: data.status
          };
        } catch (error) {
          console.error('Error in uploadFile:', error);
          return { fileId: 'error', fileName: 'error', status: 'error' };
        }
      },
      processDocument: async (params) => {
        try {
          if (api.processDocument) {
            return await api.processDocument(params);
          }
          
          // Fallback implementation
          const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://13.202.208.115:3000'}/process`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename: params.fileName })
          });
          
          if (!response.ok) {
            throw new Error(`Server responded with ${response.status}`);
          }
          
          const data = await response.json();
          return {
            summary: data.summary || null,
            fileId: params.fileId,
            status: data.status
          };
        } catch (error) {
          console.error('Error in processDocument:', error);
          return { summary: null, fileId: 'error', status: 'error' };
        }
      }
    };
    
    console.log("API Service created with methods:", Object.keys(globalApiServiceInstance));
  }
  
  return globalApiServiceInstance;
}

/**
 * Set the global API service instance
 * This should be called from a component that uses useApiService
 */
export function setGlobalApiService(service: ApiServiceType): void {
  globalApiServiceInstance = service;
  console.log("Global API service set with methods:", Object.keys(service));
}

export function getGlobalApiService(): ApiServiceType | null {
  return globalApiServiceInstance;
}
