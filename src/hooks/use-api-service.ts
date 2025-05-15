import { useState, useEffect } from 'react';
import type { ChatMessage, ChatSession } from '@/types';
import axios from 'axios';

// Connect to the FastAPI backend - ensure this matches your server
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://13.202.208.115:8000';

interface ApiServiceProps {
  baseUrl?: string;
  headers?: Record<string, string>;
}

// Types matching Python backend models
interface ChatRequestBody {
  message: string;
  selected_files: string[];
  client_id?: string;
  ollama_model?: string;
  ollama_embedding_model?: string;
  top_k_per_file?: number;
  similarity_threshold?: number;
  temperature?: number;
  top_p?: number;
}

interface ChatResponseBody {
  message: string;
  context: any[];
  client_id: string;
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

  useEffect(() => {
    // Fix: Add console log to verify API connection
    console.log(`Connecting to API at: ${baseUrl}`);
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
    setIsInitialized(true);
  }, [baseUrl, props?.headers]);

  // Status check endpoint (root endpoint in our Python backend)
  const checkStatus = async () => {
    try {
      const response = await axios.get('/');
      return {
        version: response.data.version || '1.0',
        status: response.data.status || 'online',
      };
    } catch (error) {
      console.error("API status check failed:", error);
      throw new Error("API server is unavailable");
    }
  };

  // Get configuration from the root endpoint
  const getConfig = async () => {
    try {
      const response = await axios.get('/');
      // Extract meaningful config data
      return {
        version: response.data.version || '1.0',
        status: response.data.status || 'unknown'
      };
    } catch (error) {
      console.error("Unable to fetch server config:", error);
      return {
        version: 'unknown',
        status: 'offline'
      };
    }
  };

  // Get files from the /files endpoint
  const getFiles = async (tag?: string): Promise<FileMetadata[]> => {
    try {
      console.log(`Fetching files from API${tag ? ` with tag: ${tag}` : ''}`);
      const endpoint = tag ? `/files?tag=${encodeURIComponent(tag)}` : '/files';
      const response = await axios.get(endpoint);
      
      console.log('Files API response:', response.data);
      
      if (response.data && Array.isArray(response.data.files)) {
        return response.data.files;
      } else if (response.data && Array.isArray(response.data)) {
        // Fallback if the API returns a direct array
        return response.data;
      }
      
      console.warn('Unexpected API response format for files', response.data);
      return [];
    } catch (error) {
      console.error("Error fetching files:", error);
      throw error;
    }
  };

  // Upload a file to the backend
  const uploadFile = async (file: File, onProgress: ProgressCallback): Promise<any> => {
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('description', `Uploaded on ${new Date().toLocaleString()}`);
      
      const response = await axios.post('/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / (progressEvent.total || 100));
          onProgress(percentCompleted);
        },
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

  // Process document - in our backend this is done during upload
  const processDocument = async (params: {fileId: string; fileName: string}) => {
    // This is a mock for the frontend since the Python backend processes during upload
    return {
      content: "Document content processed by backend",
      summary: "Document has been processed and is ready for queries",
      fileId: params.fileId,
      mimeType: params.fileName.split('.').pop() || ""
    };
  };

  // Get chat sessions from the /chats endpoint
  const getChatSessions = async (): Promise<ChatSession[]> => {
    try {
      const response = await axios.get('/chats');
      
      // Transform the backend format to our frontend format
      return response.data.map((chat: any) => ({
        id: chat.chat_id,
        name: chat.chat_name,
        messages: [], // Messages are loaded separately
        lastModified: new Date(chat.updated_at).getTime(),
        selectedFiles: chat.selected_files || []
      }));
    } catch (error) {
      console.error("Error fetching chat sessions:", error);
      throw error;
    }
  };

  // Get a specific chat session
  const getChatSessionById = async (sessionId: string): Promise<ChatSession> => {
    try {
      const response = await axios.get(`/chat/${sessionId}`);
      
      // Transform messages from backend format
      const messages: ChatMessage[] = response.data.messages.map((msg: any) => ({
        id: msg.timestamp,
        sender: msg.role,
        content: msg.content,
        timestamp: new Date(msg.timestamp).getTime()
      }));
      
      return {
        id: sessionId,
        name: response.data.chat_name,
        messages,
        lastModified: new Date(response.data.updated_at).getTime(),
        selectedFiles: response.data.selected_files || []
      };
    } catch (error) {
      console.error(`Error fetching chat session ${sessionId}:`, error);
      throw error;
    }
  };

  // Create a new chat session
  const createChatSession = async (session: ChatSession): Promise<ChatSession> => {
    try {
      // The backend doesn't have a direct create session endpoint,
      // it creates a session when first message is sent
      // So we return the session as is
      return session;
    } catch (error) {
      console.error("Error creating chat session:", error);
      throw error;
    }
  };

  // Update a chat session - not directly supported in our backend
  const updateChatSession = async (session: ChatSession): Promise<ChatSession> => {
    // The backend doesn't have a direct update session endpoint,
    // it updates indirectly through message sends
    return session;
  };

  // Delete a chat session
  const deleteChatSession = async (sessionId: string): Promise<void> => {
    try {
      await axios.delete(`/chat/${sessionId}`);
    } catch (error) {
      console.error(`Error deleting chat session ${sessionId}:`, error);
      throw error;
    }
  };

  // Send a message to the chat endpoint
  const sendChatMessage = async (params: {
    sessionId?: string;
    message: string;
    selected_files?: string[];
  }): Promise<{answer: string; citations?: string[]}> => {
    try {
      const requestBody = {
        message: params.message,
        selected_files: params.selected_files || [],
        client_id: params.sessionId || undefined
      };
      
      const response = await axios.post<ChatResponseBody>('/chat', requestBody);
      
      return {
        answer: response.data.message,
        citations: [] // Our backend doesn't provide citations directly
      };
    } catch (error) {
      console.error("Error sending chat message:", error);
      throw error;
    }
  };

  // Remove a file - not directly implemented in our backend yet
  const removeFile = async (fileId: string): Promise<void> => {
    try {
      await axios.delete(`/files/${fileId}`);
    } catch (error) {
      console.error(`Error removing file ${fileId}:`, error);
      throw error;
    }
  };

  return {
    checkStatus,
    getConfig,
    getFiles,
    uploadFile,
    processDocument,
    getChatSessions,
    getChatSessionById,
    createChatSession,
    updateChatSession,
    deleteChatSession,
    sendChatMessage,
    removeFile,
  };
}
