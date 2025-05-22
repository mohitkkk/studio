'use client'
import { useCallback, useEffect, useRef, useState, FormEvent } from "react";
import type { ChangeEvent } from 'react';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Paperclip, Send, FileText, Trash2, FolderOpen, XCircle } from "lucide-react";
import LoadingSpinner from "@/components/ui/loading-spinner";
import type { ChatSession, ChatMessage as ImportedChatMessage } from "@/types";
import { useToast } from "@/hooks/use-toast";
import { v4 as uuidv4 } from 'uuid';
import { useConnectionStatus } from "@/hooks/use-connection-status";
import { useApiService } from "@/hooks/use-api-service";
import FileSelector from "./file-selector"; 
import { Dialog, DialogContent, DialogTitle, DialogFooter, DialogDescription } from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import { VisuallyHidden } from "@/components/ui/visually-hidden";
import { applySettingsToRequest } from "@/lib/settings";

// Import settings functions
import { loadUploadSettings, getChatRequestSettings } from "@/lib/settings-api";
import { setGlobalApiService } from "@/lib/api-service-utils";
// Remove unused import

// Define ChatMessage interface
interface ChatMessage {
  id: string;
  sender: "user" | "assistant";
  content: string;
  timestamp: number;
  citations?: string[];
  error?: boolean;
}

// Mock function for saving/loading chat history from localStorage
const CHAT_HISTORY_KEY = "nebulaChatHistory";

const saveChatSession = (session: ChatSession) => {
  try {
    if (typeof window === 'undefined') return;
    const history = getLocalChatHistory();
    const existingIndex = history.findIndex(s => s.id === session.id);
    if (existingIndex > -1) {
      history[existingIndex] = session;
    } else {
      history.unshift(session); // Add new sessions to the beginning
    }
    localStorage.setItem(CHAT_HISTORY_KEY, JSON.stringify(history));
  } catch (error) {
    console.error("Error saving chat session:", error);
  }
};

const getLocalChatHistory = (): ChatSession[] => {
  try {
    if (typeof window === 'undefined') return [];
    const historyJson = localStorage.getItem(CHAT_HISTORY_KEY);
    return historyJson ? JSON.parse(historyJson) : [];
  } catch (error) {
    console.error("Error loading chat history:", error);
    return [];
  }
};

// Generate a new chat ID
const generateNewChatId = () => {
  return `chat_${Math.random().toString(36).substring(2, 10)}_${Date.now()}`;
};

export default function ChatPageClient({ 
  selectedFiles = [], 
  initialChatId = "" 
}: { 
  selectedFiles?: string[],
  initialChatId?: string
}) {
  // FIXED: Call hooks at the top level in the same order every time
  const serviceApiInstance = useApiService();
  // Ensure all required methods exist when setting initial state
  const [serviceApi, setServiceApi] = useState(serviceApiInstance);
  
  // State variables
  const [currentSession, setCurrentSession] = useState<ChatSession | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [visibleMessages, setVisibleMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isFileDialogOpen, setIsFileDialogOpen] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'disconnected' | 'connecting'>('connecting');
  const [backendVersion, setBackendVersion] = useState<string | null>(null);
  const [isWaitingForResponse, setIsWaitingForResponse] = useState(false);
  const [chatId, setChatId] = useState<string>(initialChatId || generateNewChatId());
  const fileInputRef = useRef<HTMLInputElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [isSummarizing, setIsSummarizing] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [fileProcessingStep, setFileProcessingStep] = useState("");
  const [documentSummary, setDocumentSummary] = useState<string | null>(null);
  const [selectedFilesList, setSelectedFilesList] = useState<string[]>(selectedFiles);
  const [availableFiles, setAvailableFiles] = useState<any[]>([]);
  const [isLoadingFiles, setIsLoadingFiles] = useState(false);
  const { toast } = useToast();
  
  // Remove uploaded file (missing function)
  const removeUploadedFile = useCallback(() => {
    setUploadedFile(null);
    setDocumentSummary(null);
    setFileProcessingStep("");
    setUploadProgress(0);
  }, []);

  // Check backend connection on component mount - with throttling
  useEffect(() => {
    let isMounted = true;
    let connectionCheckTimeout: NodeJS.Timeout | null = null;
    let retryCount = 0;
    const maxRetries = 3;
    
    const checkConnection = async () => {
      if (!isMounted) return;
      
      console.log("Checking connection with serviceApi:", serviceApi);
      
      if (!serviceApi) {
        console.warn("API service is null or undefined. Retrying soon.");
        connectionCheckTimeout = setTimeout(checkConnection, 1000);
        return;
      }
      
      // More robust check for critical methods
      const hasSendChatMessage = typeof serviceApi.sendChatMessage === 'function';
      const hasGetFiles = typeof serviceApi.getFiles === 'function';
      
      // If sendChatMessage is missing but we're logging the method list
      if (!hasSendChatMessage) {
        console.error(`Critical API method sendChatMessage is missing. Methods available:`, Object.keys(serviceApi));
        
        // Create a completely new service with guaranteed methods rather than trying to patch
        // Don't reuse the existing serviceApi since its structure might be corrupt
        const fallbackApi = {
          checkStatus: async () => ({ status: 'error', message: 'Not implemented' }),
          getFiles: async () => ({ files: [], count: 0 }),
          getChatHistory: async () => ({ messages: [] }),
          createChatSession: async () => ({ chat_id: `fallback_${Date.now()}` }),
          sendChatMessage: async (params: any) => {
            console.warn("Using emergency fallback sendChatMessage implementation");
            return { 
              message: "Sorry, the chat service is currently unavailable. Please try again later.",
              status: "error",
              client_id: params.sessionId || "fallback"
            };
          },
          updateFileSelection: async () => ({ status: 'error' }),
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
        
        setServiceApi(fallbackApi);
        
        connectionCheckTimeout = setTimeout(checkConnection, 1000);
        return;
      }
      
      // Continue with standard connection check
      if (!hasGetFiles) {
        console.warn("API method getFiles is missing, will use fallback for checking connection");
        
        // Try checkStatus if getFiles is missing
        setConnectionStatus('connecting');
        try {
          if (typeof serviceApi.checkStatus === 'function') {
            const status = await serviceApi.checkStatus();
            console.log("Connection successful with checkStatus:", status);
            setConnectionStatus('connected');
            return;
          } else {
            throw new Error("No suitable method found for connection check");
          }
        } catch (error) {
          console.error("Connection error details:", error);
          setConnectionStatus('disconnected');
          
          if (retryCount < maxRetries) {
            retryCount++;
            connectionCheckTimeout = setTimeout(checkConnection, 3000 * retryCount);
          }
          return;
        }
      }
      
      // Standard connection check with getFiles
      setConnectionStatus('connecting');
      try {
        console.log("Checking connection using getFiles()");
        const response = await serviceApi.getFiles();
        console.log("Connection successful with getFiles:", response);
        setConnectionStatus('connected');
        retryCount = 0;
      } catch (error) {
        console.error("Connection error details:", error);
        setConnectionStatus('disconnected');
        
        if (retryCount < maxRetries) {
          retryCount++;
          connectionCheckTimeout = setTimeout(checkConnection, 3000 * retryCount);
        }
      }
    };
    
    // Initial connection check
    checkConnection();
    
    // Set up periodic connection checks
    const periodicCheck = setInterval(() => {
      if (isMounted) {
        console.log("Running periodic connection check...");
        checkConnection();
      }
    }, 60000); // Check every minute
    
    return () => {
      isMounted = false;
      if (connectionCheckTimeout) {
        clearTimeout(connectionCheckTimeout);
      }
      clearInterval(periodicCheck);
    };
  }, [serviceApi, toast]);

  // Keep only messages that fit in the container
  useEffect(() => {
    if (messages.length > 0) {
      // Show a reasonable number of recent messages to avoid performance issues
      const recentMessages = [...messages].reverse().slice(0, 200); 
      setVisibleMessages(recentMessages.reverse());
    }
  }, [messages]);

  // Fetch chat history from backend or localStorage
  useEffect(() => {
    const loadChatSessions = async () => {
      // Ensure serviceApi and its methods are available
      if (connectionStatus !== 'connected' || !serviceApi || typeof serviceApi.getChatHistory !== 'function') {
        console.warn('loadChatSessions: API service not ready or not connected. Falling back to local.');
        const localSessions = getLocalChatHistory();
        if (localSessions.length > 0) {
          setCurrentSession(localSessions[0]);
        } else {
          // await startNewSession(); // Call startNewSession if no local sessions
        }
        return;
      }
      
      try {
        // Pass a special value or use the chatId state to get all sessions
        const sessions = await serviceApi.getChatHistory(chatId || 'all');
        if (sessions && sessions.length > 0) {
          setCurrentSession(sessions[0]); // Load the most recent session
        } else {
          // await startNewSession(); // No sessions on backend, start a new one
        }
      } catch (error) {
        console.error("Error loading chat sessions from backend:", error);
        const localSessions = getLocalChatHistory();
        if (localSessions.length > 0) {
          setCurrentSession(localSessions[0]);
        } else {
          // await startNewSession(); // Fallback to new session if backend and local fail
        }
      }
    };

    if (!currentSession) { // Only load if no session is currently active
      loadChatSessions();
    }
  }, [connectionStatus, serviceApi, currentSession]); // Added currentSession to dependencies

  // Start a new session or load latest one
  useEffect(() => {
    const history = getLocalChatHistory();
    if (history.length > 0 && !currentSession) {
      // setCurrentSession(history[0]); // Load latest local if no current session
      // startNewSession(); // This might be redundant given the load logic
    } else if (!currentSession) {
      // startNewSession();
    }
  }, []); 

  // Load messages from current session
  useEffect(() => {
    if (currentSession) {
      const mappedMessages = currentSession.messages.map(msg => ({
        ...msg,
        // Ensure sender is correctly typed
        sender: (msg.sender === "user" || msg.sender === "assistant") 
          ? msg.sender as "user" | "assistant" 
          : "assistant" // Default to assistant if sender is unexpected
      }));
      setMessages(mappedMessages);
      
      // Load document summary if available
      if (currentSession.documentContext) {
        setDocumentSummary(currentSession.documentContext.summary || null);
      }
    }
  }, [currentSession]);

  // Load available files from backend
  useEffect(() => {
    const loadAvailableFiles = async () => {
      if (connectionStatus !== 'connected' || !serviceApi) return;
      
      try {
        setIsLoadingFiles(true);
        console.log("Calling getFiles method...");
        
        // Check if getFiles is available
        if (typeof serviceApi.getFiles !== 'function') {
          console.error("getFiles method not found in API service");
          toast({
            title: "API Error",
            description: "File listing service not available",
            variant: "destructive"
          });
          setIsLoadingFiles(false);
          return;
        }
        
        const response = await serviceApi.getFiles();
        console.log("Raw getFiles response:", response);
        
        // Fix: Directly set the files array if it exists in the response
        if (response && typeof response === 'object') {
          if (Array.isArray(response.files)) {
            // Standard response format with files property
            setAvailableFiles(response.files);
            console.log(`Successfully loaded ${response.files.length} files from files property`);
          } else if (Array.isArray(response)) {
            // Handle case where response itself is the files array
            setAvailableFiles(response);
            console.log(`Successfully loaded ${response.length} files from direct array`);
          } else {
            // Last attempt - try to extract files if they're in a nested property
            const filesArray = findFilesArrayInObject(response);
            if (filesArray) {
              setAvailableFiles(filesArray);
              console.log(`Found files array in nested property: ${filesArray.length} files`);
            } else {
              console.error("Could not locate files array in response:", response);
              setAvailableFiles([]);
            }
          }
        } else {
          console.error("Invalid response format from getFiles:", response);
          setAvailableFiles([]);
        }
      } catch (error) {
        console.error("Error loading files:", error);
        toast({
          title: "Error",
          description: "Failed to load available files",
          variant: "destructive"
        });
        setAvailableFiles([]); // Ensure availableFiles is always an array
      } finally {
        setIsLoadingFiles(false);
      }
    };
    
    loadAvailableFiles();
  }, [connectionStatus, serviceApi, toast]);
  
  // Helper function to find files array in complex object structure
  const findFilesArrayInObject = (obj: any): any[] | null => {
    // Check if it's an array of objects with filename property (most likely files)
    if (Array.isArray(obj) && obj.length > 0 && typeof obj[0] === 'object' && obj[0].filename) {
      return obj;
    }
    
    // Search through object properties
    if (obj && typeof obj === 'object') {
      for (const key in obj) {
        // If property is named 'files' and is an array, return it
        if (key === 'files' && Array.isArray(obj[key])) {
          return obj[key];
        }
        
        // If property is an array of objects with filename property
        if (Array.isArray(obj[key]) && obj[key].length > 0 && 
            typeof obj[key][0] === 'object' && obj[key][0].filename) {
          return obj[key];
        }
        
        // Recursively search nested objects
        if (typeof obj[key] === 'object' && obj[key] !== null) {
          const result = findFilesArrayInObject(obj[key]);
          if (result) return result;
        }
      }
    }
    
    return null;
  };

  // Start a new chat session
  const startNewSession = useCallback(async () => {
    if (!serviceApi || connectionStatus !== 'connected') {
      console.log("Starting offline session");
      // Create local session without contacting backend
      const newSession = {
        id: uuidv4(),
        name: "New Chat",
        messages: [],
        lastModified: Date.now()
      };
      setCurrentSession(newSession);
      setChatId(newSession.id);
      return newSession;
    }

    try {
      setIsLoading(true);
      console.log("Creating new session on backend");
      
      // Call backend to create a session
      const response = await serviceApi.createChatSession();
      
      // Create new session object
      const newSession = {
        id: response.chat_id,
        name: response.chat_name || "New Chat",
        messages: [],
        lastModified: Date.now()
      };
      
      // Set current session state
      setCurrentSession(newSession);
      setChatId(newSession.id);
      
      // Update selected files if needed
      if (selectedFilesList.length > 0) {
        await serviceApi.updateFileSelection({
          sessionId: newSession.id,
          files: selectedFilesList
        });
      }
      
      console.log("New session created:", newSession);
      return newSession;
    } catch (error) {
      console.error("Error creating new session:", error);
      toast({
        title: "Error",
        description: "Failed to create new chat",
        variant: "destructive"
      });
      
      // Fall back to local session
      const fallbackSession = {
        id: uuidv4(),
        name: "New Chat (Offline)",
        messages: [],
        lastModified: Date.now()
      };
      setCurrentSession(fallbackSession);
      setChatId(fallbackSession.id);
      return fallbackSession;
    } finally {
      setIsLoading(false);
    }
  }, [connectionStatus, serviceApi, selectedFilesList, toast]);

  // Load existing chat history
  const loadChatHistory = useCallback(async (id: string) => {
    try {
      setIsLoading(true);
      
      if (!serviceApi || connectionStatus !== 'connected') {
        throw new Error("API service not available");
      }
      
      console.log("Loading chat history from backend:", id);
      const history = await serviceApi.getChatHistory(id);
      
      if (!history) {
        throw new Error("No history found");
      }
      
      // Convert backend format to frontend format
      const session = {
        id: history.chat_id,
        name: history.chat_name || "Chat",
        messages: (history.messages || []).map((msg: any) => ({
          id: msg.id || uuidv4(),
          sender: msg.role === "user" ? "user" : "assistant",
          content: msg.content,
          timestamp: new Date(msg.timestamp).getTime() || Date.now()
        })),
        lastModified: new Date(history.updated_at).getTime() || Date.now()
      };
      
      // Update state
      setCurrentSession(session);
      setMessages(session.messages || []);
      setChatId(session.id);
      
      // Update selected files if provided
      if (history.selected_files && Array.isArray(history.selected_files)) {
        setSelectedFilesList(history.selected_files);
      }
      
      return session;
    } catch (error) {
      console.error("Error loading chat history:", error);
      toast({ 
        title: "Error", 
        description: "Failed to load chat history", 
        variant: "destructive" 
      });
      await startNewSession();
      return null;
    } finally {
      setIsLoading(false);
    }
  }, [connectionStatus, serviceApi, startNewSession, toast]);


  // Effect for initializing chat: runs when initialChatId, loadChatHistory, or startNewSession changes.
  useEffect(() => {
    const initChat = async () => {
      if (initialChatId) {
        await loadChatHistory(initialChatId);
      } else if (!currentSession) {
        await startNewSession();
      }
    };
    
    if (connectionStatus === 'connected') {
      initChat();
    }
  }, [connectionStatus, initialChatId, currentSession, loadChatHistory, startNewSession]);

  // Handle file selection change
  const handleFileSelectionChange = useCallback(async (files: string[]) => {
    setSelectedFilesList(files);
    
    // Update backend if we have a session and connected
    if (chatId && connectionStatus === 'connected' && serviceApi?.updateFileSelection) {
      try {
        await serviceApi.updateFileSelection({
          sessionId: chatId,
          files: files
        });
        console.log(`Updated file selection on backend: ${files.length} files`);
      } catch (error) {
        console.error("Error updating file selection on backend:", error);
      }
    }
  }, [chatId, connectionStatus, serviceApi]);

  // Open file selector dialog
  const openFileSelector = useCallback(() => {
    if (connectionStatus !== 'connected') {
      toast({ title: "Backend Disconnected", description: "File selection requires connection.", variant: "destructive" });
      return;
    }
    setIsFileDialogOpen(true);
  }, [connectionStatus, toast]);

  // Send message to chat
  const handleSendMessage = async (e: FormEvent) => {
    e.preventDefault();
    
    if (!input.trim() || isWaitingForResponse) return;
    
    try {
      setIsWaitingForResponse(true);
      
      // Create user message
      const userMessage: ChatMessage = {
        id: uuidv4(),
        sender: "user",
        content: input,
        timestamp: Date.now()
      };
      
      // Add to current messages
      setMessages(prevMessages => [...prevMessages, userMessage]);
      
      // Remember input then clear it
      const sentInput = input;
      setInput("");
      
      // Check if serviceApi is available and has the required methods
      const hasSendChatMessage = serviceApi && typeof serviceApi.sendChatMessage === 'function';
      
      if (!serviceApi || !hasSendChatMessage || connectionStatus !== 'connected') {
        console.error("Cannot send message:", 
          !serviceApi ? "API service not available" : 
          !hasSendChatMessage ? "sendChatMessage method missing" : 
          "Not connected to backend");
        
        showOfflineMessage();
        return;
      }
      
      // Only execute this if connected and sendChatMessage is available
      try {
        console.log("Sending message to backend with parameters:", {
          message: sentInput,
          selected_files: selectedFilesList,
          sessionId: chatId
        });
        
        const response = await serviceApi.sendChatMessage({
          message: sentInput,
          selected_files: selectedFilesList,
          sessionId: chatId
        });
        
        console.log("Response from backend:", response);
        handleChatResponse(response);
      } catch (error) {
        handleChatError(error);
      }
      
      // Scroll to bottom after message is added
      setTimeout(() => {
        if (messagesContainerRef.current) {
          messagesContainerRef.current.scrollTop = messagesContainerRef.current.scrollHeight;
        }
      }, 100);
    } catch (error) {
      console.error("Error in message handling:", error);
    } finally {
      setIsWaitingForResponse(false);
    }
    
    // Helper function to show offline message
    function showOfflineMessage() {
      setTimeout(() => {
        const offlineMessage: ChatMessage = {
          id: uuidv4(),
          sender: "assistant",
          content: "I'm currently offline. Your message has been saved and will be processed when connection is restored.",
          timestamp: Date.now(),
          error: true
        };
        
        setMessages(prevMessages => [...prevMessages, offlineMessage]);
        setIsWaitingForResponse(false);
      }, 500); // Short delay for better UX
    }
    
    // Helper function to handle chat response
    function handleChatResponse(response: any) {
      if (response) {
        if (response.all_messages && Array.isArray(response.all_messages)) {
          // Map backend messages to our format
          interface BackendMessage {
            id?: string;
            role: "user" | "assistant" | string;
            content: string;
            timestamp: string | number | Date;
          }

          const mappedMessages: ChatMessage[] = response.all_messages.map((msg: BackendMessage): ChatMessage => ({
            id: msg.id || uuidv4(),
            sender: msg.role === "user" ? "user" : "assistant",
            content: msg.content,
            timestamp: new Date(msg.timestamp).getTime() || Date.now()
          }));
          
          setMessages(mappedMessages);
        } 
        else if (response.message) {
          const aiMessage: ChatMessage = {
            id: uuidv4(),
            sender: "assistant",
            content: response.message,
            timestamp: Date.now()
          };
          
          setMessages(prevMessages => [...prevMessages, aiMessage]);
        }
        
        // Update chat name if provided
        if (response.chat_name && currentSession) {
          setCurrentSession((prev: ChatSession | null) => prev ? {
            ...prev,
            name: response.chat_name || prev.name
          } : null);
        }
      }
    }
    
    // Helper function to handle chat errors
    function handleChatError(error: any) {
      console.error("Error sending message:", error);
      
      // Check if it's a connection error
      const isConnectionError = error instanceof Error && 
        (error.message.includes('network') || 
         error.message.includes('connection') ||
         error.message.includes('ECONNREFUSED') ||
         error.message.includes('timeout'));
      
      if (isConnectionError) {
        setConnectionStatus('disconnected');
      }
      
      // Add error message to chat
      const errorMessage: ChatMessage = {
        id: uuidv4(),
        sender: "assistant",
        content: isConnectionError
          ? "I can't reach the backend server right now. Please check your connection."
          : "Sorry, there was an error processing your message. Please try again.",
        timestamp: Date.now(),
        error: true
      };
      
      setMessages(prevMessages => [...prevMessages, errorMessage]);
      
      toast({
        title: "Error",
        description: isConnectionError 
          ? "Connection to backend failed"
          : "Failed to send message",
        variant: "destructive"
      });
    }
  };

  // Process uploaded file
  const processUploadedFile = useCallback(async (file: File) => {
    if (!serviceApi || typeof serviceApi.uploadFile !== 'function' || typeof serviceApi.processDocument !== 'function') {
        toast({ title: "API Error", description: "File processing service not available.", variant: "destructive" });
        return;
    }
    try {
      setIsSummarizing(true);
      setUploadProgress(0);
      setFileProcessingStep("Uploading file...");
      
      // Upload the file
      const response = await serviceApi.uploadFile(file, (progress) => setUploadProgress(progress));
      
      // Process the document using returned fileId
      setFileProcessingStep("Processing document...");
      const result = await serviceApi.processDocument({ fileId: response.fileId, fileName: response.fileName }); // Use correct property
      
      setDocumentSummary(result.summary || "Document processed."); // Display summary
      
      // Auto-select file if setting is enabled
      const uploadSettings = loadUploadSettings();
      if (uploadSettings.autoAddToChat) {
        handleFileSelectionChange([...selectedFilesList, response.fileName]); // Use fileName from response
      }
      
      toast({ title: "File processed", description: "File ready to use." });
    } catch (error) {
      console.error("Error processing file:", error);
      toast({ title: "Processing failed", description: error instanceof Error ? error.message : "Failed to process file.", variant: "destructive" });
    } finally {
      setIsSummarizing(false);
      setUploadProgress(100); // Ensure progress shows complete
    }
  }, [serviceApi, handleFileSelectionChange, selectedFilesList, toast]);

  // Handle file upload
  const handleFileChange = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    
    setUploadedFile(file); // Store the selected file
    setIsSummarizing(true); // Start processing indicator
    setDocumentSummary(null); // Clear previous summary
    setUploadProgress(0); // Reset progress
    setFileProcessingStep("Uploading file..."); // Set initial step

    // Ensure API service is available
    if (!serviceApi || typeof serviceApi.uploadFile !== 'function') {
        toast({ title: "API Error", description: "File upload service not available.", variant: "destructive" });
        setIsSummarizing(false);
        setUploadedFile(null); // Clear file if service fails
        return;
    }

    try {
      // Check for backend connection
      if (connectionStatus !== 'connected') throw new Error("Backend connection required for file upload.");

      // Upload file and get backend-generated filename/ID
      const uploadResponse = await serviceApi.uploadFile(file, (progress: number) => setUploadProgress(progress));
      
      // Update UI to reflect processing state
      setFileProcessingStep("Processing document...");
      
      // Use the fileId from the upload response
      const backendFilename = uploadResponse.fileId; 
      const newSelectedFiles = [...selectedFilesList, backendFilename];
      setSelectedFilesList(newSelectedFiles); // Auto-select the uploaded file
      
      // Update processing step and notify user
      setFileProcessingStep("Ready for questions!");
      toast({ title: "File Uploaded", description: `${file.name} ready.`, variant: "default" });
      setDocumentSummary(`File ${file.name} uploaded and selected for context.`);
      
      // Update current session with the newly selected file
      if (currentSession) {
        const updatedSession = { ...currentSession, selectedFiles: newSelectedFiles, lastModified: Date.now() };
        setCurrentSession(updatedSession);
        saveChatSession(updatedSession); // Save session locally
      }
    } catch (error) {
      console.error("Error processing document:", error);
      toast({ title: "Processing Error", description: connectionStatus === 'connected' ? `Could not process ${file.name}.` : "Backend connection required.", variant: "destructive" });
      setDocumentSummary("Failed to process document."); // Show error in summary area
      setFileProcessingStep("Error during processing"); // Update step
      setUploadedFile(null); // Clear uploaded file on error
    } finally {
      setIsSummarizing(false); // End processing indicator
    }
  };

  // Check connection function that can be passed to the ConnectionStatus component
  const checkConnection = useCallback(async () => {
    if (!serviceApi) return;
    
    setConnectionStatus('connecting');
    try {
      if (typeof serviceApi.checkStatus === 'function') {
        await serviceApi.checkStatus();
        setConnectionStatus('connected');
      } else if (typeof serviceApi.getFiles === 'function') {
        await serviceApi.getFiles();
        setConnectionStatus('connected');
      } else {
        throw new Error("No suitable API method found");
      }
    } catch (error) {
      console.error("Connection check failed:", error);
      setConnectionStatus('disconnected');
    }
  }, [serviceApi]);

  const triggerFileDialog = () => {
    fileInputRef.current?.click();
  };
  
  // Connection status component to display the current connection state
  function ConnectionStatus({ 
    status, 
    onReconnect 
  }: { 
    status: 'connected' | 'connecting' | 'disconnected',
    onReconnect?: () => void
  }) {
    return (
      <div 
        className={`flex items-center gap-1 text-xs px-2 py-0.5 rounded-full ${
          status === 'connected' 
            ? 'bg-green-500/20 text-green-600' 
            : status === 'connecting'
            ? 'bg-yellow-500/20 text-yellow-600'
            : 'bg-destructive/20 text-destructive cursor-pointer'
        }`}
        onClick={status === 'disconnected' && onReconnect ? onReconnect : undefined}
      >
        <div className={`w-1.5 h-1.5 rounded-full ${
          status === 'connected' 
            ? 'bg-green-500' 
            : status === 'connecting'
            ? 'bg-yellow-500'
            : 'bg-destructive'
        }`} />
        <span>
          {status === 'connected' 
            ? 'Connected' 
            : status === 'connecting' 
            ? 'Connecting...' 
            : 'Disconnected'}
        </span>
          </div>
        );
      }
  }

  // Message list component to display chat messages
  function MessageList({
    messages,
    isWaitingForResponse,
    containerRef
  }: {
    messages: ChatMessage[],
    isWaitingForResponse: boolean,
    containerRef: React.RefObject<HTMLDivElement>
  }) {
    // Pass the ref through props instead of accessing parent scope
    
    return (
      <div className="flex-1 overflow-y-auto p-2" ref={containerRef}>
        <div className="flex flex-col justify-end min-h-full">
          <div className="space-y-2">
            {messages.map((msg) => (
              <div
                key={msg.id}
                className={`flex ${
                  msg.sender === "user" ? "justify-end" : "justify-start"
                }`}
              >
                <div
                  className={`max-w-[80%] p-2 rounded-lg shadow-sm text-xs ${
                    msg.sender === "user"
                      ? "bg-primary text-primary-foreground rounded-br-none"
                      : msg.error
                      ? "bg-destructive/20 text-destructive-foreground rounded-bl-none border border-destructive/30"
                      : "bg-secondary text-secondary-foreground rounded-bl-none glassmorphic"
                  }`}
                >
                  <p className="whitespace-pre-wrap break-words">{msg.content}</p>
                  {msg.citations && msg.citations.length > 0 && (
                    <div className="mt-1 border-t border-primary-foreground/20 pt-1">
                      <p className="text-[10px] font-semibold">Citations:</p>
                      <ul className="list-disc list-inside text-[10px]">
                        {msg.citations.slice(0, 1).map((cite, i) => <li key={i} className="truncate">{cite}</li>)}
                        {msg.citations.length > 1 && <li className="truncate">+{msg.citations.length - 1} more</li>}
                      </ul>
                    </div>
                  )}
                  <p className={`text-[8px] ${msg.sender === 'user' ? 'text-primary-foreground/70 text-right' : 'text-muted-foreground/70 text-left'}`}>
                    {new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </p>
                </div>
              </div>
            ))}
            
            {isWaitingForResponse && (
              <div className="flex justify-start">
                <div className="max-w-[80%] p-2 rounded-lg shadow-sm bg-secondary text-secondary-foreground rounded-bl-none glassmorphic">
                  <LoadingSpinner size="sm" />
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  // Chat input component for entering messages and uploading files
  function ChatInput({
    input,
    setInput,
    onSendMessage,
    onUploadFile,
    isWaitingForResponse,
    isSummarizing,
    uploadedFile,
    connectionStatus
  }: {
    input: string,
    setInput: (value: string) => void,
    onSendMessage: (e: FormEvent) => void,
    onUploadFile: () => void,
    isWaitingForResponse: boolean,
    isSummarizing: boolean,
    uploadedFile: File | null,
    connectionStatus: 'connected' | 'disconnected' | 'connecting'
  }) {
    // Create a ref for the input element
    const inputRef = useRef<HTMLInputElement>(null);
    
    // Focus the input element on mount and when waiting state changes
    useEffect(() => {
      if (!isWaitingForResponse && !isSummarizing && inputRef.current) {
        inputRef.current.focus();
      }
    }, [isWaitingForResponse, isSummarizing]);
    
    // Handle click on the input to ensure it's focused
    const handleInputClick = () => {
      if (inputRef.current) {
        inputRef.current.focus();
      }
    };
    
    return (
      <CardFooter className="p-2 border-t border-border/50 shrink-0">
        <form onSubmit={onSendMessage} className="flex w-full items-center gap-1">
          <div className="relative">
            <Button 
              type="button" 
              variant="outline" 
              size="icon" 
              onClick={onUploadFile} 
              aria-label="Upload file"
              disabled={isSummarizing || connectionStatus !== 'connected'}
              className={`h-7 w-7 ${(isSummarizing || connectionStatus !== 'connected') ? "opacity-50 cursor-not-allowed" : ""}`}
            >
              <Paperclip className="h-3 w-3" />
            </Button>
            {uploadedFile && (
              <div className="absolute -top-1 -right-1 w-2 h-2 bg-primary rounded-full" />
            )}
          </div>
          
          <Input
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onClick={handleInputClick}
            placeholder={
              connectionStatus !== 'connected' 
                ? "Backend disconnected." 
                : (isSummarizing ? "Processing..." : "Type your message...")
            }
            className="flex-1 h-7 text-xs glassmorphic"
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                onSendMessage(e as unknown as FormEvent);
              }
            }}
            disabled={isWaitingForResponse || isSummarizing}
          />
          
          <Button 
            type="submit" 
            size="icon" 
            disabled={isWaitingForResponse || isSummarizing || !input.trim() || connectionStatus !== 'connected'} 
            aria-label="Send message"
            className={`h-7 w-7 transition-all bg-gradient-to-l from-[#FF297F] to-[#4B8FFF] 
              ${(isWaitingForResponse || isSummarizing || !input.trim() || connectionStatus !== 'connected') 
                 ? "opacity-50" 
                 : "opacity-100 hover:scale-105"}`}
          >
            {isWaitingForResponse ? <LoadingSpinner size="sm" /> : <Send className="h-3 w-3" />}
          </Button>
        </form>
      </CardFooter>
    );
  }
  
  // Store the API service instance globally for non-hook access
  useEffect(() => {
    // Store the API service instance globally for non-hook access
    if (serviceApi) {
      setGlobalApiService(serviceApi);
    }
  }, [serviceApi]);
  
  // Update the sendMessage function to ensure messages are added to the UI
  const sendMessage = async (messageText: string) => {
    try {
      // Create user message
      const newMessage: ChatMessage = {
        id: uuidv4(),
        sender: "user",
        content: messageText,
        timestamp: Date.now()
      };
      
      // Important: Update the local messages state FIRST before sending
      // This ensures the user message appears immediately
      setMessages((prevMessages: ChatMessage[]) => [...prevMessages, newMessage]);
      
      console.log("Sending message to backend with parameters:", {
        message: messageText,
        selected_files: selectedFilesList,
        sessionId: chatId
      });
      
      // Call the API
      const response = await serviceApi.sendChatMessage({
        message: messageText,
        selected_files: selectedFilesList,
        sessionId: chatId
      });
      
      console.log("Response from backend:", response);
      
      // After receiving a response, update the messages state with the AI response
      if (response && response.message) {
        const assistantMessage: ChatMessage = {
          id: uuidv4(),
          sender: "assistant",
          content: response.message,
          timestamp: Date.now()
        };
        
        // Update messages with the assistant's response
        setMessages((prevMessages: ChatMessage[]) => [...prevMessages, assistantMessage]);
        
        // If we received all_messages from the backend, use those to ensure consistency
        if (response.all_messages && Array.isArray(response.all_messages) && response.all_messages.length > 0) {
          // Transform the messages to match our local format
          const formattedMessages = response.all_messages.map((msg: any) => ({
            id: msg.id || uuidv4(),
            sender: msg.role === "user" ? "user" : "assistant",
            content: msg.content,
            timestamp: new Date(msg.timestamp).getTime() || Date.now()
          }));
          
          setMessages(formattedMessages);
        }
        
        // Update chat name if provided
        if (response.chat_name) {
          setCurrentSession(prev => prev ? {
            ...prev,
            name: response.chat_name || prev.name
          } : null);
        }
      }
    } catch (error) {
      console.error("Error in message handling:", error);
    }
  };
  
  // Effect for loading chat history
  useEffect(() => {
    const loadChatHistory = async () => {
      if (chatId && serviceApi && typeof serviceApi.getChatHistory === 'function') {
        try {
          const history = await serviceApi.getChatHistory(chatId);
          if (history && history.messages && Array.isArray(history.messages)) {
            // Format messages from history
            const formattedMessages = history.messages.map((msg: any) => ({
              id: msg.id || uuidv4(),
              sender: msg.role === "user" ? "user" : "assistant",
              content: msg.content,
              timestamp: new Date(msg.timestamp || Date.now()).getTime()
            }));
            
            setMessages(formattedMessages);
          }
        } catch (error) {
          console.error("Error loading chat history:", error);
        }
      }
    };
    
    loadChatHistory();
  }, [chatId, serviceApi]);
      
  // Create a ref for the end of messages to enable auto-scrolling
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]); // Changed from visibleMessages to messages to fix scroll behavior

  // Render the chat interface
  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto p-4 space-y-4 messages-container" ref={messagesContainerRef}>
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
            <p>No messages yet. Start a conversation!</p>
          </div>
        ) : (
          messages.map((message) => (
            <div 
              key={message.id} 
              className={`message ${message.sender === 'user' ? 'user-message' : 'assistant-message'} mb-4`}
            >
              <div className="message-content p-3 rounded-lg">
                {message.sender === 'user' ? (
                  <p className="whitespace-pre-wrap">{message.content}</p>
                ) : (
                  <div 
                    className="html-content" 
                    dangerouslySetInnerHTML={{ __html: message.content }} 
                  />
                )}
              </div>
            </div>
          ))
        )}
        <div className="messages-end-anchor" ref={messagesEndRef} />
      </div>
      
      {/* Message input area */}
      <form onSubmit={handleSendMessage} className="flex p-4 border-t border-border/50">
        <div className="flex-1 flex items-center gap-2">
          <Button 
            type="button"
            variant="outline" 
            size="icon" 
            onClick={openFileSelector}
            disabled={connectionStatus !== 'connected' || isWaitingForResponse}
            title="Select files"
          >
            <FolderOpen className="h-4 w-4" />
            <VisuallyHidden>Select files</VisuallyHidden>
          </Button>
          
          <Button 
            type="button"
            variant="outline" 
            size="icon" 
            onClick={triggerFileDialog}
            disabled={connectionStatus !== 'connected' || isWaitingForResponse}
            title="Upload file"
          >
            <Paperclip className="h-4 w-4" />
            <VisuallyHidden>Upload file</VisuallyHidden>
          </Button>
          
          <Input
            type="text"
            placeholder={isWaitingForResponse ? "Waiting for response..." : "Type your message..."}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={isWaitingForResponse}
            className="flex-1"
          />
          
          <Button 
            type="submit" 
            disabled={!input.trim() || isWaitingForResponse}
            variant="default"
            size="icon"
          >
            {isWaitingForResponse ? (
              <LoadingSpinner size="sm" />
            ) : (
              <Send className="h-4 w-4" />
            )}
            <VisuallyHidden>Send message</VisuallyHidden>
          </Button>
        </div>
      </form>
      
      {/* File selector dialog */}
      <Dialog open={isFileDialogOpen} onOpenChange={setIsFileDialogOpen}>
        <DialogContent>
          <DialogTitle>Select Files</DialogTitle>
          <FileSelector
            availableFiles={availableFiles}
            selectedFiles={selectedFilesList}
            onSelectionChange={handleFileSelectionChange}
            isLoading={isLoadingFiles}
          />
          <DialogFooter>
            <Button onClick={() => setIsFileDialogOpen(false)}>Done</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      
      {/* Hidden file input for uploads */}
      <input
        ref={fileInputRef}
        type="file"
        onChange={handleFileChange}
        className="hidden"
        accept=".txt,.pdf,.csv,.md,.json,.js,.py,.html,.css"
      />
    </div>
  );
}
