'use client'
import { useCallback, useEffect, useRef, useState, FormEvent } from "react";
import type { ChangeEvent } from 'react';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Paperclip, Send, FileText, Trash2, FolderOpen, XCircle, PlusCircle } from "lucide-react";
import LoadingSpinner from "@/components/ui/loading-spinner";
import type { ChatSession, ChatMessage as ImportedChatMessage } from "@/types";
import { useToast } from "@/hooks/use-toast";
import { v4 as uuidv4 } from 'uuid';
import { useConnectionStatus } from "@/hooks/use-connection-status";
import { useApiService } from "@/hooks/use-api-service";
import FileSelector from "./file-selector"; 
import { Dialog, DialogContent, DialogTitle, DialogFooter, DialogDescription, DialogHeader } from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import { VisuallyHidden } from "@/components/ui/visually-hidden";
import { applySettingsToRequest } from "@/lib/settings";
import { Alert } from "@/components/ui/alert";
import { Tooltip } from "@/components/ui/tooltip";

// Import settings functions
import { loadUploadSettings, getChatRequestSettings } from "@/lib/settings-api";
import { setGlobalApiService } from "@/lib/api-service-utils";
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import MarkdownRenderer from './markdown-renderer';

// Define ChatMessage interface
interface ChatMessage {
  id: string;
  sender: "user" | "assistant";
  content: string;
  timestamp: number;
  citations?: string[];
  error?: boolean;
}

// Updated localStorage keys for better organization
const CHAT_HISTORY_KEY = "nebulaChatHistory";
const CHAT_INDEX_KEY = "nebulaChatIndex";
const CHAT_COUNTER_KEY = "nebulaChatCounter";

// Enhanced localStorage functions for better persistence
const saveChatSession = (session: ChatSession) => {
  try {
    if (typeof window === 'undefined') return;
    
    // Ensure each message has a valid sender property before saving
    const validatedMessages = session.messages.map(msg => ({
      ...msg,
      // Explicitly ensure sender is correctly set
      sender: (msg.sender === "user" || msg.sender === "assistant") 
        ? msg.sender 
        : "assistant" // Default to assistant if sender is unexpected or missing
    }));
    
    // Create the validated session with sender information guaranteed
    const validatedSession = {
      ...session,
      messages: validatedMessages,
      selectedFiles: session.selectedFiles || [] // Ensure selected files are saved
    };
    
    // Save individual chat session by ID
    localStorage.setItem(`${CHAT_HISTORY_KEY}_${validatedSession.id}`, 
      JSON.stringify(validatedSession));
    
    // Update the chat index
    const chatIndex = getChatIndex();
    const existingIndex = chatIndex.findIndex(s => s.id === validatedSession.id);
    
    if (existingIndex > -1) {
      chatIndex[existingIndex] = {
        id: validatedSession.id,
        name: validatedSession.name,
        lastModified: validatedSession.lastModified,
        selectedFiles: validatedSession.selectedFiles || []
      };
    } else {
      chatIndex.unshift({
        id: validatedSession.id,
        name: validatedSession.name,
        lastModified: validatedSession.lastModified,
        selectedFiles: validatedSession.selectedFiles || []
      });
    }
    
    // Save updated index
    localStorage.setItem(CHAT_INDEX_KEY, JSON.stringify(chatIndex));
    console.log(`Saved chat session ${validatedSession.id} with ${validatedMessages.length} messages to local storage`);
  } catch (error) {
    console.error("Error saving chat session:", error);
  }
};

// Get chat index (list of all chats)
const getChatIndex = (): Array<{id: string, name: string, lastModified: number, selectedFiles: string[]}> => {
  try {
    if (typeof window === 'undefined') return [];
    const indexJson = localStorage.getItem(CHAT_INDEX_KEY);
    return indexJson ? JSON.parse(indexJson) : [];
  } catch (error) {
    console.error("Error loading chat index:", error);
    return [];
  }
};

// Load a specific chat session by ID
const loadChatSession = (sessionId: string): ChatSession | null => {
  try {
    if (typeof window === 'undefined') return null;
    const sessionJson = localStorage.getItem(`${CHAT_HISTORY_KEY}_${sessionId}`);
    if (!sessionJson) return null;
    
    // Parse and validate the session
    const session = JSON.parse(sessionJson);
    
    // Ensure messages have valid sender properties
    if (session && session.messages) {
      session.messages = session.messages.map((msg: any) => ({
        ...msg,
        sender: msg.sender === "user" ? "user" : "assistant" // Ensure valid sender
      }));
    }
    
    return session;
  } catch (error) {
    console.error(`Error loading chat session ${sessionId}:`, error);
    return null;
  }
};

// Get next chat number for naming
const getNextChatNumber = (): number => {
  try {
    if (typeof window === 'undefined') return 1;
    const counter = localStorage.getItem(CHAT_COUNTER_KEY);
    const nextNumber = counter ? parseInt(counter) + 1 : 1;
    localStorage.setItem(CHAT_COUNTER_KEY, nextNumber.toString());
    return nextNumber;
  } catch (error) {
    console.error("Error getting next chat number:", error);
    return 1;
  }
};

// Generate a new chat ID with timestamp
const generateNewChatId = () => {
  return `chat_${Math.random().toString(36).substring(2, 10)}_${Date.now()}`;
};

// Enhanced function to get local chat history
const getLocalChatHistory = (): ChatSession[] => {
  try {
    if (typeof window === 'undefined') return [];
    
    // Get all chats from the index
    const chatIndex = getChatIndex();
    
    // Load each chat by ID
    return chatIndex
      .map(chatInfo => loadChatSession(chatInfo.id))
      .filter(session => session !== null) as ChatSession[];
  } catch (error) {
    console.error("Error loading chat history:", error);
    return [];
  }
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
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [isSummarizing, setIsSummarizing] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [fileProcessingStep, setFileProcessingStep] = useState("");
  const [documentSummary, setDocumentSummary] = useState<string | null>(null);
  const [selectedFilesList, setSelectedFilesList] = useState<string[]>(selectedFiles);
  const [availableFiles, setAvailableFiles] = useState<any[]>([]);
  const [isLoadingFiles, setIsLoadingFiles] = useState(false);
  // FIXED: Add missing chats state to store the list of available chats
  const [chats, setChats] = useState<Array<{id: string, name: string, lastModified: number, selectedFiles: string[]}>>([]);
  const { toast } = useToast();
  
  // For automatic scrolling to bottom
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);
  
  // Debug logging for messages state
  useEffect(() => {
    if (messages.length > 0) {
      console.log("Current messages state:", messages.map(m => ({
        content: m.content.substring(0, 30) + "...",
        sender: m.sender
      })));
    }
  }, [messages]);
  
  // Remove uploaded file
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
        // FIXED: Fixed the reference to connectionCheckTimeout
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

  // Create a new chat
  const createNewChat = useCallback(() => {
    // Generate next chat number for naming
    const chatNumber = getNextChatNumber();
    const newChatId = generateNewChatId();
    
    // Create new session
    const newSession: ChatSession = {
      id: newChatId,
      name: `Chat ${chatNumber}`,
      messages: [],
      selectedFiles: [],
      lastModified: Date.now()
    };
    
    // Save to storage
    saveChatSession(newSession);
    
    // Update state
    setCurrentSession(newSession);
    setMessages([]);
    setChatId(newChatId);
    setSelectedFilesList([]);
    
    // Update chats list
    setChats(prevChats => [{
      id: newChatId,
      name: newSession.name,
      lastModified: newSession.lastModified,
      selectedFiles: []
    }, ...prevChats]);
    
    return newSession;
  }, []);

  // Start a new chat session
  const startNewSession = useCallback(async () => {
    if (!serviceApi || connectionStatus !== 'connected') {
      console.log("Starting offline session");
      return createNewChat();
    }

    try {
      setIsLoading(true);
      console.log("Creating new session on backend");
      
      // Call backend to create a session
      const response = await serviceApi.createChatSession();
      
      // Generate next chat number for naming
      const chatNumber = getNextChatNumber();
      
      // Create new session object
      const newSession = {
        id: response.chat_id,
        name: `Chat ${chatNumber}`,
        messages: [],
        selectedFiles: [],
        lastModified: Date.now()
      };
      
      // Set current session state
      setCurrentSession(newSession);
      setChatId(newSession.id);
      setMessages([]);
      
      // Save to local storage
      saveChatSession(newSession);
      
      // Update chats list
      setChats(prevChats => [{
        id: newSession.id,
        name: newSession.name,
        lastModified: newSession.lastModified,
        selectedFiles: []
      }, ...prevChats]);
      
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
      return createNewChat();
    } finally {
      setIsLoading(false);
    }
  }, [connectionStatus, serviceApi, toast, createNewChat]);

  // Load a specific chat
  const loadChat = useCallback((chatId: string) => {
    const session = loadChatSession(chatId);
    if (session) {
      setCurrentSession(session);
      setMessages(session.messages || []);
      setChatId(session.id);
      setSelectedFilesList(session.selectedFiles || []);
    }
  }, []);

  // Load chat list on component mount
  useEffect(() => {
    const chatIndex = getChatIndex();
    setChats(chatIndex);
    
    // If the chat list is empty, create a new chat
    if (chatIndex.length === 0 && !initialChatId) {
      createNewChat();
    }
  }, [createNewChat, initialChatId]);

  // Load existing chat history from backend or localStorage
  const loadChatHistory = useCallback(async (id: string) => {
    try {
      setIsLoading(true);
      
      // First try to load from local storage
      const localSession = loadChatSession(id);
      if (localSession) {
        setCurrentSession(localSession);
        setMessages(localSession.messages || []);
        setChatId(localSession.id);
        setSelectedFilesList(localSession.selectedFiles || []);
        return localSession;
      }
      
      // If not found locally and API is available, try loading from backend
      if (serviceApi && connectionStatus === 'connected') {
        console.log("Loading chat history from backend:", id);
        const history = await serviceApi.getChatHistory(id);
        
        if (history) {
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
            selectedFiles: history.selected_files || [],
            lastModified: new Date(history.updated_at).getTime() || Date.now()
          };
          
          // Update state
          setCurrentSession(session);
          setMessages(session.messages || []);
          setChatId(session.id);
          setSelectedFilesList(session.selectedFiles || []);
          
          // Save to local storage for future use
          saveChatSession(session);
          
          return session;
        }
      }
      
      // If we reach here, no chat found - create a new one
      toast({
        title: "Chat not found",
        description: "Creating a new chat session",
        variant: "default"
      });
      
      return startNewSession();
    } catch (error) {
      console.error("Error loading chat history:", error);
      toast({ 
        title: "Error", 
        description: "Failed to load chat history", 
        variant: "destructive" 
      });
      return startNewSession();
    } finally {
      setIsLoading(false);
    }
  }, [connectionStatus, serviceApi, startNewSession, toast]);

  // Effect for initializing chat
  useEffect(() => {
    const initChat = async () => {
      // If an initial chat ID was provided, try to load it
      if (initialChatId) {
        await loadChatHistory(initialChatId);
      } 
      // If we have a current chatId but no session, try to load that chat
      else if (chatId && !currentSession) {
        const localSession = loadChatSession(chatId);
        if (localSession) {
          setCurrentSession(localSession);
          setMessages(localSession.messages || []);
        } else {
          // No local session found - load from chat index or create new
          const chatIndex = getChatIndex();
          if (chatIndex.length > 0) {
            loadChat(chatIndex[0].id);
          } else {
            await startNewSession();
          }
        }
      } 
      // If no current session and no specific chat to load, create new or load latest
      else if (!currentSession) {
        const chatIndex = getChatIndex();
        if (chatIndex.length > 0) {
          loadChat(chatIndex[0].id);
        } else {
          await startNewSession();
        }
      }
    };
    
    initChat();
  }, [chatId, currentSession, initialChatId, loadChat, loadChatHistory, startNewSession]);

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
    
    // Always update local storage with selected files for current chat
    if (currentSession) {
      const updatedSession = {
        ...currentSession,
        selectedFiles: files,
        lastModified: Date.now()
      };
      saveChatSession(updatedSession);
      setCurrentSession(updatedSession);
      
      // Update chat list
      setChats(prevChats => {
        const updatedChats = [...prevChats];
        const chatIndex = updatedChats.findIndex(c => c.id === chatId);
        if (chatIndex > -1) {
          updatedChats[chatIndex] = {
            ...updatedChats[chatIndex],
            selectedFiles: files,
            lastModified: Date.now()
          };
        }
        return updatedChats;
      });
    }
  }, [chatId, connectionStatus, serviceApi, currentSession]);

  // Open file selector dialog
  const openFileSelector = useCallback(() => {
    if (connectionStatus !== 'connected') {
      toast({ title: "Backend Disconnected", description: "File selection requires connection.", variant: "destructive" });
      return;
    }
    setIsFileDialogOpen(true);
  }, [connectionStatus, toast]);

  // Trigger file input click
  const triggerFileDialog = useCallback(() => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  }, [fileInputRef]);

  // Make sure we never use local IP in uploads
  const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://13.202.208.115:3000';

  // Handle file change
  const handleFileChange = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    
    setUploadedFile(file);
    setIsSummarizing(true);
    setDocumentSummary(null);
    setUploadProgress(0);
    setFileProcessingStep("Uploading file...");

    if (!serviceApi || typeof serviceApi.uploadFile !== 'function') {
        toast({ title: "API Error", description: "File upload service not available.", variant: "destructive" });
        setIsSummarizing(false);
        setUploadedFile(null);
        return;
    }

    try {
      if (connectionStatus !== 'connected') throw new Error("Backend connection required for file upload.");

      const uploadResponse = await serviceApi.uploadFile(file, (progress: number) => setUploadProgress(progress));
      
      setFileProcessingStep("Processing document...");
      
      const backendFilename = uploadResponse.fileId; 
      const newSelectedFiles = [...selectedFilesList, backendFilename];
      setSelectedFilesList(newSelectedFiles);
      
      setFileProcessingStep("Ready for questions!");
      toast({ title: "File Uploaded", description: `${file.name} ready.`, variant: "default" });
      setDocumentSummary(`File ${file.name} uploaded and selected for context.`);
      
      if (currentSession) {
        const updatedSession = { ...currentSession, selectedFiles: newSelectedFiles, lastModified: Date.now() };
        setCurrentSession(updatedSession);
        saveChatSession(updatedSession);
      }
    } catch (error) {
      console.error("Error processing document:", error);
      toast({ title: "Processing Error", description: connectionStatus === 'connected' ? `Could not process ${file.name}.` : "Backend connection required.", variant: "destructive" });
      setDocumentSummary("Failed to process document.");
      setFileProcessingStep("Error during processing");
      setUploadedFile(null);
    } finally {
      setIsSummarizing(false);
    }
  };

  // Send message
  const sendMessage = async (messageText: string) => {
    try {
      const newMessage: ChatMessage = {
        id: uuidv4(),
        sender: "user",
        content: messageText,
        timestamp: Date.now()
      };
      
      setMessages((prevMessages: ChatMessage[]) => [...prevMessages, newMessage]);
      
      console.log("Sending message to backend with parameters:", {
        message: messageText,
        selected_files: selectedFilesList,
        sessionId: chatId
      });
      
      const response = await serviceApi.sendChatMessage({
        message: messageText,
        selected_files: selectedFilesList,
        sessionId: chatId
      });
      
      console.log("Response from backend:", response);
      
      if (response && response.message) {
        const assistantMessage: ChatMessage = {
          id: uuidv4(),
          sender: "assistant",
          content: response.message,
          timestamp: Date.now()
        };
        
        setMessages((prevMessages: ChatMessage[]) => {
          const updatedMessages = [...prevMessages, assistantMessage];
          
          if (currentSession) {
            const updatedSession = {
              ...currentSession,
              messages: updatedMessages,
              selectedFiles: selectedFilesList,
              lastModified: Date.now()
            };
            
            saveChatSession(updatedSession);
            setCurrentSession(updatedSession);
            
            setChats(prevChats => {
              const updatedChats = [...prevChats];
              const chatIndex = updatedChats.findIndex(c => c.id === chatId);
              if (chatIndex > -1) {
                updatedChats[chatIndex] = {
                  ...updatedChats[chatIndex],
                  lastModified: Date.now()
                };
              }
              return updatedChats;
            });
          }
          
          return updatedMessages;
        });
        
        if (response.chat_name && currentSession) {
          setCurrentSession(prev => prev ? {
            ...prev,
            name: response.chat_name || prev.name
          } : null);
          
          // Update chat name in the list
          setChats(prevChats => {
            const updatedChats = [...prevChats];
            const chatIndex = updatedChats.findIndex(c => c.id === chatId);
            if (chatIndex > -1) {
              updatedChats[chatIndex] = {
                ...updatedChats[chatIndex],
                name: response.chat_name
              };
            }
            return updatedChats;
          });
        }

        if (response.client_id && response.client_id !== chatId) {
          console.log(`Syncing local chat ID ${chatId} to backend ID ${response.client_id}`);
          
          if (currentSession) {
            const newSession = {
              ...currentSession,
              id: response.client_id,
              messages: [...messages, assistantMessage],
              lastModified: Date.now()
            };
            
            saveChatSession(newSession);
            setCurrentSession(newSession);
            setChatId(response.client_id);
            
            // Update chat ID in the list
            setChats(prevChats => {
              const updatedChats = prevChats.filter(c => c.id !== chatId);
              updatedChats.unshift({
                id: response.client_id,
                name: newSession.name,
                lastModified: newSession.lastModified,
                selectedFiles: newSession.selectedFiles || []
              });
              return updatedChats;
            });
          }
        }
      }
    } catch (error) {
      console.error("Error in message handling:", error);
      toast({
        title: "Error",
        description: "Failed to send message. Please try again.",
        variant: "destructive"
      });
    }
  };

  // Handle send message
  const handleSendMessage = async (e: FormEvent) => {
    e.preventDefault();
    
    if (!input.trim() || isWaitingForResponse) return;
    
    try {
      setIsWaitingForResponse(true);
      await sendMessage(input);
      setInput("");
    } catch (error) {
      console.error("Error sending message:", error);
      toast({
        title: "Error",
        description: "Failed to send message",
        variant: "destructive"
      });
    } finally {
      setIsWaitingForResponse(false);
    }
  };

  // Verify and repair local storage
  const verifyAndRepairLocalStorage = useCallback(() => {
    // Check all localStorage items and fix any messages with incorrect sender property
    try {
      if (typeof window === 'undefined') return;
      
      const history = getLocalChatHistory();
      let needsUpdate = false;
      
      for (let i = 0; i < history.length; i++) {
        const session = history[i];
        let sessionNeedsUpdate = false;
        
        if (Array.isArray(session.messages)) {
          for (let j = 0; j < session.messages.length; j++) {
            const msg = session.messages[j];
            
            // Fix sender if needed
            if (msg.sender !== "user" && msg.sender !== "assistant") {
              console.warn(`Repairing message in session ${session.id} with invalid sender: ${msg.sender}`);
              session.messages[j] = {
                ...msg,
                sender: msg.content.includes("<") ? "assistant" : "user"
              };
              sessionNeedsUpdate = true;
            }
          }
        }
        
        if (sessionNeedsUpdate) {
          history[i] = session;
          needsUpdate = true;
        }
      }
      
      if (needsUpdate) {
        // Update each session individually
        history.forEach(session => {
          saveChatSession(session);
        });
        console.log("Repaired localStorage chat history with corrected sender properties");
      }
    } catch (error) {
      console.error("Error repairing localStorage:", error);
    }
  }, []);

  // Run repair on component mount
  useEffect(() => {
    verifyAndRepairLocalStorage();
  }, [verifyAndRepairLocalStorage]);

  // Add event listeners for sidebar chat selection
  useEffect(() => {
    // Function to handle chat selection from sidebar
    const handleChatSelected = (event: Event) => {
      const customEvent = event as CustomEvent;
      if (customEvent.detail && customEvent.detail.chatId) {
        const newChatId = customEvent.detail.chatId;
        
        // Load the selected chat
        const chatData = localStorage.getItem(`${CHAT_HISTORY_KEY}_${newChatId}`);
        if (chatData) {
          const parsedChat = JSON.parse(chatData);
          setCurrentSession(parsedChat);
          setMessages(parsedChat.messages || []);
          setChatId(newChatId);
          setSelectedFilesList(parsedChat.selectedFiles || []);
        }
      }
    };
    
    // Listen for chat selection and creation events from sidebar
    window.addEventListener('chatSelected', handleChatSelected);
    window.addEventListener('chatCreated', handleChatSelected);
    
    return () => {
      window.removeEventListener('chatSelected', handleChatSelected);
      window.removeEventListener('chatCreated', handleChatSelected);
    };
  }, []);

  // Add a function to refresh available files
  const refreshAvailableFiles = useCallback(async () => {
    setIsLoadingFiles(true);
    try {
      const response = await serviceApi.getFiles();
      // Handle both array responses and responses with a files property
      const files = Array.isArray(response) ? response : response?.files || [];
      setAvailableFiles(files);
    } catch (error) {
      console.error("Error fetching available files:", error);
      toast({
        title: "Error",
        description: "Failed to load available files.",
        variant: "destructive"
      });
    } finally {
      setIsLoadingFiles(false);
    }
  }, [serviceApi, toast]);

  // Helper function to detect message format
  const detectMessageFormat = (content: string): 'markdown' | 'html' => {
    // Check for HTML tags
    if (/<\/?[a-z][\s\S]*>/i.test(content)) {
      return 'html';
    }
    
    // Check for common Markdown patterns
    const markdownPatterns = [
      /\*\*(.+?)\*\*/,                // Bold
      /\*(.+?)\*/,                    // Italic
      /^#+\s+.+$/m,                   // Headings
      /^\s*[\-\*\+]\s+.+/m,           // Unordered list
      /^\s*\d+\.\s+.+/m,              // Ordered list
      /\[.+?\]\(.+?\)/,               // Links
      /```[\s\S]+?```/m,              // Code fences
      /`.+?`/                        // Inline code
    ];
    
    // If any Markdown pattern matches, treat as Markdown
    for (const pattern of markdownPatterns) {
      if (pattern.test(content)) {
        return 'markdown';
      }
    }
    
    // Default to HTML if no patterns matched
    return 'html';
  };

  // Function to control file selector dialog visibility
  const setShowFileSelector = useCallback((show: boolean) => {
    if (show) {
      // Reuse existing openFileSelector function which has connection validation
      openFileSelector();
    } else {
      setIsFileDialogOpen(false);
    }
  }, [openFileSelector, setIsFileDialogOpen]);

  // Render the chat interface 
  return (
    <div className="flex flex-col min-h-[98vh] max-h-[98vh] w-full max-w-full overflow-hidden  text-white">
      {/* Chat header - simplified without dropdown */}
      <div className="flex items-center justify-between p-2 border border-gray-800 bg-black">
        <div className="flex items-center gap-2">
          {/* Display current chat name instead of dropdown */}
          <h2 className="text-sm font-medium text-white truncate">
            {currentSession?.name || "Chat"}
          </h2>
        </div>
        
        <div className="flex items-center gap-2">
          {/* Connection status indicator */}
          <div 
            className={`h-3 w-3 rounded-full ${
              connectionStatus === 'connected' 
                ? 'bg-green-500' 
                : connectionStatus === 'connecting' 
                  ? 'bg-yellow-500 animate-pulse' 
                  : 'bg-red-500'
            }`}
            title={`Backend: ${connectionStatus}`}
          />
          
          {/* Files button - Added next to New Chat */}
          <Button 
            variant="outline" 
            size="sm" 
            onClick={() => setShowFileSelector(true)}
            className="text-xs flex items-center gap-1"
            title="Select Files"
          >
            <FileText className="h-3 w-3" />
            <span className="hidden sm:inline">Files</span>
          </Button>
          
          {/* Keep the New Chat button */}
          <Button 
            variant="outline" 
            size="sm" 
            onClick={startNewSession}
            className="text-xs flex items-center gap-1"
          >
            <PlusCircle className="h-3 w-3" />
            <span className="hidden sm:inline">New Chat</span>
            <span className="sm:hidden">New</span>
          </Button>
        </div>
      </div>

      {/* Messages container */}
      <div className="flex-1 overflow-y-auto p-2 md:p-4 space-y-4 messages-container no-scrollbar border-x border-gray-800 backdrop-blur-md bg-black-600/10" ref={messagesContainerRef}>
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-gray-400">
            <p>No messages yet. Start a conversation!</p>
          </div>
        ) : (
          <div className="flex flex-col space-y-3">
            {messages.map((message) => (
              <div 
                key={message.id} 
                className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'} mb-3 w-full`}
              >
                <div 
                  className={`message-bubble max-w-[60%] p-3 shadow-sm
                    ${message.sender === 'user' 
                      ? 'bg-stone-600 text-white rounded-2xl rounded-br-none ml-auto' 
                      : 'bg-stone-900 text-gray-200 rounded-2xl rounded-bl-none mr-auto border border-stone-700'
                    }`}
                >
                  {message.sender === 'user' ? (
                    <p className="whitespace-pre-wrap text-sm">{message.content}</p>
                  ) : (
                    <div className="html-content text-sm" dangerouslySetInnerHTML={{ __html: message.content }} />
                  )}
                  <div className={`text-[10px] mt-1 opacity-70 ${message.sender === 'user' ? 'text-right' : ''}`}>
                    {new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
        <div className="messages-end-anchor h-4" ref={messagesEndRef} />
      </div>
      
      {/* Message input */}
      <CardFooter className="p-2 border border-gray-800 bg-black">
        <form onSubmit={handleSendMessage} className="flex w-full items-center gap-1">
          <div className="relative">
            <Button 
              type="button" 
              variant="outline" 
              size="icon" 
              onClick={openFileSelector} 
              aria-label="Select files"
              disabled={connectionStatus !== 'connected'}
              className="h-7 w-7"
            >
              <FolderOpen className="h-3 w-3" />
            </Button>
          </div>
          
          <div className="relative">
            <Button 
              type="button" 
              variant="outline" 
              size="icon" 
              onClick={triggerFileDialog} 
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
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={
              connectionStatus !== 'connected' 
                ? "Backend disconnected." 
                : (isSummarizing ? "Processing..." : "Type your message...")
            }
            className="flex-1 h-7 text-xs glassmorphic "
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSendMessage(e as unknown as FormEvent);
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
      
      {/* File selector dialog */}
      {isFileDialogOpen && (
        <Dialog open={isFileDialogOpen} onOpenChange={setIsFileDialogOpen}>
          <DialogContent className="max-w-[95%] w-full sm:max-w-[85%] md:max-w-[75%] lg:max-w-[65%] 
            max-h-[90vh] sm:max-h-[85vh] mx-auto p-0 rounded-lg overflow-hidden">
            <DialogHeader className="p-3 sm:p-4 pb-1 sm:pb-2">
              <DialogTitle className="text-base sm:text-lg">Select Files</DialogTitle>
              <DialogDescription className="text-xs sm:text-sm">
                Choose files to use as context for your conversation.
              </DialogDescription>
            </DialogHeader>
            <div className="px-2 sm:px-4 pb-2 sm:pb-4 w-full h-[60vh] sm:h-[65vh] md:h-[70vh] overflow-y-auto">
              <FileSelector
                availableFiles={availableFiles}
                selectedFiles={selectedFilesList}
                onSelectionChange={handleFileSelectionChange}
                isLoading={isLoadingFiles}
                onClose={() => setIsFileDialogOpen(false)}
                onRefresh={refreshAvailableFiles}
              />
            </div>
          </DialogContent>
        </Dialog>
      )}
      
      {/* Hidden file input */}
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

