'use client'
import { useCallback, useEffect, useRef, useState, FormEvent } from "react";
import type { ChangeEvent } from 'react';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Paperclip, Send, FileText, Trash2, FolderOpen, XCircle } from "lucide-react";
import LoadingSpinner from "@/components/ui/loading-spinner";
import type { ChatSession } from "@/types";
import { useToast } from "@/hooks/use-toast";
import { v4 as uuidv4 } from 'uuid';
import { useConnectionStatus } from "@/hooks/use-connection-status";
import { api, useApiService } from "@/hooks/use-api-service";
import FileSelector from "./file-selector"; 
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import { VisuallyHidden } from "@/components/ui/visually-hidden";
import { applySettingsToRequest } from "@/lib/settings";

// Import settings functions
import { loadUploadSettings, getChatRequestSettings } from "@/lib/settings-api";

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

export default function ChatPageClient({ 
  selectedFiles = [], 
  initialChatId = "" 
}: { 
  selectedFiles?: string[],
  initialChatId?: string
}) {
  // Initialize API service
  const api = useApiService();
  
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
  const fileInputRef = useRef<HTMLInputElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [isSummarizing, setIsSummarizing] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [fileProcessingStep, setFileProcessingStep] = useState("");
  const [documentSummary, setDocumentSummary] = useState<string | null>(null);
  const [selectedFilesList, setSelectedFilesList] = useState<string[]>(selectedFiles);
  const { toast } = useToast();
  
  // Check backend connection on component mount - with throttling
  useEffect(() => {
    // Avoid multiple connection checks running in parallel
    let isMounted = true;
    let connectionCheckTimeout: NodeJS.Timeout | null = null;
    
    const checkConnection = async () => {
      if (!isMounted) return;
      
      setConnectionStatus('connecting');
      try {
        // Try to use getConfig instead of checkStatus
        // If getConfig is also unavailable, fall back to a simple GET request
        try {
          if (typeof api.getConfig === 'function') {
            await api.getConfig();
          } else {
            // Simple ping test using any available method
            await api.getChatSessions();
          }
        } catch (specificError) {
          console.warn("First connection check method failed:", specificError);
          // One more attempt with different method as fallback
          await api.getChatSessions();
        }
        
        if (!isMounted) return;
        setConnectionStatus('connected');
        setBackendVersion('connected'); // No version info available
        
        // Schedule next check after a reasonable interval
        connectionCheckTimeout = setTimeout(checkConnection, 30000); // 30 seconds
      } catch (error) {
        if (!isMounted) return;
        
        console.error("Backend connection failed:", error);
        setConnectionStatus('disconnected');
        toast({
          title: "Connection Error",
          description: "Could not connect to the AI backend. Using offline mode.",
          variant: "destructive",
          duration: 5000,
        });
        
        // Retry after a longer interval when disconnected
        connectionCheckTimeout = setTimeout(checkConnection, 60000); // 1 minute
      }
    };
    
    // Initial check
    checkConnection();
    
    // Cleanup function
    return () => {
      isMounted = false;
      if (connectionCheckTimeout) {
        clearTimeout(connectionCheckTimeout);
      }
    };
  }, [toast]); // Only include toast in dependencies, not api

  // Keep only messages that fit in the container
  useEffect(() => {
    if (messages.length > 0) {
      // Start with most recent messages and add older ones until we fill the container
      const recentMessages = [...messages].reverse().slice(0, 200); // Start with at most 200 most recent
      setVisibleMessages(recentMessages.reverse());
    }
  }, [messages]);

  // Auto-scroll to bottom of messages
  useEffect(() => {
    if (messagesContainerRef.current) {
      messagesContainerRef.current.scrollTop = messagesContainerRef.current.scrollHeight;
    }
  }, [messages]);

  // Fetch chat history from backend or localStorage
  useEffect(() => {
    const loadChatSessions = async () => {
      if (connectionStatus !== 'connected') return;
      
      try {
        // Check if method exists before calling it
        if (typeof api.getChatSessions === 'function') {
          const sessions = await api.getChatSessions();
          if (sessions && sessions.length > 0) {
            setCurrentSession(sessions[0]);
          } else {
            startNewSession();
          }
        } else {
          console.warn('getChatSessions method not available on API');
          startNewSession();
        }
      } catch (error) {
        console.error("Error loading chat sessions:", error);
        // Fallback to localStorage or create new session
        const localSessions = getLocalChatHistory();
        if (localSessions.length > 0) {
          setCurrentSession(localSessions[0]);
        } else {
          startNewSession();
        }
      }
    };

    if (!currentSession) {
      loadChatSessions();
    }
  }, [connectionStatus]);

  // Start a new session or load latest one
  useEffect(() => {
    const history = getLocalChatHistory();
    if (history.length > 0 && !currentSession) {
      startNewSession();
    } else if (!currentSession) {
      startNewSession();
    }
  }, []);

  // Load messages from current session
  useEffect(() => {
    if (currentSession) {
      // Map the messages to ensure the sender property matches the expected union type
      const mappedMessages = currentSession.messages.map(msg => ({
        ...msg,
        sender: (msg.sender === "user" || msg.sender === "assistant") 
          ? msg.sender as "user" | "assistant" 
          : "assistant" // Default fallback
      }));
      setMessages(mappedMessages);
      
      if (currentSession.documentContext) {
        setDocumentSummary(currentSession.documentContext.summary || null);
      }
    }
  }, [currentSession]);

  // Load available files from backend - with throttling and proper cleanup
  useEffect(() => {
    let isMounted = true;
    let loadFilesTimeout: NodeJS.Timeout | null = null;
    let loadAttempts = 0;
    const maxAttempts = 3;
    
    const loadAvailableFiles = async () => {
      if (!isMounted || connectionStatus !== 'connected') return;
      
      try {
        // Only try loading files a maximum number of times
        if (loadAttempts >= maxAttempts) {
          console.log(`Reached max attempts (${maxAttempts}) for loading files, stopping automatic retries`);
          return;
        }
        
        loadAttempts++;
        
        // Get files from the backend
        const files = await api.getFiles();
        if (!isMounted) return;
        
        if (files && files.files && Array.isArray(files.files)) {
          console.log(`Loaded ${files.files.length} files from backend`);
        } else if (Array.isArray(files)) {
          console.log(`Loaded ${files.length} files from backend`);
        } else {
          console.warn("Unexpected files format:", files);
        }
        
        // Schedule next refresh after reasonable interval - only if this was successful
        if (isMounted) {
          loadFilesTimeout = setTimeout(loadAvailableFiles, 300000); // 5 minutes
        }
      } catch (error) {
        if (!isMounted) return;
        console.error("Error loading files from backend:", error);
        
        // Retry with increasing backoff
        const backoff = Math.min(15000, 2000 * Math.pow(2, loadAttempts)); // Max 15 seconds
        console.log(`Will retry loading files in ${backoff/1000}s`);
        
        loadFilesTimeout = setTimeout(loadAvailableFiles, backoff);
      }
    };
    
    // Only load files when connected, with a small delay
    if (connectionStatus === 'connected') {
      // Small delay to ensure other components are initialized
      loadFilesTimeout = setTimeout(loadAvailableFiles, 1000);
    }
    
    // Cleanup function
    return () => {
      isMounted = false;
      if (loadFilesTimeout) {
        clearTimeout(loadFilesTimeout);
      }
    };
  }, [connectionStatus]);

  // Generate a new chat ID
  const generateNewChatId = () => {
    return `chat_${Math.random().toString(36).substring(2, 10)}_${Date.now()}`;
  };
  
  // Start a new chat session
  const startNewSession = async () => {
    const newChatId = generateNewChatId();
    setChatId(newChatId);
    setMessages([]);
    const newSession: ChatSession = {
      id: newChatId,
      name: `Chat ${new Date().toLocaleTimeString()}`,
      messages: [{
        id: uuidv4(),
        sender: 'assistant', // Changed from 'bot'
        content: "Hello! How can I assist you today?",
        timestamp: Date.now()
      }],
      lastModified: Date.now(),
    };
    
    setCurrentSession(newSession);
    setUploadedFile(null);
    setDocumentSummary(null);
    
    // Save to backend if connected
    if (connectionStatus === 'connected') {
      try {
        await api.createChatSession(newSession);
      } catch (error) {
        console.error("Error saving chat session to backend:", error);
        // Fallback to localStorage
        saveChatSession(newSession);
        toast({
          title: "Offline Mode",
          description: "Session saved locally. Connect to backend to sync.",
          variant: "default",
        });
      }
    } else {
      // Save locally if offline
      saveChatSession(newSession);
    }
  };

  // Handle file selection change with better error tolerance
  const handleFileSelectionChange = useCallback(async (files: string[]) => {
    // Use JSON.stringify for deep comparison of arrays
    const currentFilesString = JSON.stringify(selectedFilesList);
    const newFilesString = JSON.stringify(files);
    
    // Prevent unnecessary updates if files haven't changed
    if (currentFilesString === newFilesString) {
      return;
    }
    
    // Update local state immediately for responsiveness
    setSelectedFilesList(files);
    
    // Update current session with selected files
    if (currentSession) {
      const updatedSession = {
        ...currentSession,
        selectedFiles: files,
        lastModified: Date.now()
      };
      
      // Update local state and save locally right away
      setCurrentSession(updatedSession);
      saveChatSession(updatedSession);
      
      // Only show feedback if files changed and user added files
      if (currentFilesString !== newFilesString && files.length > 0) {
        toast({
          title: "Files Selected",
          description: `${files.length} file${files.length !== 1 ? 's' : ''} selected for context`,
          variant: "default",
        });
      }
      
      // Update backend in background (non-blocking) if connected
      if (connectionStatus === 'connected') {
        // Wrap in a try/catch to silently handle errors without disrupting UI
        try {
          await api.updateFileSelection({
            sessionId: currentSession.id,
            files: files
          });
        } catch (error) {
          console.warn("Background file selection update failed:", error);
          // No UI notification since this is a background operation
        }
      }
    }
  }, [selectedFilesList, currentSession, connectionStatus, toast, api]);

  // Open file selector dialog
  const openFileSelector = useCallback(() => {
    // Pre-check connection before opening dialog
    if (connectionStatus !== 'connected') {
      toast({
        title: "Backend Disconnected",
        description: "File selection requires connection to the backend.",
        variant: "destructive",
      });
      return;
    }
    setIsFileDialogOpen(true);
  }, [connectionStatus, toast]);

  // Send message to chat with improved empty message handling
  const handleSendMessage = async (e: FormEvent) => {
    e.preventDefault();
    
    // Don't send empty messages
    if (!input.trim() || isWaitingForResponse) return;

    try {
      setIsWaitingForResponse(true);
      
      // Create message for local display
      const userMessage: ChatMessage = {
        id: Date.now().toString(),
        sender: "user",
        content: input,
        timestamp: Date.now()
      };
      
      // Add user message to state and also update current session
      setMessages(prevMessages => {
        const newMessages = [...prevMessages, userMessage];
        // Update session with new messages
        if (currentSession) {
          const updatedSession = {
            ...currentSession,
            messages: newMessages,
            lastModified: Date.now()
          };
          setCurrentSession(updatedSession);
          saveChatSession(updatedSession);
        }
        return newMessages;
      });
      
      // Remember input then clear it
      const sentInput = input;
      setInput("");
      
      // Send to API with proper error handling
      if (connectionStatus === 'connected' && typeof api.sendChatMessage === 'function') {
        const response = await api.sendChatMessage({
          message: sentInput,
          selected_files: selectedFilesList,
          sessionId: chatId
        });
        
        // Check if the response includes all messages
        if (response && response.all_messages && Array.isArray(response.all_messages)) {
          // Replace current messages with the complete history from the server
          setMessages(response.all_messages);
        } else if (response && response.message) {
          // Fallback: just add the AI response to the existing messages
          const aiMessage: ChatMessage = {
            id: Date.now().toString(),
            sender: "assistant",
            content: response.message,
            timestamp: Date.now()
          };
          
          setMessages(prevMessages => [...prevMessages, aiMessage]);
        }
        
        // Update chat name if provided
        if (response && response.chat_name && currentSession) {
          setCurrentSession({
            ...currentSession,
            name: response.chat_name
          });
        }
      } else {
        // Offline mode - just add a placeholder response
        const offlineResponse: ChatMessage = {
          id: Date.now().toString(),
          sender: "assistant",
          content: "I'm currently offline. Your message has been saved locally and will be processed when connection is restored.",
          timestamp: Date.now(),
          error: true
        };
        setMessages(prevMessages => [...prevMessages, offlineResponse]);
      }
      
      // Scroll to bottom after message is added
      setTimeout(() => {
        if (messagesContainerRef.current) {
          messagesContainerRef.current.scrollTop = messagesContainerRef.current.scrollHeight;
        }
      }, 100);
      
    } catch (error) {
      console.error("Error sending message:", error);
      
      // Add error message to chat
      const errorMessage: ChatMessage = {
        id: Date.now().toString(),
        sender: "assistant",
        content: "Sorry, there was an error processing your message. Please try again.",
        timestamp: Date.now(),
        error: true
      };
      setMessages(prevMessages => [...prevMessages, errorMessage]);
      
      toast({
        title: "Error",
        description: "Failed to send message. Please try again.",
        variant: "destructive"
      });
    } finally {
      setIsWaitingForResponse(false);
    }
  };

  // Process uploaded file with progress tracking
  const processUploadedFile = useCallback(async (file: File) => {
    try {
      setIsSummarizing(true);
      setUploadProgress(0);
      setFileProcessingStep("Uploading file...");
      
      // Upload file with progress tracking
      const response = await api.uploadFile(file, (progress) => {
        setUploadProgress(progress);
      });
      
      // Process document
      setFileProcessingStep("Processing document...");
      const result = await api.processDocument({
        fileId: response.fileId,
        fileName: response.fileName
      });
      
      setDocumentSummary(result.summary || "Document processed successfully.");
      
      // Auto-add to chat if enabled
      const uploadSettings = loadUploadSettings();
      if (uploadSettings.autoAddToChat) {
        handleFileSelectionChange([...selectedFilesList, response.fileName]);
      }
      
      toast({
        title: "File processed",
        description: "Your file has been successfully processed and is ready to use."
      });
    } catch (error) {
      console.error("Error processing file:", error);
      toast({
        title: "Processing failed",
        description: error instanceof Error ? error.message : "Failed to process the file.",
        variant: "destructive"
      });
    } finally {
      setIsSummarizing(false);
      setUploadProgress(100);
    }
  }, [api, handleFileSelectionChange, selectedFilesList, toast]);

  // Handle file upload
  const handleFileChange = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    
    // Reset states
    setUploadedFile(file);
    setIsSummarizing(true);
    setDocumentSummary(null);
    setUploadProgress(0);
    setFileProcessingStep("Uploading file...");

    try {
      if (connectionStatus !== 'connected') {
        throw new Error("Backend connection required for file processing");
      }

      // Start file upload to backend
      const uploadResponse = await api.uploadFile(
        file,
        (progress: number) => setUploadProgress(progress)
      );
      
      setFileProcessingStep("Processing document...");
      
      // In our backend, processing happens during upload
      // We'll just get the file ID and add it to selected files
      const fileId = uploadResponse.fileId;
      
      // Add the file to the selected files list
      const newSelectedFiles = [...selectedFilesList, fileId];
      setSelectedFilesList(newSelectedFiles);
      
      // Show success message
      setFileProcessingStep("Ready for questions!");
      toast({
        title: "File Uploaded Successfully",
        description: `${file.name} is now ready for questions!`,
        variant: "default",
      });
      
      // Create a standard summary message for uploaded files
      setDocumentSummary(`File ${file.name} has been uploaded and is ready for questions. It has been automatically selected for chat context.`);
      
      // Update session with the new selected files
      if (currentSession) {
        const updatedSession = {
          ...currentSession,
          selectedFiles: newSelectedFiles,
          lastModified: Date.now()
        };
        setCurrentSession(updatedSession);
        saveChatSession(updatedSession);
      }
    } catch (error) {
      console.error("Error processing document:", error);
      toast({
        title: "Processing Error",
        description: connectionStatus === 'connected' ? 
          `Could not process ${file.name}. Please try a different file.` : 
          "Backend connection required for file processing.",
        variant: "destructive",
      });
      setDocumentSummary("Failed to process the document.");
      setFileProcessingStep("Error processing file");
      setUploadedFile(null);
    } finally {
      setIsSummarizing(false);
    }
  };

  // Trigger file dialog
  const triggerFileDialog = () => {
    fileInputRef.current?.click();
  };
  
  // Remove uploaded file
  const removeUploadedFile = useCallback(() => {
    setUploadedFile(null);
    setDocumentSummary(null);
    setFileProcessingStep("");
    setUploadProgress(0);
  }, []);

  // If an initialChatId prop is provided, use it. Otherwise generate a new one
  const [chatId, setChatId] = useState<string>(initialChatId || generateNewChatId());
  
  // First useEffect - handle initialization once on mount
  useEffect(() => {
    // Prevent the effect from running multiple times
    const hasInitialized = useRef(false);
    
    if (hasInitialized.current) return;
    hasInitialized.current = true;
    
    // Only create a new session if no initialChatId was provided
    if (!initialChatId && !currentSession) {
      startNewSession();
    } else if (initialChatId && !currentSession) {
      // Use the provided initialChatId directly - don't depend on chatId state
      setCurrentSession({
        id: initialChatId,
        name: "New Chat",
        messages: [{
          id: uuidv4(),
          sender: 'assistant', // Changed from 'bot'
          content: "Hello! How can I assist you today?",
          timestamp: Date.now()
        }],
        lastModified: Date.now(),
        selectedFiles: []
      });
    }
  }, []); // Empty dependency array, runs only once on mount

  // Second useEffect - handle selected files separately
  useEffect(() => {
    // Skip this effect on first render to avoid conflicts with other initialization
    const isFirstRender = useRef(true);
    if (isFirstRender.current) {
      isFirstRender.current = false;
      return;
    }
    
    // Load selected files from props if available
    if (selectedFiles?.length > 0) {
      handleFileSelectionChange(selectedFiles);
    }
  }, [selectedFiles, handleFileSelectionChange]);
  
  // Add a function to load chat history
  const loadChatHistory = useCallback(async (id: string) => {
    try {
      setIsLoading(true);
      
      if (!api || typeof api.getChatSessions !== 'function') {
        console.warn('API service not fully initialized, falling back to local data');
        // Fall back to local history
        const localHistory = getLocalChatHistory();
        const localSession = localHistory.find(s => s.id === id);
        
        if (localSession) {
          setCurrentSession(localSession);
          setSelectedFilesList(localSession.selectedFiles || []);
        } else {
          // Create new session if no matching one found
          startNewSession();
        }
        return;
      }
      
      // Normal API flow if the function exists
      const sessions = await api.getChatSessions();
      const session = sessions.find((s: any) => s.id === id);
      
      if (session) {
        setCurrentSession(session);
        setSelectedFilesList(session.selectedFiles || []);
      } else {
        // Fall back to local
        const localHistory = getLocalChatHistory();
        const localSession = localHistory.find(s => s.id === id);
        
        if (localSession) {
          setCurrentSession(localSession);
          setSelectedFilesList(localSession.selectedFiles || []);
        } else {
          startNewSession();
        }
      }
    } catch (error) {
      console.error("Error loading chat history:", error);
      toast({
        title: "Error",
        description: "Failed to load chat history.",
        variant: "destructive"
      });
      startNewSession();
    } finally {
      setIsLoading(false);
    }
  }, [handleFileSelectionChange]);

  // Update first useEffect to load history if chatId exists
  useEffect(() => {
    if (initialChatId) {
      // Load existing chat
      loadChatHistory(initialChatId);
    } else {
      // Create new chat
      startNewSession();
    }
  }, [initialChatId, loadChatHistory]);
  
  return (
    <div className="flex flex-col h-[100vh] w-full overflow-hidden pt-3 pb-4">
      <Card className="flex flex-col h-full border-0 shadow-2xl glassmorphic animated-smoke-bg">
        <CardHeader className="p-2 border-b border-border/50 shrink-0">
          <div className="flex justify-between items-center">
            <CardTitle className="text-base truncate max-w-[70%]">
              {currentSession?.name || "Nebula Chat"}
              {connectionStatus !== 'connected' && (
                <span className="ml-2 text-[10px] bg-destructive/20 text-destructive px-1.5 py-0.5 rounded-full">
                  Offline
                </span>
              )}
            </CardTitle>
            <div className="flex gap-1">
              <Button 
                variant="outline" 
                size="sm" 
                onClick={openFileSelector}
                className="h-7 text-xs px-2 flex items-center gap-1"
                disabled={connectionStatus !== 'connected'}
              >
                <FolderOpen className="h-3 w-3" />
                <span className="hidden sm:inline">Files</span>
                {selectedFilesList.length > 0 && (
                  <span className="bg-primary text-primary-foreground rounded-full px-1.5 text-[10px]">
                    {selectedFilesList.length}
                  </span>
                )}
              </Button>
              <Button 
                variant="outline" 
                size="sm" 
                onClick={startNewSession} 
                className="h-7 text-xs px-2"
              >
                New Chat
              </Button>
            </div>
          </div>
          
          {/* Selected files display */}
          {selectedFilesList.length > 0 && (
            <div className="mt-1 p-1 rounded-md bg-secondary/20 text-secondary-foreground">
              <div className="flex flex-wrap gap-1">
                {selectedFilesList.map((filename, index) => (
                  <Badge key={filename} variant="secondary" className="text-[10px] flex items-center gap-1 max-w-[150px]">
                    <FileText className="h-2 w-2" />
                    <span className="truncate">{filename.replace(/^\d+_/, '')}</span>
                    <Button 
                      variant="ghost" 
                      size="icon" 
                      className="h-3 w-3 p-0 ml-1" 
                      onClick={(e) => {
                        e.stopPropagation();
                        handleFileSelectionChange(selectedFilesList.filter(f => f !== filename));
                      }}
                    >
                      <XCircle className="h-2 w-2" />
                    </Button>
                  </Badge>
                ))}
                {selectedFilesList.length > 3 && (
                  <Badge variant="outline" className="text-[10px]">
                    +{selectedFilesList.length - 3} more
                  </Badge>
                )}
              </div>
            </div>
          )}
          
          {uploadedFile && (
            <div className="mt-1 p-1 rounded-md bg-secondary/50 text-secondary-foreground">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-1">
                  <FileText size={12} />
                  <span className="font-medium text-xs truncate max-w-[150px]">{uploadedFile.name}</span>
                </div>
                <Button variant="ghost" size="icon" className="h-5 w-5" onClick={(e) => {
                  e.preventDefault();
                  removeUploadedFile();
                }}>
                  <Trash2 size={12} />
                </Button>
              </div>
              
              {isSummarizing && (
                <div className="mt-1">
                  <div className="flex justify-between text-[10px] mb-1">
                    <span className="truncate max-w-[80%]">{fileProcessingStep || "Processing..."}</span>
                    <span>{Math.round(uploadProgress)}%</span>
                  </div>
                  <div className="w-full bg-secondary h-1 rounded-full overflow-hidden">
                    <div 
                      className="bg-primary h-full transition-all duration-300" 
                      style={{ width: `${uploadProgress}%` }}
                    />
                  </div>
                </div>
              )}
            </div>
          )}
          {documentSummary && !isSummarizing && (
             <div className="mt-1 text-[10px] p-1 rounded-md bg-muted/30 flex items-center justify-between">
                <span className="font-medium">Document processed</span>
                <Button 
                  variant="ghost" 
                  className="h-5 text-[10px] px-1" 
                  onClick={() => toast({
                    title: "Document Summary",
                    description: documentSummary.substring(0, 200) + (documentSummary.length > 200 ? "..." : ""),
                  })}>
                  View Summary
                </Button>
             </div>
          )}
        </CardHeader>
        
        {/* Chat messages area - only this section should scroll */}
        <div className="flex-1 overflow-y-auto p-2" ref={messagesContainerRef}>
          <div className="flex flex-col justify-end min-h-full">
            <div className="space-y-2">
              {visibleMessages.map((msg) => (
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
                    <p className="whitespace-pre-wrap break-words line-clamp-6">{msg.content}</p>
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
              {isLoading && (
                <div className="flex justify-start">
                  <div className="max-w-[80%] p-2 rounded-lg shadow-sm bg-secondary text-secondary-foreground rounded-bl-none glassmorphic">
                    <LoadingSpinner size="sm" />
                  </div>
                </div>
              )}
              {messages.length > visibleMessages.length && (
                <div className="text-center text-[10px] text-muted-foreground">
                  {messages.length - visibleMessages.length} earlier messages not shown
                </div>
              )}
            </div>
          </div>
        </div>
        
        <CardFooter className="p-2 border-t border-border/50 shrink-0">
          <form onSubmit={handleSendMessage} className="flex w-full items-center gap-1">
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
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              className="hidden"
              accept=".txt,.pdf,.md,.docx,.json,text/*,application/pdf,application/msword,application/vnd.openxmlformats-officedocument.wordprocessingml.document" 
            />
            <Input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={
                connectionStatus !== 'connected' 
                  ? "Backend disconnected. Limited functionality available." 
                  : (isSummarizing ? "Processing..." : "Type your message...")
              }
              className="flex-1 h-7 text-xs glassmorphic"
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSendMessage(e as unknown as FormEvent);
                }
              }}
              disabled={isLoading || isSummarizing}
            />
            <Button 
              type="submit" 
              size="icon" 
              disabled={isLoading || isSummarizing || !input.trim()} 
              aria-label="Send message"
              className={`h-7 w-7 transition-all bg-gradient-to-l from-[#FF297F] to-[#4B8FFF] ${(isLoading || !input.trim()) ? "opacity-50" : "opacity-100 hover:scale-105"}`}
            >
              {isLoading ? <LoadingSpinner size="sm" /> : <Send className="h-3 w-3" />}
            </Button>
          </form>
        </CardFooter>
      </Card>
      {/* File selector dialog */}
      <Dialog open={isFileDialogOpen} onOpenChange={setIsFileDialogOpen}>
        <DialogContent className="sm:max-w-[600px] p-0">
          <VisuallyHidden>
            <DialogTitle>File Selector</DialogTitle>
          </VisuallyHidden>
          <FileSelector 
            selectedFiles={selectedFilesList} 
            onSelectionChange={handleFileSelectionChange}
            onClose={() => setIsFileDialogOpen(false)}
          />
        </DialogContent>
      </Dialog>
    </div>
  );
}

interface ChatMessage {
  id: string;
  sender: "user" | "assistant"; // Use a union type to enforce consistent values
  content: string;
  timestamp: number;
  citations?: string[];
  error?: boolean;
}
