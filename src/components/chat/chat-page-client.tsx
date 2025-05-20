
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
  const serviceApi = useApiService(); // Renamed to avoid conflict with imported api
  
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
    let isMounted = true;
    let connectionCheckTimeout: NodeJS.Timeout | null = null;
    
    const checkConnection = async () => {
      if (!isMounted) return;

      if (!serviceApi || typeof serviceApi.getChatSessions !== 'function') {
        console.warn("API service not ready in checkConnection. Retrying soon.");
        if (isMounted) {
            connectionCheckTimeout = setTimeout(checkConnection, 1000); // Retry after 1 sec
        }
        return;
      }
      
      setConnectionStatus('connecting');
      try {
        // First, try getConfig as it's simpler
        if (typeof serviceApi.getConfig === 'function') {
          await serviceApi.getConfig();
        } else {
          // Fallback to getChatSessions if getConfig is not available (e.g. older API service structure)
          await serviceApi.getChatSessions();
        }
        
        if (!isMounted) return;
        setConnectionStatus('connected');
        // Assuming getConfig or getChatSessions might return version info or imply connection
        setBackendVersion('connected'); // Simplified version display
        
        // Schedule next check
        connectionCheckTimeout = setTimeout(checkConnection, 30000); // Check every 30 seconds
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
        
        // Retry connection check more frequently if disconnected
        connectionCheckTimeout = setTimeout(checkConnection, 60000); // Retry after 1 minute if disconnected
      }
    };
    
    checkConnection();
    
    return () => {
      isMounted = false;
      if (connectionCheckTimeout) {
        clearTimeout(connectionCheckTimeout);
      }
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

  // Auto-scroll to bottom of messages
  useEffect(() => {
    if (messagesContainerRef.current) {
      messagesContainerRef.current.scrollTop = messagesContainerRef.current.scrollHeight;
    }
  }, [messages]); // Re-check: should be visibleMessages or a combination

  // Fetch chat history from backend or localStorage
  useEffect(() => {
    const loadChatSessions = async () => {
      // Ensure serviceApi and its methods are available
      if (connectionStatus !== 'connected' || !serviceApi || typeof serviceApi.getChatSessions !== 'function') {
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
        const sessions = await serviceApi.getChatSessions();
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
    let isMounted = true;
    let loadFilesTimeout: NodeJS.Timeout | null = null;
    let loadAttempts = 0;
    const maxAttempts = 3;
    
    const loadAvailableFiles = async () => {
      if (!isMounted || connectionStatus !== 'connected' || !serviceApi || typeof serviceApi.getFiles !== 'function') return;
      
      try {
        if (loadAttempts >= maxAttempts) return; // Max retries reached
        loadAttempts++;
        
        const filesData = await serviceApi.getFiles();
        if (!isMounted) return;
        
        // Handle file data (logging is fine, actual state update for files might be elsewhere or not needed here)
        if (filesData && filesData.files && Array.isArray(filesData.files)) {
          console.log(`Loaded ${filesData.files.length} files from backend`);
        } else if (Array.isArray(filesData)) { // If API returns array directly
          console.log(`Loaded ${filesData.length} files from backend`);
        }
        
        // Schedule next refresh
        if (isMounted) {
          loadFilesTimeout = setTimeout(loadAvailableFiles, 300000); // Refresh every 5 minutes
        }
      } catch (error) {
        if (!isMounted) return;
        console.error("Error loading files from backend:", error);
        // Exponential backoff for retries
        const backoff = Math.min(15000, 2000 * Math.pow(2, loadAttempts));
        loadFilesTimeout = setTimeout(loadAvailableFiles, backoff);
      }
    };
    
    if (connectionStatus === 'connected' && serviceApi) {
      // Initial load with a short delay
      loadFilesTimeout = setTimeout(loadAvailableFiles, 1000);
    }
    
    return () => {
      isMounted = false;
      if (loadFilesTimeout) clearTimeout(loadFilesTimeout);
    };
  }, [connectionStatus, serviceApi]);

  // Generate a new chat ID
  const generateNewChatId = () => {
    return `chat_${Math.random().toString(36).substring(2, 10)}_${Date.now()}`;
  };
  
  // Start a new chat session
  const startNewSession = useCallback(async () => {
    const newChatId = generateNewChatId();
    setChatId(newChatId); // Ensure chatId state is updated for FileSelector if needed
    const newSessionData: ChatSession = {
      id: newChatId,
      name: `Chat ${new Date().toLocaleTimeString()}`,
      messages: [{
        id: uuidv4(),
        sender: 'assistant', // Ensure sender is correctly typed
        content: "Hello! How can I assist you today?",
        timestamp: Date.now()
      }],
      lastModified: Date.now(),
      selectedFiles: [] // Initialize with empty selected files
    };
    
    setCurrentSession(newSessionData);
    setMessages(newSessionData.messages); // Set messages from the new session
    setUploadedFile(null);
    setDocumentSummary(null);
    setSelectedFilesList([]); // Reset selected files list for new session
    
    // Save to backend if connected, otherwise save locally
    if (connectionStatus === 'connected' && serviceApi && typeof serviceApi.createChatSession === 'function') {
      try {
        await serviceApi.createChatSession(newSessionData);
      } catch (error) {
        console.error("Error saving new chat session to backend:", error);
        saveChatSession(newSessionData); // Fallback to local
        toast({ title: "Offline Mode", description: "New session saved locally.", variant: "default" });
      }
    } else {
      saveChatSession(newSessionData);
    }
  }, [connectionStatus, serviceApi, toast]);

  // Load chat history
  const loadChatHistory = useCallback(async (id: string) => {
    // Ensure serviceApi and its methods are available
    if (!serviceApi || typeof serviceApi.getChatSessions !== 'function') {
        console.warn('loadChatHistory: API service not fully initialized, falling back to local data');
        const localHistory = getLocalChatHistory();
        const localSession = localHistory.find(s => s.id === id);
        if (localSession) {
          setCurrentSession(localSession);
          setSelectedFilesList(localSession.selectedFiles || []);
        } else {
          await startNewSession(); // await to ensure session is set
        }
        setIsLoading(false);
        return;
    }

    try {
      setIsLoading(true);
      const sessions = await serviceApi.getChatSessions(); // Assuming this returns ChatSession[]
      // Find the session by ID
      const session = sessions.find((s: any) => s.id === id); // Use 'any' if structure is uncertain
      
      if (session) {
        setCurrentSession(session);
        setSelectedFilesList(session.selectedFiles || []);
      } else {
        // If not found on backend, try local
        const localHistory = getLocalChatHistory();
        const localSession = localHistory.find(s => s.id === id);
        if (localSession) {
          setCurrentSession(localSession);
          setSelectedFilesList(localSession.selectedFiles || []);
        } else {
          await startNewSession(); // await
        }
      }
    } catch (error) {
      console.error("Error loading chat history:", error);
      toast({ title: "Error", description: "Failed to load chat history.", variant: "destructive" });
      await startNewSession(); // await
    } finally {
      setIsLoading(false);
    }
  }, [serviceApi, toast, startNewSession]);


  // Effect for initializing chat: runs when initialChatId, loadChatHistory, or startNewSession changes.
  useEffect(() => {
    const initChat = async () => {
      if (initialChatId) {
        await loadChatHistory(initialChatId);
      } else if (!currentSession) { // Only start new session if no current one and no initialChatId
        await startNewSession();
      }
    };
    initChat();
  }, [initialChatId, loadChatHistory, startNewSession, currentSession]);


  // Handle file selection change
  const handleFileSelectionChange = useCallback(async (files: string[]) => {
    // Prevent update if selection hasn't changed
    const currentFilesString = JSON.stringify(selectedFilesList);
    const newFilesString = JSON.stringify(files);
    if (currentFilesString === newFilesString) return;
    
    setSelectedFilesList(files);
    
    if (currentSession) {
      const updatedSession = { ...currentSession, selectedFiles: files, lastModified: Date.now() };
      setCurrentSession(updatedSession);
      saveChatSession(updatedSession); // Save locally immediately
      
      // Notify user of selection change
      if (currentFilesString !== newFilesString && files.length > 0) {
        toast({ title: "Files Selected", description: `${files.length} file${files.length !== 1 ? 's' : ''} selected.`, variant: "default" });
      }
      
      // Update backend if connected
      if (connectionStatus === 'connected' && serviceApi && typeof serviceApi.updateFileSelection === 'function') {
        try {
          await serviceApi.updateFileSelection({ sessionId: currentSession.id, files: files });
        } catch (error) {
          // Log error but don't block UI, local save is primary
          console.warn("Background file selection update to backend failed:", error);
        }
      }
    }
  }, [selectedFilesList, currentSession, connectionStatus, serviceApi, toast]);

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
      const userMessage: ChatMessage = { id: Date.now().toString(), sender: "user", content: input, timestamp: Date.now() };
      
      // Optimistically update UI with user's message
      setMessages(prevMessages => {
        const newMessages = [...prevMessages, userMessage];
        if (currentSession) {
          const updatedSession = { ...currentSession, messages: newMessages, lastModified: Date.now() };
          setCurrentSession(updatedSession);
          saveChatSession(updatedSession); // Save updated session locally
        }
        return newMessages;
      });
      
      const sentInput = input; // Store input before clearing
      setInput(""); // Clear input field immediately
      
      // If connected, send to backend
      if (connectionStatus === 'connected' && serviceApi && typeof serviceApi.sendChatMessage === 'function') {
        // Apply settings to the request payload
        const requestPayload = applySettingsToRequest({
          message: sentInput,
          selected_files: selectedFilesList,
          sessionId: chatId // Use the current chatId state
        });

        const response = await serviceApi.sendChatMessage(requestPayload);
        
        // Handle different response structures
        if (response && response.all_messages && Array.isArray(response.all_messages)) {
          setMessages(response.all_messages); // Backend provides full history
        } else if (response && response.message) {
          // Add AI message if only single response is returned
          const aiMessage: ChatMessage = { id: Date.now().toString(), sender: "assistant", content: response.message, timestamp: Date.now() };
          setMessages(prevMessages => [...prevMessages, aiMessage]);
        }
        
        // Update chat name if backend suggests one
        if (response && response.chat_name && currentSession) {
          setCurrentSession({ ...currentSession, name: response.chat_name });
        }
      } else {
        // Offline: Add a placeholder response
        const offlineResponse: ChatMessage = { id: Date.now().toString(), sender: "assistant", content: "Offline. Message saved locally.", timestamp: Date.now(), error: true };
        setMessages(prevMessages => [...prevMessages, offlineResponse]);
      }
      
      // Ensure scroll to bottom after messages update
      setTimeout(() => { messagesContainerRef.current?.scrollTo({ top: messagesContainerRef.current.scrollHeight, behavior: 'smooth' }); }, 100);
      
    } catch (error) {
      console.error("Error sending message:", error);
      const errorMessage: ChatMessage = { id: Date.now().toString(), sender: "assistant", content: "Error processing message.", timestamp: Date.now(), error: true };
      setMessages(prevMessages => [...prevMessages, errorMessage]);
      toast({ title: "Error", description: "Failed to send message.", variant: "destructive" });
    } finally {
      setIsWaitingForResponse(false);
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
      
      // Process the document using returned fileId (which should be filename)
      setFileProcessingStep("Processing document...");
      const result = await serviceApi.processDocument({ fileId: response.filename, fileName: response.fileName }); // Ensure correct params
      
      setDocumentSummary(result.summary || "Document processed."); // Display summary
      
      // Auto-select file if setting is enabled
      const uploadSettings = loadUploadSettings();
      if (uploadSettings.autoAddToChat) {
        handleFileSelectionChange([...selectedFilesList, response.filename]); // Use filename from response
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
      
      // Use the filename from the upload response (which is fileId in some contexts)
      const backendFilename = uploadResponse.filename; 
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

  const [chatId, setChatId] = useState<string>(initialChatId || generateNewChatId());
  
  // Handle selected files prop change
  useEffect(() => {
    // Only update if the prop actually changes and is different from current state
    if (selectedFiles && JSON.stringify(selectedFiles) !== JSON.stringify(selectedFilesList)) {
      handleFileSelectionChange(selectedFiles);
    }
  }, [selectedFiles, handleFileSelectionChange, selectedFilesList]);
  
  return (
    <div className="flex flex-col flex-1 h-full w-full overflow-hidden pt-3 pb-4">
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
          
          {selectedFilesList.length > 0 && (
            <div className="mt-1 p-1 rounded-md bg-secondary/20 text-secondary-foreground">
              <div className="flex flex-wrap gap-1">
                {selectedFilesList.slice(0,3).map((filename, index) => ( // Show only first 3
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
              {isLoading && ( // General loading for session loading, not message sending
                <div className="flex justify-center p-4">
                  <LoadingSpinner size="md" />
                </div>
              )}
               {isWaitingForResponse && ( // Specific spinner for when AI is responding
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
                  ? "Backend disconnected." 
                  : (isSummarizing ? "Processing..." : "Type your message...")
              }
              className="flex-1 h-7 text-xs glassmorphic"
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSendMessage(e as unknown as FormEvent);
                }
              }}
              disabled={isWaitingForResponse || isSummarizing} // Disable when waiting for response or summarizing
            />
            <Button 
              type="submit" 
              size="icon" 
              disabled={isWaitingForResponse || isSummarizing || !input.trim()} 
              aria-label="Send message"
              className={`h-7 w-7 transition-all bg-gradient-to-l from-[#FF297F] to-[#4B8FFF] ${(isWaitingForResponse || isSummarizing || !input.trim()) ? "opacity-50" : "opacity-100 hover:scale-105"}`}
            >
              {isWaitingForResponse ? <LoadingSpinner size="sm" /> : <Send className="h-3 w-3" />}
            </Button>
          </form>
        </CardFooter>
      </Card>
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
  sender: "user" | "assistant"; 
  content: string;
  timestamp: number;
  citations?: string[];
  error?: boolean;
}

