"use client";

import { useState, useRef, useEffect, FormEvent, useCallback } from "react";
import type { ChangeEvent } from 'react';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Paperclip, Send, FileText, Trash2, FolderOpen, XCircle } from "lucide-react";
import LoadingSpinner from "@/components/ui/loading-spinner";
import type { ChatMessage, ChatSession } from "@/types";
import { useToast } from "@/hooks/use-toast";
import { v4 as uuidv4 } from 'uuid';
import { useApiService } from "@/hooks/use-api-service";
import FileSelector from "./file-selector"; 
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import { VisuallyHidden } from "@/components/ui/visually-hidden";

// Mock function for saving/loading chat history from localStorage
const CHAT_HISTORY_KEY = "nebulaChatHistory";

const saveChatSession = (session: ChatSession) => {
  try {
    if (typeof window === 'undefined') return;
    const history = loadChatHistory();
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

const loadChatHistory = (): ChatSession[] => {
  try {
    if (typeof window === 'undefined') return [];
    const historyJson = localStorage.getItem(CHAT_HISTORY_KEY);
    return historyJson ? JSON.parse(historyJson) : [];
  } catch (error) {
    console.error("Error loading chat history:", error);
    return [];
  }
};

export default function ChatPageClient({ selectedFiles = [] }: { selectedFiles?: string[] }) {
  // State variables
  const [currentSession, setCurrentSession] = useState<ChatSession | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [visibleMessages, setVisibleMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [documentContent, setDocumentContent] = useState<string | null>(null);
  const [documentSummary, setDocumentSummary] = useState<string | null>(null);
  const [isSummarizing, setIsSummarizing] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [fileProcessingStep, setFileProcessingStep] = useState<string | null>(null);
  const [isFileDialogOpen, setIsFileDialogOpen] = useState(false);
  const [selectedFilesList, setSelectedFilesList] = useState<string[]>(selectedFiles);
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'disconnected' | 'connecting'>('disconnected');
  const [backendVersion, setBackendVersion] = useState<string | null>(null);

  // Refs
  const fileInputRef = useRef<HTMLInputElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const { toast } = useToast();
  
  // API service hook
  const api = useApiService();

  // Check backend connection on component mount
  useEffect(() => {
    const checkConnection = async () => {
      setConnectionStatus('connecting');
      try {
        const status = await api.checkStatus();
        setConnectionStatus('connected');
        setBackendVersion(status.version);
        
        // Optional: Load any server config
        const config = await api.getConfig();
      } catch (error) {
        console.error("Backend connection failed:", error);
        setConnectionStatus('disconnected');
        toast({
          title: "Connection Error",
          description: "Could not connect to the AI backend. Using offline mode.",
          variant: "destructive",
          duration: 5000,
        });
      }
    };
    
    checkConnection();
  }, []);

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
        const sessions = await api.getChatSessions();
        if (sessions.length > 0) {
          setCurrentSession(sessions[0]); // Load most recent session
        } else {
          startNewSession();
        }
      } catch (error) {
        console.error("Error loading chat sessions:", error);
        // Fallback to localStorage or create new session
        const localSessions = loadChatHistory();
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
    const history = loadChatHistory();
    if (history.length > 0 && !currentSession) {
      startNewSession();
    } else if (!currentSession) {
      startNewSession();
    }
  }, []);

  // Load messages from current session
  useEffect(() => {
    if (currentSession) {
      setMessages(currentSession.messages);
      if (currentSession.documentContext) {
        setDocumentSummary(currentSession.documentContext.summary || null);
      }
    }
  }, [currentSession]);

  // Load available files from backend
  useEffect(() => {
    const loadAvailableFiles = async () => {
      if (connectionStatus !== 'connected') return;
      
      try {
        // Get files from the backend
        const files = await api.getFiles();
        console.log("Available files:", files);
      } catch (error) {
        console.error("Error loading files from backend:", error);
      }
    };
    
    loadAvailableFiles();
  }, [connectionStatus]);

  // Start a new chat session
  const startNewSession = async () => {
    const newSession: ChatSession = {
      id: uuidv4(),
      name: `Chat ${new Date().toLocaleTimeString()}`,
      messages: [{
        id: uuidv4(),
        sender: 'bot',
        content: "Hello! How can I assist you today? Upload a document or ask a question.",
        timestamp: Date.now()
      }],
      lastModified: Date.now(),
    };
    
    setCurrentSession(newSession);
    setMessages(newSession.messages);
    setUploadedFile(null);
    setDocumentContent(null);
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

  // Handle file selection change
  const handleFileSelectionChange = (files: string[]) => {
    setSelectedFilesList(files);
    
    // Update current session with selected files
    if (currentSession) {
      const updatedSession = {
        ...currentSession,
        selectedFiles: files,
        lastModified: Date.now()
      };
      
      setCurrentSession(updatedSession);
      
      // Save to backend if connected
      if (connectionStatus === 'connected') {
        api.updateChatSession(updatedSession)
          .catch(error => {
            console.error("Error updating session with selected files:", error);
            saveChatSession(updatedSession);
          });
      } else {
        saveChatSession(updatedSession);
      }
    }
  };
  
  // Open file selector dialog
  const openFileSelector = () => {
    setIsFileDialogOpen(true);
  };
  
  // Send message to chat
  const handleSendMessage = async (e: FormEvent) => {
    e.preventDefault();
    if (!input.trim() && !uploadedFile && selectedFilesList.length === 0) return;

    const userMessage: ChatMessage = {
      id: uuidv4(),
      sender: "user",
      content: input,
      timestamp: Date.now(),
    };
    
    const updatedMessages = [...messages, userMessage];
    setMessages(updatedMessages);
    setInput("");
    setIsLoading(true);

    try {
      let botResponseContent = "I'm having trouble understanding that.";
      let citations: string[] | undefined;

      if (connectionStatus === 'connected') {
        // Use backend API for chat with selected files
        const response = await api.sendChatMessage({
          sessionId: currentSession?.id,
          message: input,
          selected_files: selectedFilesList // Use the selected files list for the API
        });
        
        botResponseContent = response.answer;
        citations = response.citations;
      } else {
        // Offline fallback
        if (documentContent) {
          botResponseContent = "I can see you've uploaded a document, but I'm currently working offline. Please connect to the backend for complete functionality.";
        } else {
          botResponseContent = "I'm currently in offline mode with limited capabilities. Please check your connection to the backend service.";
        }
      }
      
      const botMessage: ChatMessage = {
        id: uuidv4(),
        sender: "bot",
        content: botResponseContent,
        timestamp: Date.now(),
        citations,
      };
      
      const finalMessages = [...updatedMessages, botMessage];
      setMessages(finalMessages);

      if (currentSession) {
        const updatedSession = { 
          ...currentSession, 
          messages: finalMessages, 
          lastModified: Date.now() 
        };
        setCurrentSession(updatedSession);
        
        // Save locally always to ensure persistence
        saveChatSession(updatedSession);
      }
    } catch (error) {
      console.error("Error with chat API:", error);
      const errorMessage: ChatMessage = {
        id: uuidv4(),
        sender: "bot",
        content: "Sorry, an error occurred while processing your request. Please try again or check your connection.",
        timestamp: Date.now(),
      };
      setMessages(prev => [...prev, errorMessage]);
      toast({
        title: "AI Error",
        description: "Could not get a response from the AI backend.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

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
  const removeUploadedFile = useCallback(async () => {
    if (connectionStatus === 'connected' && uploadedFile && currentSession?.documentContext?.fileId) {
      try {
        await api.removeFile(currentSession.documentContext.fileId);
      } catch (error) {
        console.error("Error removing file from backend:", error);
      }
    }
    
    setUploadedFile(null);
    setDocumentContent(null);
    setDocumentSummary(null);
    
    if (fileInputRef.current) fileInputRef.current.value = "";
    
    if (currentSession) {
      const updatedSession = {
        ...currentSession,
        documentContext: undefined,
        lastModified: Date.now()
      };
      setCurrentSession(updatedSession);
      
      if (connectionStatus === 'connected') {
        await api.updateChatSession(updatedSession);
      } else {
        saveChatSession(updatedSession);
      }
    }
    
    toast({
      title: "File Removed",
      description: "The document has been removed from the conversation.",
    });
  }, [currentSession, uploadedFile, connectionStatus, api]);

  return (
    <div className="flex flex-col h-[100vh] w-full overflow-hidden pt-3 pb-4 ">
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
                <Button variant="ghost" size="icon" className="h-5 w-5" onClick={removeUploadedFile}>
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
              className={`h-7 w-7 transition-all ${(isLoading || !input.trim()) ? "opacity-50" : "opacity-100 hover:scale-105"}`}
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

