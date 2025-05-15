
"use client";

import { useState, useRef, useEffect, FormEvent } from "react";
import type { ChangeEvent } from 'react';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Paperclip, Send, FileText, Trash2, Edit3, Check, X } from "lucide-react";
import LoadingSpinner from "@/components/ui/loading-spinner";
import type { ChatMessage, ChatSession } from "@/types";
import { ragBasedChat, RagBasedChatInput } from "@/ai/flows/rag-based-chat";
import { summarizeDocument, SummarizeDocumentInput } from "@/ai/flows/summarize-document";
import { useToast } from "@/hooks/use-toast";
import { v4 as uuidv4 } from 'uuid'; // For generating unique IDs

// Mock function for saving/loading chat history from localStorage
const CHAT_HISTORY_KEY = "nebulaChatHistory";

const saveChatSession = (session: ChatSession) => {
  try {
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
    const historyJson = localStorage.getItem(CHAT_HISTORY_KEY);
    return historyJson ? JSON.parse(historyJson) : [];
  } catch (error) {
    console.error("Error loading chat history:", error);
    return [];
  }
};


export default function ChatPageClient() {
  const [currentSession, setCurrentSession] = useState<ChatSession | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [documentContent, setDocumentContent] = useState<string | null>(null);
  const [documentSummary, setDocumentSummary] = useState<string | null>(null);
  const [isSummarizing, setIsSummarizing] = useState(false);

  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { toast } = useToast();

  useEffect(() => {
    // Auto-scroll to bottom of messages
    if (scrollAreaRef.current) {
      const viewport = scrollAreaRef.current.querySelector('div[data-radix-scroll-area-viewport]');
      if (viewport) {
        viewport.scrollTop = viewport.scrollHeight;
      }
    }
  }, [messages]);

  useEffect(() => {
    // Start a new session or load the latest one
    const history = loadChatHistory();
    if (history.length > 0 && !currentSession) {
      // For now, let's start a new session by default or if query param indicates
      // To load a specific session, we'd need routing/query params.
      startNewSession();
    } else if (!currentSession) {
      startNewSession();
    }
  }, []);


  useEffect(() => {
    if (currentSession) {
      setMessages(currentSession.messages);
      if (currentSession.documentContext) {
        setDocumentSummary(currentSession.documentContext.summary || null);
        // Note: We don't re-set uploadedFile or documentContent from history for simplicity
        // to avoid re-uploading. Context is mainly for display.
      }
    }
  }, [currentSession]);

  const startNewSession = () => {
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
    saveChatSession(newSession);
  };

  const handleSendMessage = async (e: FormEvent) => {
    e.preventDefault();
    if (!input.trim() && !uploadedFile) return;

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

      if (documentContent) {
        const ragInput: RagBasedChatInput = {
          documentContent: documentContent,
          query: input,
        };
        const ragOutput = await ragBasedChat(ragInput);
        botResponseContent = ragOutput.answer;
        // Assuming RAG flow includes citation parsing, otherwise this needs more logic
        // For now, let's pretend citations are part of the answer string or a separate field
      } else {
         // Fallback or general knowledge chat if no document
        botResponseContent = "Please upload a document to ask questions about its content, or I can try to answer generally.";
        // Simulate a generic response if no RAG context
        // const genericResponse = await someGenericChatFlow({ query: input });
        // botResponseContent = genericResponse.answer;
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
        const updatedSession = { ...currentSession, messages: finalMessages, lastModified: Date.now() };
        setCurrentSession(updatedSession);
        saveChatSession(updatedSession);
      }

    } catch (error) {
      console.error("Error with AI:", error);
      const errorMessage: ChatMessage = {
        id: uuidv4(),
        sender: "bot",
        content: "Sorry, an error occurred while processing your request.",
        timestamp: Date.now(),
      };
      setMessages(prev => [...prev, errorMessage]);
      toast({
        title: "AI Error",
        description: "Could not get a response from the AI.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileChange = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setUploadedFile(file);
      setIsSummarizing(true);
      setDocumentSummary(null); // Clear previous summary

      const reader = new FileReader();
      reader.onload = async (e) => {
        const fileDataUri = e.target?.result as string;
        // For RAG, we might need raw text. For summarization, data URI is fine.
        // This is a simplified text extraction. Real-world might need more robust parsing.
        if (file.type === "text/plain") {
          setDocumentContent(await file.text());
        } else {
          // For non-plain text, we'll just use the URI for summarization
          // and won't set documentContent for RAG as it expects raw text.
          // This part can be expanded with PDF/DOCX parsers.
           toast({ title: "File Type", description: "For RAG, plain text files work best. Summarization will proceed.", variant: "default" });
        }

        try {
          const summaryInput: SummarizeDocumentInput = { documentDataUri: fileDataUri };
          const summaryOutput = await summarizeDocument(summaryInput);
          setDocumentSummary(summaryOutput.summary);
          toast({
            title: "File Processed",
            description: `${file.name} summarized successfully.`,
          });
          if (currentSession) {
            const updatedSession = {
              ...currentSession,
              documentContext: { fileName: file.name, summary: summaryOutput.summary },
              lastModified: Date.now()
            };
            setCurrentSession(updatedSession);
            saveChatSession(updatedSession);
          }
        } catch (error) {
          console.error("Error summarizing document:", error);
          toast({
            title: "Summarization Error",
            description: `Could not summarize ${file.name}.`,
            variant: "destructive",
          });
          setDocumentSummary("Failed to summarize the document.");
        } finally {
          setIsSummarizing(false);
        }
      };
      reader.readAsDataURL(file);
    }
  };

  const triggerFileDialog = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className="flex flex-col h-full max-h-[calc(100vh-var(--header-height,0px)-3rem)]">
      <Card className="flex-1 flex flex-col glassmorphic border-0 shadow-2xl animated-smoke-bg">
        <CardHeader className="p-4 border-b border-border/50">
          <div className="flex justify-between items-center">
            <CardTitle className="text-lg">
              {currentSession?.name || "Nebula Chat"}
            </CardTitle>
            <Button variant="outline" size="sm" onClick={startNewSession}>New Chat</Button>
          </div>
          {uploadedFile && (
            <div className="mt-2 p-2 rounded-md bg-secondary/50 text-xs text-secondary-foreground flex items-center justify-between">
              <div className="flex items-center gap-2">
                <FileText size={16} />
                <span>{uploadedFile.name}</span>
                {isSummarizing && <LoadingSpinner size="sm" dotClassName="bg-primary-foreground" />}
              </div>
              <Button variant="ghost" size="icon" className="h-6 w-6" onClick={() => { setUploadedFile(null); setDocumentContent(null); setDocumentSummary(null); }}>
                <Trash2 size={14} />
              </Button>
            </div>
          )}
          {documentSummary && !isSummarizing && (
             <details className="mt-2 text-xs p-2 rounded-md bg-muted/30">
                <summary className="cursor-pointer font-medium">View Document Summary</summary>
                <p className="mt-1 text-muted-foreground whitespace-pre-wrap">{documentSummary}</p>
             </details>
          )}
        </CardHeader>
        <ScrollArea className="flex-1 p-4" ref={scrollAreaRef}>
          <div className="space-y-6">
            {messages.map((msg) => (
              <div
                key={msg.id}
                className={`flex ${
                  msg.sender === "user" ? "justify-end" : "justify-start"
                }`}
              >
                <div
                  className={`max-w-[70%] p-3 rounded-xl shadow-md text-sm md:text-base ${
                    msg.sender === "user"
                      ? "bg-primary text-primary-foreground rounded-br-none"
                      : "bg-secondary text-secondary-foreground rounded-bl-none glassmorphic"
                  }`}
                >
                  <p className="whitespace-pre-wrap">{msg.content}</p>
                  {msg.citations && msg.citations.length > 0 && (
                    <div className="mt-2 border-t border-primary-foreground/20 pt-1">
                      <p className="text-xs font-semibold">Citations:</p>
                      <ul className="list-disc list-inside text-xs">
                        {msg.citations.map((cite, i) => <li key={i}>{cite}</li>)}
                      </ul>
                    </div>
                  )}
                   <p className={`text-xs mt-1 ${msg.sender === 'user' ? 'text-primary-foreground/70 text-right' : 'text-muted-foreground/70 text-left'}`}>
                    {new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </p>
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="flex justify-start">
                 <div className="max-w-[70%] p-3 rounded-xl shadow-md bg-secondary text-secondary-foreground rounded-bl-none glassmorphic">
                    <LoadingSpinner />
                 </div>
              </div>
            )}
          </div>
        </ScrollArea>
        <CardFooter className="p-4 border-t border-border/50">
          <form onSubmit={handleSendMessage} className="flex w-full items-start gap-2">
            <Button type="button" variant="outline" size="icon" onClick={triggerFileDialog} aria-label="Attach file">
              <Paperclip className="h-5 w-5" />
            </Button>
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              className="hidden"
              accept=".txt,.pdf,.md,.docx" 
            />
            <Textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={uploadedFile ? `Ask about ${uploadedFile.name}...` : "Type your message or upload a file..."}
              className="flex-1 resize-none min-h-[40px] max-h-[120px] glassmorphic"
              rows={1}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSendMessage(e as unknown as FormEvent);
                }
              }}
              disabled={isLoading || isSummarizing}
            />
            <Button type="submit" size="icon" disabled={isLoading || isSummarizing} aria-label="Send message">
              <Send className="h-5 w-5" />
            </Button>
          </form>
        </CardFooter>
      </Card>
    </div>
  );
}

    