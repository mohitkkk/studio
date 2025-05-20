"use client";

import { useState, useEffect, useRef } from 'react';
import { ChatMessage } from './message';
import { Loader2, SendIcon } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { useToast } from '@/components/ui/use-toast';
import { useRouter } from 'next/navigation'; // Changed from 'next/router'
import { Upload } from 'lucide-react';

interface Message {
  content: string;
  isUser: boolean;
  timestamp: string;
  format: 'html' | 'markdown';
}

interface ChatInterfaceProps {
  sessionId?: string;
  selectedFiles: string[];
  initialMessages?: Message[];
  preferredFormat?: 'html' | 'markdown';
}

export function ChatInterface({ 
  sessionId = crypto.randomUUID(), 
  selectedFiles = [],
  initialMessages = [],
  preferredFormat = 'html'
}: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>(initialMessages);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { toast } = useToast();
  const router = useRouter();

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!inputMessage.trim()) return;
    
    // Add user message immediately
    const userMessage: Message = {
      content: inputMessage,
      isUser: true,
      timestamp: new Date().toLocaleTimeString(),
      format: 'html'
    };
    
    setMessages((prev) => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);
    
    try {
      // Make API request to the backend
      const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://13.202.208.115:3000';
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: userMessage.content,
          client_id: sessionId,
          selected_files: selectedFiles,
          format: preferredFormat // Request specific format from backend
        }),
      });
      
      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}`);
      }
      
      const data = await response.json();
      
      // Add the bot response
      const botMessage: Message = {
        content: data.message,
        isUser: false,
        timestamp: new Date().toLocaleTimeString(),
        format: data.format || preferredFormat // Use format from response or fallback to preferred
      };
      
      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      toast({
        title: "Error",
        description: "Failed to send message. Please try again.",
        variant: "destructive"
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Main chat area with messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center space-y-2">
              <h3 className="text-lg font-medium">Start a new conversation</h3>
              <p className="text-muted-foreground text-sm">
                {selectedFiles.length > 0 
                  ? `Selected ${selectedFiles.length} file(s) for context` 
                  : "No files selected yet. You can select files from the sidebar."}
              </p>
            </div>
          </div>
        ) : (
          messages.map((message, index) => (
            <ChatMessage
              key={index}
              content={message.content}
              isUser={message.isUser}
              format={message.format}
              timestamp={message.timestamp}
            />
          ))
        )}
        <div ref={messagesEndRef} />
      </div>
      
      {/* Input area and upload button */}
      <div className="border-t p-4">
        <form onSubmit={handleSubmit} className="flex flex-col gap-3">
          {/* Single upload button instead of selected files section */}
          <div className="flex justify-end">
            <Button 
              type="button" 
              variant="ghost" 
              size="sm"
              onClick={() => router.push('/upload')}
              className="flex items-center gap-2"
            >
              <Upload className="h-4 w-4" />
              Upload Files
            </Button>
          </div>
          
          <div className="flex gap-2">
            <Textarea
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              placeholder="Type your message..."
              className="flex-1 min-h-[60px] max-h-[200px] resize-none"
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSubmit(e);
                }
              }}
              disabled={isLoading}
            />
            <Button type="submit" disabled={isLoading || !inputMessage.trim()}>
              {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <SendIcon className="h-4 w-4" />}
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
}
