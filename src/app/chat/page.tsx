"use client";

import { useEffect, useState } from "react";
import ChatPageClient from "@/components/chat/chat-page-client";
import { api } from "@/lib/api";
import { getApiService, setGlobalApiService } from "@/lib/api-service-utils";

export default function ChatPage() {
  const [initialized, setInitialized] = useState(false);
  const [chatId, setChatId] = useState<string>("");
  
  useEffect(() => {
    // Initialize API integration
    const apiService = getApiService();
    setGlobalApiService(apiService);
    
    // Always create a new chat session when the page loads
    const initNewChat = async () => {
      try {
        // Make sure sendChatMessage method exists
        if (!apiService.sendChatMessage) {
          console.error("Critical method 'sendChatMessage' is missing from API service");
        }
        
        // Call the API endpoint to create a chat session
        let newChat;
        
        try {
          if (api.createNewChat) {
            newChat = await api.createNewChat();
          } else if (apiService.createChatSession) {
            newChat = await apiService.createChatSession();
          } else {
            throw new Error("No method available to create chat session");
          }
        } catch (error) {
          console.error("Failed to create new chat session:", error);
          // Fallback to client-side ID generation if API fails
          newChat = { chat_id: `chat_${Math.random().toString(36).substring(2, 10)}_${Date.now()}` };
        }
        
        setChatId(newChat.chat_id);
      } catch (error) {
        console.error("Failed to initialize chat session:", error);
        // Fallback to client-side ID generation if API fails
        setChatId(`chat_${Math.random().toString(36).substring(2, 10)}_${Date.now()}`);
      } finally {
        setInitialized(true);
      }
    };
    
    initNewChat();
  }, []);
  
  if (!initialized) {
    return <div className="flex items-center justify-center h-screen">Loading...</div>;
  }
  
  return <ChatPageClient key={chatId} initialChatId={chatId} />;
}