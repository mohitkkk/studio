"use client";

import { useEffect, useState } from "react";
import ChatPageClient from "@/components/chat/chat-page-client";
import { api } from "@/lib/api";

export default function ChatPage() {
  const [initialized, setInitialized] = useState(false);
  const [chatId, setChatId] = useState<string>("");
  
  useEffect(() => {
    // Always create a new chat session when the page loads
    const initNewChat = async () => {
      try {
        // Call the new API endpoint to create a chat session
        const newChat = await api.createNewChat();
        setChatId(newChat.chat_id);
      } catch (error) {
        console.error("Failed to create new chat session:", error);
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
  
  return <ChatPageClient key={chatId} chatId={chatId} />;
}