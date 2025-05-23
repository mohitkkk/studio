'use client'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { v4 as uuidv4 } from 'uuid'
import LoadingSpinner from '@/components/ui/loading-spinner'

// Constants for local storage keys
const CHAT_HISTORY_KEY = "nebulaChatHistory";
const CHAT_INDEX_KEY = "nebulaChatIndex";
const CHAT_COUNTER_KEY = "nebulaChatCounter";

export default function NewChatPage() {
  const router = useRouter()

  // Helper function to get next chat number
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
  }

  // Helper function to get chat index
  const getChatIndex = (): Array<{id: string, name: string, lastModified: number, selectedFiles: string[]}> => {
    try {
      if (typeof window === 'undefined') return [];
      const indexJson = localStorage.getItem(CHAT_INDEX_KEY);
      return indexJson ? JSON.parse(indexJson) : [];
    } catch (error) {
      console.error("Error loading chat index:", error);
      return [];
    }
  }

  useEffect(() => {
    // Create a new chat and redirect to it
    const createAndRedirect = () => {
      try {
        // Generate chat ID and name
        const chatId = `chat_${uuidv4()}`;
        const chatNumber = getNextChatNumber();
        const chatName = `Chat ${chatNumber}`;
        
        // Create welcome message
        const welcomeMessage = {
          id: uuidv4(),
          sender: "assistant",
          content: "<p>Hello! How can I help you today? You can ask me questions about any files you select.</p>",
          timestamp: Date.now()
        };
        
        // Create new session
        const newSession = {
          id: chatId,
          name: chatName,
          messages: [welcomeMessage],
          selectedFiles: [],
          lastModified: Date.now()
        };
        
        // Save session
        localStorage.setItem(`${CHAT_HISTORY_KEY}_${chatId}`, JSON.stringify(newSession));
        
        // Update chat index
        const chatIndex = getChatIndex();
        chatIndex.unshift({
          id: chatId,
          name: chatName,
          lastModified: Date.now(),
          selectedFiles: []
        });
        localStorage.setItem(CHAT_INDEX_KEY, JSON.stringify(chatIndex));
        
        // Dispatch event to update sidebar
        window.dispatchEvent(new Event('chatUpdated'));
        
        // Redirect to the new chat
        router.push(`/chat/${chatId}`);
      } catch (error) {
        console.error("Error creating new chat:", error);
        // Redirect to main chat page in case of error
        router.push('/');
      }
    };

    createAndRedirect();
  }, [router]);

  // Show loading while creating chat and redirecting
  return (
    <div className="flex items-center justify-center h-screen">
      <div className="text-center">
        <LoadingSpinner size="lg" />
        <p className="mt-4 text-muted-foreground">Creating new chat...</p>
      </div>
    </div>
  );
}
