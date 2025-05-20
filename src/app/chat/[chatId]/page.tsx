"use client";

import { useEffect, useState } from 'react';
import { ChatInterface } from '@/components/chat/chat-interface';
import { useRouter } from 'next/navigation';

export default function ChatPage({ params }: { params: { chatId: string } }) {
  const [selectedFiles, setSelectedFiles] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const router = useRouter();
  
  // Fetch chat history and selected files when component mounts
  useEffect(() => {
    if (params.chatId === 'new') {
      setIsLoading(false);
      return;
    }
    
    // Fetch chat history and selected files
    const fetchChatData = async () => {
      try {
        const response = await fetch(`/api/chat/${params.chatId}`);
        if (response.ok) {
          const data = await response.json();
          setSelectedFiles(data.selected_files || []);
        } else {
          // Handle invalid chat ID
          router.push('/chat/new');
        }
      } catch (error) {
        console.error('Error fetching chat data:', error);
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchChatData();
  }, [params.chatId, router]);
  
  if (isLoading) {
    return <div className="flex items-center justify-center h-screen">Loading...</div>;
  }
  
  return (
    <div className="container h-screen py-4">
      <ChatInterface 
        sessionId={params.chatId === 'new' ? undefined : params.chatId}
        selectedFiles={selectedFiles}
        preferredFormat="html"
      />
    </div>
  );
}
