
"use client";

import { use, useEffect, useState } from 'react'; // Import React.use
import { ChatInterface } from '@/components/chat/chat-interface';
import { useRouter } from 'next/navigation';

interface PageParams {
  chatId: string;
}

// Adjust the params prop type to expect a Promise as per the warning
export default function ChatPage({ params: paramsPromise }: { params: Promise<PageParams> }) {
  // Unwrap the params Promise using React.use()
  // This will suspend the component if the promise is not yet resolved.
  const params = use(paramsPromise);
  const { chatId } = params; // Destructure after resolving

  const [selectedFiles, setSelectedFiles] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(true); // This isLoading is for fetchChatData
  const router = useRouter();

  // Fetch chat history and selected files when component mounts
  useEffect(() => {
    // Now use the resolved `chatId`
    if (chatId === 'new') {
      setIsLoading(false);
      return;
    }

    // Fetch chat history and selected files
    const fetchChatData = async () => {
      setIsLoading(true); // Set loading true at the start of data fetching
      try {
        const response = await fetch(`/api/chat/${chatId}`); // Use resolved chatId
        if (response.ok) {
          const data = await response.json();
          setSelectedFiles(data.selected_files || []);
        } else {
          // Handle invalid chat ID
          router.push('/chat/new');
        }
      } catch (error) {
        console.error('Error fetching chat data:', error);
        // Optionally, redirect or show an error message
        router.push('/chat/new'); // Example: redirect on error
      } finally {
        setIsLoading(false);
      }
    };

    // Only fetch data if chatId is not 'new' and is available
    if (chatId) {
        fetchChatData();
    } else {
        // If chatId is somehow not available after promise resolution (should not happen if promise resolves correctly)
        setIsLoading(false); 
    }

  }, [chatId, router]); // `chatId` is now a dependency from resolved params

  // The component will suspend via `use(paramsPromise)` if params are not ready.
  // The `isLoading` state here primarily refers to the `fetchChatData` operation.
  if (isLoading) {
    return <div className="flex items-center justify-center h-screen">Loading chat data...</div>;
  }

  return (
    <div className="container h-screen py-4">
      <ChatInterface
        sessionId={chatId === 'new' ? undefined : chatId} // Use resolved chatId
        selectedFiles={selectedFiles}
        preferredFormat="html"
      />
    </div>
  );
}
