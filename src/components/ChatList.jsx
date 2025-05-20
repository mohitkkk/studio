import React from 'react';
import { useChatHistory } from '../hooks/useApi';

export function ChatList() {
  const { chatHistory, loading, error, refresh } = useChatHistory();
  
  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error loading chats</div>;
  
  return (
    <div>
      <button onClick={refresh}>Refresh</button>
      <ul>
        {chatHistory.map(chat => (
          <li key={chat.id}>{chat.title || 'Untitled chat'}</li>
        ))}
      </ul>
    </div>
  );
}
