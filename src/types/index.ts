
export interface ChatMessage {
  id: string;
  sender: 'user' | 'bot';
  content: string;
  timestamp: number;
  citations?: string[]; // For RAG
}

export interface ChatSession {
  id: string;
  name: string;
  messages: ChatMessage[];
  lastModified: number;
  // Optional: store document context if chat is based on a specific document
  documentContext?: {
    fileName: string;
    summary?: string;
  };
}

export interface EnvVariable {
  id: string;
  key: string;
  value: string;
}

    