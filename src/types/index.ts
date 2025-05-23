export interface ChatMessage {
  id: string;
  sender: string;  // 'user', 'bot', 'system', or 'assistant'
  content: string;
  timestamp: number;
  citations?: string[];
}

export interface ChatSession {
  id: string;
  name: string;
  messages: Array<{
    id: string;
    sender: "user" | "assistant";
    content: string;
    timestamp: number;
    citations?: string[];
    error?: boolean;
  }>;
  selectedFiles?: string[]; // Add this property
  lastModified: number;
  documentContext?: {
    summary?: string;
  };
}

export interface FileMetadata {
  filename: string;
  description: string;
  tags: string[];
  added_date: string;
  updated_date: string;
  suggested_questions?: Array<{question: string}>;
}

export interface ApiConfig {
  ollama_model: string;
  temperature: number;
  top_p: number;
  top_k_per_file?: number;
  similarity_threshold?: number;
}

export interface ChatRequestOptions {
  temperature?: number;
  top_p?: number;
  remove_hedging?: boolean;
  remove_references?: boolean;
  remove_disclaimers?: boolean;
}

