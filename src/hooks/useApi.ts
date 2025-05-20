import { useState, useEffect, useCallback, useRef } from 'react';
import { api, ChatData } from '../lib/api';

export function useChatHistory() {
  const [data, setData] = useState<ChatData[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const isMounted = useRef(true);
  const lastFetchTime = useRef(0);
  const minimumRefreshInterval = 5000; // 5 seconds

  // Use useCallback to prevent the function from being recreated on each render
  const fetchData = useCallback(async (force = false) => {
    const now = Date.now();
    // Prevent frequent refetching unless forced
    if (!force && loading) return;
    if (!force && now - lastFetchTime.current < minimumRefreshInterval) return;
    
    setLoading(true);
    try {
      lastFetchTime.current = now;
      const result = await api.getChatHistory();
      if (isMounted.current) {
        setData(result);
        setError(null);
      }
    } catch (err) {
      console.error('Error in useChatHistory hook:', err);
      if (isMounted.current) {
        setError(err instanceof Error ? err : new Error(String(err)));
      }
    } finally {
      if (isMounted.current) {
        setLoading(false);
      }
    }
  }, [loading]); // Only depend on loading state

  // Set up the ref cleanup
  useEffect(() => {
    isMounted.current = true;
    return () => {
      isMounted.current = false;
    };
  }, []);
  
  // This effect runs only once when the component mounts
  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Function to manually refresh data when needed
  const refresh = useCallback(() => {
    api.clearCache('chatHistory');
    fetchData(true);
  }, [fetchData]);

  return { 
    chatHistory: data, 
    loading, 
    error, 
    refresh 
  };
}

// Add more custom hooks for other API methods as needed
