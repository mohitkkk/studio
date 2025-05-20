import { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { api } from './use-api-service';

export type ConnectionStatus = 'connected' | 'disconnected' | 'connecting';

export function useConnectionStatus(checkInterval = 30000) {
  const [status, setStatus] = useState<ConnectionStatus>('connecting');
  const [version, setVersion] = useState<string | null>(null);
  const isMounted = useRef(true);
  const checkTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const checkConnection = async () => {
    if (!isMounted.current) return;
    
    try {
      setStatus('connecting');
      
      // Try simple axios get request first as most reliable method
      const response = await axios.get('/', { 
        timeout: 5000,
        validateStatus: () => true // Accept any status to avoid throwing
      });
      
      if (!isMounted.current) return;
      
      if (response.status >= 200 && response.status < 300) {
        setStatus('connected');
        setVersion(response.data?.version || 'unknown');
      } else {
        // Fallback to ping method
        try {
          const result = await api.checkStatus();
          if (!isMounted.current) return;
          
          setStatus('connected');
          setVersion(result.version || 'unknown');
        } catch (innerError) {
          if (!isMounted.current) return;
          setStatus('disconnected');
        }
      }
    } catch (error) {
      if (!isMounted.current) return;
      console.error("Backend connection failed:", error);
      setStatus('disconnected');
    }
    
    // Schedule next check if component is still mounted
    if (isMounted.current) {
      const nextInterval = status === 'connected' ? checkInterval : 60000; // Longer retry when disconnected
      checkTimeoutRef.current = setTimeout(checkConnection, nextInterval);
    }
  };
  
  useEffect(() => {
    // Initial check
    checkConnection();
    
    // Cleanup function
    return () => {
      isMounted.current = false;
      if (checkTimeoutRef.current) {
        clearTimeout(checkTimeoutRef.current);
      }
    };
  }, []); // Empty dependency array - only run on mount
  
  return { status, version };
}
