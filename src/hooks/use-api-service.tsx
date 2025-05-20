import { useState, useEffect, useCallback, useRef } from 'react';
import { api } from '../lib/api';

export function useApiService() {
  // A stable reference to the API
  const apiRef = useRef(api);
  
  // Return the stable API reference
  return apiRef.current;
}

import { FileMetadata } from '../lib/api'; // Make sure to import FileMetadata type

export function useFileList(tag?: string) {
  const [files, setFiles] = useState<FileMetadata[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const api = useApiService();
  
  // Keep track of mounted state to prevent state updates after unmount
  const isMounted = useRef(true);
  
  // Use a ref to track current tag value without causing effect dependencies
  const tagRef = useRef(tag);
  
  // Update ref whenever tag changes
  useEffect(() => {
    tagRef.current = tag;
  }, [tag]);
  
  // Create a stable fetchFiles function that reads from the ref
  const fetchFiles = useCallback(async () => {
    if (isLoading) return;
    
    setIsLoading(true);
    setError(null);
    
    try {
      // Use the current value from the ref
      const currentTag = tagRef.current;
      const data = await api.getFiles(currentTag);
      
      // Only update state if component is still mounted
      if (isMounted.current) {
        setFiles(data);
      }
    } catch (err) {
      // Only update state if component is still mounted
      if (isMounted.current) {
        console.error('Error fetching files:', err);
        setError(err);
      }
    } finally {
      // Only update state if component is still mounted
      if (isMounted.current) {
        setIsLoading(false);
      }
    }
  }, [api]); // Only depends on stable api reference
  
  // Call fetchFiles only once on mount or when tag changes
  useEffect(() => {
    fetchFiles();
    
    // Clean up function to prevent state updates after unmount
    return () => {
      isMounted.current = false;
    };
  }, [fetchFiles, tag]);
  
  // Return a stable refresh function along with data
  const refresh = useCallback(() => {
    api.clearCache(`files_${tagRef.current || 'all'}`);
    fetchFiles();
  }, [api, fetchFiles]);
  
  return { files, isLoading, error, refresh };
}
