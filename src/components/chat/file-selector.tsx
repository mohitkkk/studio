"use client";

import { useState, useEffect, useRef, useCallback } from 'react';
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { FileText, XCircle, Filter, FolderOpen, RefreshCw, AlertCircle, Search, X } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { useToast } from "@/hooks/use-toast";
import { useApiService } from "@/hooks/use-api-service";
import LoadingSpinner from "@/components/ui/loading-spinner";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import HorizontalScroll from './horizontalscroll';


// Define FileMetadata type
interface FileMetadata {
  filename: string;
  description: string;
  tags: string[];
  added_date: string;
  updated_date: string;
  suggested_questions?: Array<{question: string}>;
}

interface FileSelectorProps {
  selectedFiles: string[];
  onSelectionChange: (files: string[]) => void;
  onClose: () => void;
}

export default function FileSelector({ selectedFiles, onSelectionChange, onClose }: FileSelectorProps) {
  const [files, setFiles] = useState<FileMetadata[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [activeTag, setActiveTag] = useState<string | null>(null);
  const [selectedFileIds, setSelectedFileIds] = useState<Set<string>>(new Set(selectedFiles));
  const [apiError, setApiError] = useState<string | null>(null);
  const [retryCount, setRetryCount] = useState(0);
  const { toast } = useToast();
  const api = useApiService();
  const apiCallInProgress = useRef(false);
  
  // Get all unique tags from files
  const allTags = Array.from(
    new Set(
      files.flatMap(file => file.tags || [])
    )
  ).sort();

  // Check if any files were filtered by search or tag
  const hasFilteredFiles = searchTerm || activeTag;

  // Clear existing API errors when retrying or changing filters
  useEffect(() => {
    if (apiError) setApiError(null);
  }, [activeTag, retryCount, apiError]);

  // Use useCallback to stabilize the loadFiles function, but with correct dependencies
  const loadFiles = useCallback(async () => {
    if (apiCallInProgress.current) return;
    apiCallInProgress.current = true;
    setIsLoading(true);
    setApiError(null);
    
    let isMounted = true; // Track if component is still mounted
    
    try {
      // Add timeout handling and retry logic
      const filesData = await Promise.race([
        api.getFiles(activeTag || undefined),
        new Promise((_, reject) => 
          setTimeout(() => reject(new Error('Request timeout')), 10000)
        )
      ]);
      
      // Only update state if component is still mounted
      if (!isMounted) return;
      
      // Handle standard API response format (files inside files property)
      if (filesData && filesData.files && Array.isArray(filesData.files)) {
        setFiles(filesData.files);
      } 
      // Handle direct array format
      else if (Array.isArray(filesData)) {
        setFiles(filesData);
      }
      // Check for empty response
      else if (filesData && Object.keys(filesData).length === 0) {
        setFiles([]);
        setApiError('No files returned from server');
      }
      // Handle empty results
      else {
        setFiles([]);
        setApiError('Received unexpected data format from server');
      }
    } catch (error: any) {
      if (!isMounted) return;
      
      // Provide specific error messages based on error type
      if (error.message?.includes('Network Error')) {
        setApiError('Network error - Unable to connect to server');
      } else if (error.response?.status === 404) {
        setApiError('API endpoint not found');
      } else if (error.response?.status === 500) {
        setApiError('Server error - Please try again later');
      } else {
        setApiError('Failed to connect to server');
      }
      
      setFiles([]);
    } finally {
      if (isMounted) {
        setIsLoading(false);
      }
      setTimeout(() => {
        apiCallInProgress.current = false;
      }, 1000);
    }
    
    return () => {
      isMounted = false;
    };
  }, [activeTag, api]); // Include only stable dependencies

  // Modify the useEffect to avoid infinite loops
  useEffect(() => {
    // Skip if an API call is already in progress
    if (apiCallInProgress.current) return;
    
    loadFiles();
    
    // No cleanup needed here as loadFiles manages its own mounted state
    return () => {
      // Component unmount cleanup is handled by the isMounted ref in loadFiles
    };
  }, [loadFiles, retryCount]); // Only depend on stable callbacks and triggers

  // Filter files based on search term
  const filteredFiles = files.filter(file => 
    file.filename.toLowerCase().includes(searchTerm.toLowerCase()) || 
    file.description?.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // Modify the toggleFileSelection function to avoid unnecessary re-renders
  const toggleFileSelection = useCallback((filename: string) => {
    setSelectedFileIds(prevSelection => {
      const updatedSelection = new Set(prevSelection);
      
      if (updatedSelection.has(filename)) {
        updatedSelection.delete(filename);
      } else {
        updatedSelection.add(filename);
      }
      
      return updatedSelection;
    });
  }, []);

  // Apply selection and close
  const handleApply = () => {
    onSelectionChange(Array.from(selectedFileIds));
    onClose();
  };

  // Handle retry logic
  const handleRetry = () => {
    setRetryCount(count => count + 1);
    apiCallInProgress.current = false;
  };

  // Clean displayed filename (remove timestamp prefix)
  const cleanFilename = (filename: string) => {
    return filename.replace(/^\d+_/, '');
  };

  const toggleFile = useCallback((filename: string) => {
    setSelectedFileIds(prevSelection => {
      const updatedSelection = new Set(prevSelection);
      
      if (updatedSelection.has(filename)) {
        updatedSelection.delete(filename);
      } else {
        updatedSelection.add(filename);
      }
      
      // Notify parent component about the change
      onSelectionChange(Array.from(updatedSelection));
      
      return updatedSelection;
    });
  }, [onSelectionChange]);

  return (
    <div className="flex flex-col h-full">
      <div className="p-4 border-b">
        <div className="relative">
          <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search files..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-8"
          />
        </div>
        
        {selectedFiles.length > 0 && (
          <div className="flex flex-wrap gap-1 mt-3">
            <div className="mr-2 text-sm text-muted-foreground">Selected:</div>
            {selectedFiles.map(file => {
              // Get display name (remove timestamp prefix if present)
              const displayName = file.split('_').slice(1).join('_') || file;
              const toggleFile = useCallback((file: string) => {
                // Create a new array without the file we want to remove
                const updatedSelection = selectedFiles.filter(f => f !== file);
                
                // Update the internal state
                setSelectedFileIds(new Set(updatedSelection));
                
                // Call the callback with the updated array
                onSelectionChange(updatedSelection);
              }, [selectedFiles, onSelectionChange]);

              return (
                <Badge key={file} variant="secondary" className="flex items-center gap-1">
                  {displayName}
                  <button 
                    onClick={() => toggleFile(file)} 
                    className="ml-1 rounded-full hover:bg-muted"
                  >
                    <X className="h-3 w-3" />
                  </button>
                </Badge>
              );
            })}
          </div>
        )}
      </div>
      
      <ScrollArea className="flex-1">
        {isLoading ? (
          <div className="flex justify-center items-center p-8">
            <div className="loading loading-spinner"></div>
          </div>
        ) : filteredFiles.length === 0 ? (
          <div className="flex flex-col items-center justify-center p-8 text-center">
            <p className="text-muted-foreground">No files found</p>
            {searchTerm && (
              <Button 
                variant="link" 
                onClick={() => setSearchTerm("")}
                className="mt-2"
              >
                Clear search
              </Button>
            )}
          </div>
        ) : (
          <div className="divide-y">
            {filteredFiles.map(file => (
              <div
                key={file.filename}
                className={`p-3 cursor-pointer transition-colors ${
                  selectedFiles.includes(file.filename) 
                    ? "bg-primary/10 hover:bg-primary/15" 
                    : "hover:bg-muted/60"
                }`}
                onClick={() => toggleFile(file.filename)}
              >
                <div className="flex items-start">
                  <FileText className="h-5 w-5 mt-0.5 flex-shrink-0 text-muted-foreground" />
                  <div className="ml-3 flex-1 min-w-0">
                    <div className="font-medium truncate">
                      {file.filename.split('_').slice(1).join('_') || file.filename}
                    </div>
                    {file.description && (
                      <p className="text-sm text-muted-foreground line-clamp-2 mt-0.5">
                        {file.description}
                      </p>
                    )}
                    {file.tags && file.tags.length > 0 && (
                      <div className="flex flex-wrap gap-1 mt-2">
                        {file.tags.slice(0, 3).map((tag: string) => (
                          <Badge key={tag} variant="outline" className="text-xs">
                            {tag}
                          </Badge>
                        ))}
                        {file.tags.length > 3 && (
                          <Badge variant="outline" className="text-xs">
                            +{file.tags.length - 3}
                          </Badge>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </ScrollArea>
      
      {onClose && (
        <div className="p-4 border-t">
          <Button onClick={onClose} className="w-full">
            Done
          </Button>
        </div>
      )}
    </div>
  );
}
