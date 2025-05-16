"use client";

import { useState, useEffect, useRef } from 'react';
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { FileText, XCircle, Filter, FolderOpen, RefreshCw, AlertCircle } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { useToast } from "@/hooks/use-toast";
import { useApiService } from "@/hooks/use-api-service";
import LoadingSpinner from "@/components/ui/loading-spinner";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

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
  }, [activeTag, retryCount]);

  // Load files from API with improved error handling and retry logic
  useEffect(() => {
    // Skip if an API call is already in progress
    if (apiCallInProgress.current) return;
    
    const loadFiles = async () => {
      if (apiCallInProgress.current) return;
      apiCallInProgress.current = true;
      setIsLoading(true);
      setApiError(null);
      
      try {
        console.log(`Fetching files from backend... (attempt #${retryCount + 1}, tag: ${activeTag || 'none'})`);
        const filesData = await api.getFiles(activeTag || undefined);
        console.log('Received files data:', filesData);
        
        // Handle standard API response format (files inside files property)
        if (filesData && filesData.files && Array.isArray(filesData.files)) {
          console.log(`Setting ${filesData.files.length} files from response.files array`);
          setFiles(filesData.files);
        } 
        // Handle direct array format
        else if (Array.isArray(filesData)) {
          console.log(`Setting ${filesData.length} files from direct array`);
          setFiles(filesData);
        }
        // Check for empty response
        else if (filesData && Object.keys(filesData).length === 0) {
          console.warn('Empty response from files endpoint');
          setFiles([]);
          setApiError('No files returned from server');
        }
        // Handle empty results
        else {
          console.warn('Unexpected files data format:', filesData);
          setFiles([]);
          setApiError('Received unexpected data format from server');
        }
      } catch (error: any) {
        console.error("Error loading files:", error);
        
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
        
        toast({
          title: "Error",
          description: "Could not load files. Check server connection.",
          variant: "destructive"
        });
        setFiles([]);
      } finally {
        setIsLoading(false);
        // Release lock after a short delay to prevent rapid re-fetching
        setTimeout(() => {
          apiCallInProgress.current = false;
        }, 1000);
      }
    };
    
    loadFiles();
  }, [activeTag, toast, retryCount]);

  // Filter files based on search term
  const filteredFiles = files.filter(file => 
    file.filename.toLowerCase().includes(searchTerm.toLowerCase()) || 
    file.description?.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // Handle file selection toggle
  const toggleFileSelection = (filename: string) => {
    const updatedSelection = new Set(selectedFileIds);
    
    if (updatedSelection.has(filename)) {
      updatedSelection.delete(filename);
    } else {
      updatedSelection.add(filename);
    }
    
    setSelectedFileIds(updatedSelection);
  };

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

  return (
    <Card className="w-full max-w-[600px] max-h-[80vh] overflow-hidden flex flex-col">
      <CardHeader className="p-3 pb-0">
        <div className="flex justify-between items-center">
          <CardTitle className="text-lg">Select Files</CardTitle>
          <Button variant="ghost" size="icon" onClick={onClose} className="h-8 w-8">
            <XCircle className="h-4 w-4" />
          </Button>
        </div>
        <div className="flex items-center gap-2 mt-2">
          <Input 
            placeholder="Search files..." 
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="flex-1"
          />
        </div>
        
        {/* Tags filter - only show if we have tags */}
        {allTags.length > 0 && (
          <ScrollArea className="whitespace-nowrap py-2 h-8">
            <div className="flex gap-2">
              <Badge 
                variant={activeTag === null ? "default" : "outline"}
                className="cursor-pointer"
                onClick={() => setActiveTag(null)}
              >
                All Files
              </Badge>
              {allTags.map(tag => (
                <Badge 
                  key={tag} 
                  variant={activeTag === tag ? "default" : "outline"}
                  className="cursor-pointer"
                  onClick={() => setActiveTag(tag === activeTag ? null : tag)}
                >
                  {tag}
                </Badge>
              ))}
            </div>
          </ScrollArea>
        )}
      </CardHeader>
      <CardContent className="p-3 flex-1 overflow-hidden">
        {isLoading ? (
          <div className="flex flex-col justify-center items-center h-[200px] gap-4">
            <LoadingSpinner size="lg" />
            <p className="text-sm text-muted-foreground animate-pulse">Loading files...</p>
          </div>
        ) : apiError ? (
          <div className="flex flex-col items-center justify-center h-[200px] p-4">
            <Alert variant="destructive" className="mb-4">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>Error Loading Files</AlertTitle>
              <AlertDescription>{apiError}</AlertDescription>
            </Alert>
            
            <Button 
              onClick={handleRetry}
              variant="outline"
              size="sm"
              className="mt-2 flex items-center gap-2"
            >
              <RefreshCw className="h-4 w-4 mr-1" />
              Retry
            </Button>
            
            <p className="text-xs text-muted-foreground mt-4 text-center">
              Make sure the backend server is running at:<br />
              <code className="bg-muted px-1 py-0.5 rounded text-xs">
                {process.env.NEXT_PUBLIC_API_URL || 'http://13.202.208.115:8000'}
              </code>
            </p>
          </div>
        ) : filteredFiles.length === 0 ? (
          <div className="text-center p-8 text-muted-foreground h-[200px] flex flex-col items-center justify-center">
            {hasFilteredFiles ? (
              <>
                <p className="mb-2">No files match your search criteria</p>
                <Button 
                  variant="ghost" 
                  size="sm" 
                  onClick={() => {
                    setSearchTerm('');
                    setActiveTag(null);
                  }}
                >
                  Clear Filters
                </Button>
              </>
            ) : files.length > 0 ? (
              <p>No files match the selected filter</p>
            ) : (
              <div className="flex flex-col items-center gap-2">
                <p>No files available</p>
                <p className="text-xs">You can upload files to get started</p>
                <Button 
                  size="sm" 
                  variant="outline" 
                  onClick={() => window.open('/upload', '_blank')}
                  className="mt-2"
                >
                  Upload Files
                </Button>
              </div>
            )}
          </div>
        ) : (
          <ScrollArea className="h-[300px] pr-2">
            <div className="space-y-2">
              {filteredFiles.map((file, index) => (
                <div 
                  key={`${file.filename}-${index}`}
                  className={`flex items-start p-2 rounded-md hover:bg-accent/30 transition-colors ${
                    selectedFileIds.has(file.filename) ? "bg-accent/50" : ""
                  }`}
                >
                  <Checkbox
                    checked={selectedFileIds.has(file.filename)}
                    onCheckedChange={() => toggleFileSelection(file.filename)}
                    className="mr-2 mt-1"
                  />
                  <div 
                    className="flex-1 min-w-0 cursor-pointer" 
                    onClick={() => toggleFileSelection(file.filename)}
                  >
                    <div className="flex items-center">
                      <FileText className="h-4 w-4 mr-2 text-muted-foreground" />
                      <span className="font-medium text-sm truncate">
                        {cleanFilename(file.filename)}
                      </span>
                    </div>
                    {file.description && (
                      <p className="text-xs text-muted-foreground truncate mt-1">
                        {file.description.replace(/^AUTO-SUMMARY: \*\*/g, '').replace(/\*\*/g, '').substring(0, 100)}
                        {file.description.length > 100 ? '...' : ''}
                      </p>
                    )}
                    {file.tags && file.tags.length > 0 && (
                      <div className="flex gap-1 mt-1 flex-wrap">
                        {file.tags.slice(0, 3).map((tag, i) => (
                          <Badge key={`${file.filename}-tag-${i}`} variant="outline" className="text-[10px] py-0 px-1">
                            {tag}
                          </Badge>
                        ))}
                        {file.tags.length > 3 && (
                          <Badge variant="outline" className="text-[10px] py-0 px-1">
                            +{file.tags.length - 3} more
                          </Badge>
                        )}
                      </div>
                    )}
                    <p className="text-[10px] text-muted-foreground/80 mt-1">
                      Added: {new Date(file.added_date).toLocaleDateString()}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </ScrollArea>
        )}
      </CardContent>
      <CardFooter className="p-3 pt-0 border-t flex justify-between">
        <div className="text-sm">
          {selectedFileIds.size} file{selectedFileIds.size !== 1 ? 's' : ''} selected
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={() => setSelectedFileIds(new Set())}>
            Clear
          </Button>
          <Button 
            size="sm" 
            onClick={handleApply} 
            disabled={isLoading}
          >
            Apply Selection
          </Button>
        </div>
      </CardFooter>
    </Card>
  );
}
