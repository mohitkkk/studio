"use client";

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { CheckCircle, Circle, Search, X, Tag, RefreshCw, XCircle, AlertTriangle, Play } from 'lucide-react';
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Badge } from "../ui/badge";
import LoadingSpinner from "../ui/loading-spinner";
import { TooltipWrapper as Tooltip } from "../ui/tooltip";
import { useToast } from "../ui/use-toast";

interface FileSelectorProps {
  availableFiles: any[];
  selectedFiles: string[];
  onSelectionChange: (files: string[]) => void;
  isLoading?: boolean;
  onClose?: () => void;
  onRefresh?: () => void;
}

export default function FileSelector({
  availableFiles,
  selectedFiles,
  onSelectionChange,
  isLoading = false,
  onClose,
  onRefresh
}: FileSelectorProps) {
  // State must be defined unconditionally at the top
  const [searchTerm, setSearchTerm] = useState("");
  const [filteredFiles, setFilteredFiles] = useState<any[]>([]);
  const [activeTagFilter, setActiveTagFilter] = useState<string | null>(null);
  const [allTags, setAllTags] = useState<string[]>([]);
  
  // References
  const searchInputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);
  
  // Filter files when search term or available files change
  useEffect(() => {
    if (!availableFiles || !Array.isArray(availableFiles)) {
      setFilteredFiles([]);
      return;
    }

    // Extract all tags from files
    const tags = new Set<string>();
    availableFiles.forEach(file => {
      if (file.tags && Array.isArray(file.tags)) {
        file.tags.forEach((tag: string) => tags.add(tag.toLowerCase()));
      }
    });
    setAllTags(Array.from(tags).sort());

    // Apply filters
    let filtered = [...availableFiles];
    
    // Filter by tag first if active
    if (activeTagFilter) {
      filtered = filtered.filter(file => 
        file.tags && 
        Array.isArray(file.tags) && 
        file.tags.some((tag: string) => tag.toLowerCase() === activeTagFilter.toLowerCase())
      );
    }
    
    // Then apply search term filter
    if (searchTerm) {
      const term = searchTerm.toLowerCase();
      filtered = filtered.filter(file => 
        file.filename.toLowerCase().includes(term) || 
        (file.description && file.description.toLowerCase().includes(term))
      );
    }
    
    setFilteredFiles(filtered);
  }, [searchTerm, availableFiles, activeTagFilter]);
  
  // Toggle selection of a file
  const toggleFileSelection = useCallback((filename: string) => {
    if (selectedFiles.includes(filename)) {
      onSelectionChange(selectedFiles.filter(f => f !== filename));
    } else {
      onSelectionChange([...selectedFiles, filename]);
    }
  }, [selectedFiles, onSelectionChange]);

  // Handle tag filter click
  const handleTagClick = useCallback((tag: string) => {
    setActiveTagFilter(currentTag => currentTag === tag ? null : tag);
  }, []);
  
  // Focus search input on mount
  useEffect(() => {
    if (searchInputRef.current) {
      searchInputRef.current.focus();
    }
  }, []);
  
  // Simplify filename display
  const formatFilename = (filename: string): string => {
    // Remove timestamp prefix if present (e.g., 1627484953_filename.txt)
    const withoutTimestamp = filename.replace(/^\d+_/, '');
    // Decode URL encoding if present
    try {
      return decodeURIComponent(withoutTimestamp);
    } catch (e) {
      return withoutTimestamp;
    }
  };

  const { toast } = useToast();
  const [processingFiles, setProcessingFiles] = useState<Record<string, boolean>>({});
  
  // Check if a file is fully processed/ready
  const isFileReady = useCallback((file: any) => {
    if (!file || !file.pipeline_step) return false;
    
    if (file.pipeline_type === "visual") {
      return file.indexing_complete && 
        ["indexing_complete", "indexing_complete_no_chunks"].includes(file.pipeline_step);
    } else if (file.pipeline_type === "textual") {
      return file.pipeline_step === "textual_file_prepared";
    }
    
    return false;
  }, []);
  
  // Get processing status text and color
  const getProcessingStatus = useCallback((file: any) => {
    if (!file || !file.pipeline_step) return { text: "Unknown", color: "text-gray-500" };
    
    // If the file is being actively processed by us
    if (processingFiles[file.filename]) {
      return { text: "Processing...", color: "text-blue-500" };
    }
    
    // If the file is ready
    if (isFileReady(file)) {
      return { text: "Ready", color: "text-green-500" };
    }
    
    // Visual pipeline states
    if (file.pipeline_type === "visual") {
      switch (file.pipeline_step) {
        case "visual_images_extracted":
          return { text: "Needs processing", color: "text-amber-500" };
        case "layout_analysis_complete":
          return { text: "OCR pending", color: "text-amber-500" };
        case "ocr_extraction_complete":
          return { text: "Indexing pending", color: "text-amber-500" };
        default:
          if (file.pipeline_step.includes("failed")) {
            return { text: "Processing failed", color: "text-red-500" };
          }
          return { text: file.pipeline_step, color: "text-amber-500" };
      }
    }
    
    // Textual pipeline states
    if (file.pipeline_type === "textual") {
      if (file.pipeline_step.includes("failed")) {
        return { text: "Processing failed", color: "text-red-500" };
      }
      return { text: file.pipeline_step, color: "text-amber-500" };
    }
    
    return { text: file.pipeline_step, color: "text-gray-500" };
  }, [isFileReady, processingFiles]);
  
  // Process a stuck document
  const processDocument = async (filename: string, event: React.MouseEvent) => {
    event.stopPropagation(); // Prevent toggleFileSelection
    
    try {
      setProcessingFiles(prev => ({ ...prev, [filename]: true }));
      
      const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3000';
      const response = await fetch(`${API_BASE_URL}/process`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename })
      });
      
      if (response.ok) {
        const data = await response.json();
        toast({
          title: "Processing initiated",
          description: data.message || "Document processing started. This may take a minute.",
          variant: data.status === "success" ? "default" : "destructive"
        });
        
        // Refresh after a delay to see updated status
        if (data.status === "success") {
          setTimeout(() => {
            if (onRefresh) onRefresh();
          }, 2000);
        }
      } else {
        const errorData = await response.json().catch(() => ({ detail: `Server responded with status ${response.status}` }));
        throw new Error(errorData.detail || `Server responded with status ${response.status}`);
      }
    } catch (error) {
      console.error('Error processing document:', error);
      toast({
        title: "Processing failed",
        description: error instanceof Error ? error.message : "Failed to process document",
        variant: "destructive"
      });
    } finally {
      setTimeout(() => {
        setProcessingFiles(prev => ({ ...prev, [filename]: false }));
      }, 3000);
    }
  };

  return (
    <div className="w-full h-full max-h-[70vh] flex flex-col">
      {/* Header with close button */}
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-medium">Select Files</h3>
        {onClose && (
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={onClose}
            className="h-8 w-8 p-0"
            aria-label="Close file selector"
          >
            <XCircle className="h-4 w-4" />
          </Button>
        )}
      </div>
      
      {/* Search & filter bar with Refresh button */}
      <div className="flex items-center gap-2 mb-3">
        <div className="relative flex-1">
          <Search className="absolute left-2 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            ref={searchInputRef}
            type="text"
            placeholder="Search files..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-8 pr-8"
          />
          {searchTerm && (
            <Button
              variant="ghost"
              size="icon"
              className="absolute right-1 top-1/2 transform -translate-y-1/2 h-6 w-6 p-0"
              onClick={() => setSearchTerm("")}
            >
              <X className="h-3 w-3" />
            </Button>
          )}
        </div>
        
        {/* Refresh button */}
        {onRefresh && (
          <Button
            variant="outline"
            size="icon"
            onClick={onRefresh}
            className="h-9 w-9"
            title="Refresh file list"
            disabled={isLoading}
          >
            <RefreshCw className="h-4 w-4" />
          </Button>
        )}
      </div>
      
      {/* Tag filters */}
      {allTags.length > 0 && (
        <div className="flex flex-wrap gap-1 mb-3 max-h-12 overflow-y-auto pb-1 mac-scrollbar">
          {allTags.map((tag) => (
            <Badge 
              key={tag}
              variant={activeTagFilter === tag ? "default" : "outline"}
              className="cursor-pointer text-xs"
              onClick={() => handleTagClick(tag)}
            >
              <Tag className="mr-1 h-2.5 w-2.5" />
              {tag}
              {activeTagFilter === tag && (
                <X className="ml-1 h-2.5 w-2.5" onClick={(e) => {
                  e.stopPropagation();
                  setActiveTagFilter(null);
                }} />
              )}
            </Badge>
          ))}
        </div>
      )}
      
      {/* Loading indicator */}
      {isLoading && (
        <div className="flex justify-center items-center py-8">
          <LoadingSpinner size="md" />
        </div>
      )}
      
      {/* File list - increased height to reduce bottom space */}
      {!isLoading && (
        <div 
          className="flex-1 overflow-y-auto border rounded-md min-h-[200px]"
          ref={listRef}
        >
          {filteredFiles.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full min-h-[200px] text-muted-foreground">
              <p>{availableFiles.length === 0 ? "No files available" : "No matching files found"}</p>
            </div>
          ) : (
            <div className="divide-y">
              {filteredFiles.map((file) => (
                <div 
                  key={file.filename}
                  className={`flex items-start p-3 hover:bg-gray-950 cursor-pointer transition-colors ${
                    selectedFiles.includes(file.filename) ? 'bg-gray-950' : ''
                  }`}
                  onClick={() => toggleFileSelection(file.filename)}
                >
                  <div className="mr-3 mt-0.5">
                    {selectedFiles.includes(file.filename) ? (
                      <CheckCircle className="h-5 w-5 text-primary" />
                    ) : (
                      <Circle className="h-5 w-5 text-muted-foreground" />
                    )}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex justify-between items-start">
                      <p className="font-medium truncate">{formatFilename(file.filename)}</p>
                      <div className="flex items-center ml-2">
                        {!isFileReady(file) && !file.pipeline_step?.includes("failed") && (
                          <Tooltip content="Process document">
                            <Button
                              size="icon"
                              variant="ghost"
                              className="h-6 w-6"
                              onClick={(e) => processDocument(file.filename, e)}
                              disabled={processingFiles[file.filename]}
                            >
                              {processingFiles[file.filename] ? (
                                <LoadingSpinner size="sm" />
                              ) : (
                                <Play className="h-3.5 w-3.5" />
                              )}
                            </Button>
                          </Tooltip>
                        )}
                        <Tooltip content={getProcessingStatus(file).text}>
                          <div className={`text-xs ml-1 ${getProcessingStatus(file).color}`}>
                            {!isFileReady(file) && <AlertTriangle className="h-3.5 w-3.5" />}
                            <span className="ml-1 hidden md:inline">{getProcessingStatus(file).text}</span>
                          </div>
                        </Tooltip>
                      </div>
                    </div>
                    <p className="text-sm text-gray-300 truncate">{file.description || "No description"}</p>
                    {file.tags && Array.isArray(file.tags) && file.tags.length > 0 && (
                      <div className="flex flex-wrap gap-1 mt-1">
                        {file.tags.map((tag: string) => (
                          <Badge key={tag} variant="secondary" className="text-xs">
                            {tag}
                          </Badge>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
      
      {/* Selection summary - adjusted margins to reduce extra space */}
      <div className="mt-2 flex justify-between items-center">
        <div className="text-sm text-muted-foreground">
          {selectedFiles.length} file{selectedFiles.length !== 1 ? 's' : ''} selected
        </div>
        <div className="flex gap-2">
          {selectedFiles.length > 0 && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                const fileList = selectedFiles.map(f => formatFilename(f)).join(", ");
                navigator.clipboard.writeText(`Using files: ${fileList}`);
                // Optional: Add toast notification here if you have a toast system
              }}
              title="Copy selected files as Markdown reference"
            >
              Copy Reference
            </Button>
          )}
          <Button
            variant="outline"
            size="sm"
            onClick={() => onSelectionChange([])}
            disabled={selectedFiles.length === 0}
          >
            Clear Selection
          </Button>
        </div>
      </div>
    </div>
  );
}
