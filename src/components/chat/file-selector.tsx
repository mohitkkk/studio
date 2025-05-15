"use client";

import { useState, useEffect } from 'react';
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { FileText, XCircle, Filter, FolderOpen } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { useToast } from "@/hooks/use-toast";
import { useApiService } from "@/hooks/use-api-service";
import type { FileMetadata } from "@/types";
import LoadingSpinner from "@/components/ui/loading-spinner";

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
  const { toast } = useToast();
  const api = useApiService();
  
  // Get all unique tags from files
  const allTags = Array.from(
    new Set(
      files.flatMap(file => file.tags || [])
    )
  ).sort();

  // Load files from API - with improved error handling
  useEffect(() => {
    const loadFiles = async () => {
      setIsLoading(true);
      try {
        console.log('Fetching files from backend...');
        const filesData = await api.getFiles(activeTag || undefined);
        
        // Debug to see what's coming from the API
        console.log('Received files from API:', filesData);
        
        if (Array.isArray(filesData)) {
          setFiles(filesData);
          if (filesData.length === 0) {
            console.log('No files returned from API');
          }
        } else {
          console.error("Invalid files data format:", filesData);
          toast({
            title: "Error",
            description: "Retrieved invalid files data format from server",
            variant: "destructive"
          });
          setFiles([]);
        }
      } catch (error) {
        console.error("Error loading files:", error);
        toast({
          title: "Backend Connection Error",
          description: "Could not connect to the backend server. Check that the server is running.",
          variant: "destructive"
        });
        setFiles([]);
      } finally {
        setIsLoading(false);
      }
    };
    
    loadFiles();
  }, [activeTag, api, toast]);

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
        
        {/* Tags filter */}
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
      </CardHeader>

      <CardContent className="p-3 flex-1 overflow-hidden">
        {isLoading ? (
          <div className="flex justify-center items-center h-[200px]">
            <LoadingSpinner size="lg" />
          </div>
        ) : filteredFiles.length === 0 ? (
          <div className="text-center p-8 text-muted-foreground">
            {searchTerm ? "No files match your search" : (
              <div className="flex flex-col items-center gap-2">
                <p>No files available</p>
                <p className="text-xs">Check your backend connection at:</p>
                <pre className="text-xs bg-muted p-1 rounded">{process.env.NEXT_PUBLIC_API_URL || 'http://13.202.208.115:8000'}</pre>
              </div>
            )}
          </div>
        ) : (
          <ScrollArea className="h-[300px] pr-2">
            <div className="space-y-2">
              {filteredFiles.map((file) => (
                <div 
                  key={file.filename}
                  className={`flex items-start p-2 rounded-md hover:bg-accent/30 transition-colors ${
                    selectedFileIds.has(file.filename) ? "bg-accent/50" : ""
                  }`}
                >
                  <Checkbox
                    checked={selectedFileIds.has(file.filename)}
                    onCheckedChange={() => toggleFileSelection(file.filename)}
                    className="mr-2 mt-1"
                  />
                  <div className="flex-1 min-w-0" onClick={() => toggleFileSelection(file.filename)}>
                    <div className="flex items-center">
                      <FileText className="h-4 w-4 mr-2 text-muted-foreground" />
                      <span className="font-medium text-sm truncate">{file.filename.replace(/^\d+_/, '')}</span>
                    </div>
                    {file.description && (
                      <p className="text-xs text-muted-foreground truncate mt-1">
                        {file.description}
                      </p>
                    )}
                    {file.tags && file.tags.length > 0 && (
                      <div className="flex gap-1 mt-1 flex-wrap">
                        {file.tags.map(tag => (
                          <Badge key={tag} variant="outline" className="text-[10px] py-0 px-1">
                            {tag}
                          </Badge>
                        ))}
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
          {selectedFileIds.size} files selected
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={() => setSelectedFileIds(new Set())}>
            Clear
          </Button>
          <Button size="sm" onClick={handleApply}>
            Apply Selection
          </Button>
        </div>
      </CardFooter>
    </Card>
  );
}
