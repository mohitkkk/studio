"use client";

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Upload } from 'lucide-react';
import { useToast } from '@/components/ui/use-toast';

interface FileUploadButtonProps {
  onFileSelect?: (files: File[]) => void;
  className?: string;
}

export function FileUploadButton({ onFileSelect, className }: FileUploadButtonProps) {
  const [isUploading, setIsUploading] = useState(false);
  const { toast } = useToast();
  
  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    
    setIsUploading(true);
    
    try {
      // Convert FileList to array for easier handling
      const fileArray = Array.from(files);
      
      // Call the callback if provided
      if (onFileSelect) {
        onFileSelect(fileArray);
      }
      
      // Optional: Upload files to server
      // const uploadPromises = fileArray.map(file => uploadFile(file));
      // await Promise.all(uploadPromises);
      
      toast({
        title: "Files ready",
        description: `${fileArray.length} file(s) selected`,
      });
    } catch (error) {
      console.error('File upload error:', error);
      toast({
        variant: "destructive",
        title: "Upload failed",
        description: "There was a problem with your file upload.",
      });
    } finally {
      setIsUploading(false);
    }
  };
  
  return (
    <div className={className}>
      <Button 
        variant="secondary" 
        size="sm"
        disabled={isUploading}
        onClick={() => document.getElementById('file-upload')?.click()}
        className="w-full"
      >
        <Upload className="h-4 w-4 mr-2" />
        {isUploading ? 'Uploading...' : 'Upload Files'}
      </Button>
      <input
        id="file-upload"
        type="file"
        multiple
        onChange={handleFileChange}
        className="hidden"
        accept=".txt,.pdf,.md,.doc,.docx"
      />
    </div>
  );
}
