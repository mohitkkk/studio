"use client";

import { useState, useRef, ChangeEvent } from 'react';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from '@/components/ui/scroll-area';
import { UploadCloud, FileText, Trash2 } from "lucide-react";
import LoadingSpinner from "@/components/ui/loading-spinner";
import axios from 'axios';

// Get API URL from environment variables with fallback
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://13.202.208.115:3000';

export default function FileUploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [summary, setSummary] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [filePreview, setFilePreview] = useState<string | null>(null); // For text files
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setSummary(null); // Clear previous summary
      setFilePreview(null); // Clear previous preview
      setError(null); // Clear previous error

      // Preview for text files
      if (selectedFile.type === "text/plain") {
        const reader = new FileReader();
        reader.onload = (e) => setFilePreview(e.target?.result as string);
        reader.readAsText(selectedFile);
      }
    }
  };

  const handleSummarize = async () => {
    if (!file) {
      setError('Please select a file first');
      return;
    }

    try {
      setIsProcessing(true);
      setError(null);
      const fileToUpload = file;
      
      // Use FormData for file upload
      const formData = new FormData();
      formData.append('file', fileToUpload);
      
      // Directly use axios to upload to our API server - no server action
      console.log(`Uploading file to ${API_URL}/upload`);
      const uploadResponse = await axios.post(`${API_URL}/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      if (!uploadResponse.data || !uploadResponse.data.filename) {
        throw new Error('File upload failed - no filename returned');
      }
      
      const fileId = uploadResponse.data.filename;
      
      // Now process the document directly with our API
      const processResponse = await axios.post(`${API_URL}/process`, {
        filename: fileId
      });
      
      if (processResponse.data && processResponse.data.summary) {
        setSummary(processResponse.data.summary);
      } else {
        throw new Error('No summary returned from processing service');
      }
    } catch (err) {
      console.error('Error processing or summarizing document:', err);
      setError(`Error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setIsProcessing(false);
    }
  };

  const triggerFileDialog = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className="space-y-6">
      <Card className="glassmorphic animated-smoke-bg">
        <CardHeader>
          <CardTitle className="text-xl">Upload and Summarize Document</CardTitle>
          <CardDescription>
            Upload a document (TXT, PDF, DOCX) and get a concise summary powered by AI.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div
            className="flex flex-col items-center justify-center p-6 border-2 border-dashed border-border rounded-lg cursor-pointer hover:border-primary transition-colors"
            onClick={triggerFileDialog}
          >
            <UploadCloud className="w-12 h-12 text-muted-foreground" />
            <p className="mt-2 text-sm text-muted-foreground">
              {file ? file.name : "Click or drag file to this area to upload"}
            </p>
            <Input
              ref={fileInputRef}
              type="file"
              className="hidden"
              onChange={handleFileChange}
              accept=".txt,.pdf,.md,.docx"
            />
             {file && (
                <Button variant="ghost" size="sm" className="mt-2 text-destructive hover:text-destructive-foreground hover:bg-destructive" onClick={(e) => { e.stopPropagation(); setFile(null); setSummary(null); setFilePreview(null); if(fileInputRef.current) fileInputRef.current.value = '';}}>
                    <Trash2 className="w-4 h-4 mr-1" /> Clear selection
                </Button>
            )}
          </div>

          {file && (
            <div className="p-3 border rounded-md bg-secondary/30">
              <div className="flex items-center gap-2">
                <FileText className="w-5 h-5 text-primary" />
                <div>
                  <p className="font-medium">{file.name}</p>
                  <p className="text-xs text-muted-foreground">
                    {(file.size / 1024).toFixed(2)} KB - {file.type}
                  </p>
                </div>
              </div>
            </div>
          )}

          {filePreview && (
            <Card className="mt-4">
              <CardHeader><CardTitle className="text-base">File Preview (Plain Text)</CardTitle></CardHeader>
              <CardContent>
                <ScrollArea className="h-40 whitespace-pre-wrap text-xs p-2 border rounded-md bg-muted/20">
                  {filePreview}
                </ScrollArea>
              </CardContent>
            </Card>
          )}

        </CardContent>
        <CardFooter>
          <Button onClick={handleSummarize} disabled={!file || isProcessing} className="w-full">
            {isProcessing ? <LoadingSpinner className="mr-2" /> : <UploadCloud className="mr-2 h-4 w-4" />}
            Summarize Document
          </Button>
        </CardFooter>
      </Card>

      {isProcessing && !summary && (
        <Card className="glassmorphic animated-smoke-bg">
          <CardHeader><CardTitle className="text-base">Processing...</CardTitle></CardHeader>
          <CardContent className="flex justify-center py-8">
            <LoadingSpinner size="lg" />
          </CardContent>
        </Card>
      )}

      {summary && (
        <Card className="glassmorphic animated-smoke-bg">
          <CardHeader>
            <CardTitle className="text-lg">Summary</CardTitle>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-60 whitespace-pre-wrap text-sm p-3 border rounded-md bg-muted/20 leading-relaxed">
              {summary}
            </ScrollArea>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

