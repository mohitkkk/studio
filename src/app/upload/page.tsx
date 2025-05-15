
"use client";

import { useState, useRef, ChangeEvent } from 'react';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from '@/components/ui/scroll-area';
import { UploadCloud, FileText, Trash2 } from "lucide-react";
import LoadingSpinner from "@/components/ui/loading-spinner";
import { summarizeDocument, SummarizeDocumentInput } from "@/ai/flows/summarize-document";
import { useToast } from "@/hooks/use-toast";

export default function FileUploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [summary, setSummary] = useState<string | null>(null);
  const [filePreview, setFilePreview] = useState<string | null>(null); // For text files
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { toast } = useToast();

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setSummary(null); // Clear previous summary
      setFilePreview(null); // Clear previous preview

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
      toast({ title: "No file selected", description: "Please select a file to summarize.", variant: "destructive" });
      return;
    }

    setIsProcessing(true);
    setSummary(null);

    try {
      const reader = new FileReader();
      reader.onload = async (e) => {
        const fileDataUri = e.target?.result as string;
        const input: SummarizeDocumentInput = { documentDataUri: fileDataUri };
        const output = await summarizeDocument(input);
        setSummary(output.summary);
        toast({ title: "Summarization Complete", description: `Summary generated for ${file.name}.` });
      };
      reader.readAsDataURL(file);
    } catch (error) {
      console.error("Error summarizing document:", error);
      setSummary("Failed to generate summary.");
      toast({ title: "Summarization Error", description: "An error occurred while summarizing the document.", variant: "destructive" });
    } finally {
      // FileReader is async, so setIsProcessing might be better inside onload's finally, but for simplicity:
      // We'll set it to false once the process starts, UI will update with summary when ready.
      // Better: await the promise from reader.onload if it returned one.
      // For now, rely on summary state to show result.
      // A more robust way:
      const processFile = new Promise<void>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = async (e) => {
          try {
            const fileDataUri = e.target?.result as string;
            const input: SummarizeDocumentInput = { documentDataUri: fileDataUri };
            const output = await summarizeDocument(input);
            setSummary(output.summary);
            toast({ title: "Summarization Complete", description: `Summary generated for ${file.name}.` });
            resolve();
          } catch (err) {
            reject(err);
          }
        };
        reader.onerror = reject;
        reader.readAsDataURL(file);
      });

      try {
        await processFile;
      } catch (error) {
        console.error("Error processing or summarizing document:", error);
        setSummary("Failed to generate summary.");
        toast({ title: "Processing Error", description: "An error occurred.", variant: "destructive" });
      } finally {
        setIsProcessing(false);
      }
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

    