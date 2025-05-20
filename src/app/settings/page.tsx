"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { useToast } from "@/hooks/use-toast";
import { ArrowLeft, RefreshCw, Save } from "lucide-react";
import { loadUploadSettings, saveUploadSettings } from "@/lib/settings-api";

export default function SettingsPage() {
  const router = useRouter();
  const { toast } = useToast();
  const [activeTab, setActiveTab] = useState("model");
  const [loaded, setLoaded] = useState(false);
  
  // === LLM Settings ===
  const [modelSettings, setModelSettings] = useState({
    ollama_model: "llama3",
    ollama_embedding_model: "mxbai-embed-large",
    temperature: 0.7,
    top_p: 1.0,
    top_k_per_file: 5,
    similarity_threshold: 0.5,
    system_prompt: ""
  });
  
  // === Response Formatting ===
  const [formatSettings, setFormatSettings] = useState({
    format: "html",
    clean_response: true,
    remove_hedging: true,
    remove_references: true,
    remove_disclaimers: true,
    ensure_html_structure: true,
    custom_hedging_patterns: [] as string[],
    custom_reference_patterns: [] as string[],
    custom_disclaimer_patterns: [] as string[],
    html_tags_to_fix: [] as string[]
  });
  
  // === No Information Settings ===
  const [noInfoSettings, setNoInfoSettings] = useState({
    no_info_title: "Information Not Found",
    no_info_message: "I couldn't find specific information about that in the selected document(s).",
    include_suggestions: true,
    custom_suggestions: [] as string[],
    no_info_html_format: true
  });
  
  // === File Upload Settings ===
  const [uploadSettings, setUploadSettings] = useState({
    processOnUpload: true,
    autoAddToChat: true,
    maxFileSize: 10,
    allowedFileTypes: [] as string[]
  });
  
  // Custom pattern inputs
  const [newHedgingPattern, setNewHedgingPattern] = useState("");
  const [newReferencePattern, setNewReferencePattern] = useState("");
  const [newDisclaimerPattern, setNewDisclaimerPattern] = useState("");
  const [newHtmlTag, setNewHtmlTag] = useState("");
  const [newSuggestion, setNewSuggestion] = useState("");
  const [newFileType, setNewFileType] = useState("");
  
  // Load settings from localStorage on component mount
  useEffect(() => {
    const savedModelSettings = localStorage.getItem("chatbot_model_settings");
    const savedFormatSettings = localStorage.getItem("chatbot_format_settings");
    const savedNoInfoSettings = localStorage.getItem("chatbot_noinfo_settings");
    
    if (savedModelSettings) {
      setModelSettings(JSON.parse(savedModelSettings));
    }
    
    if (savedFormatSettings) {
      setFormatSettings(JSON.parse(savedFormatSettings));
    }
    
    if (savedNoInfoSettings) {
      setNoInfoSettings(JSON.parse(savedNoInfoSettings));
    }
    
    // Also load upload settings
    setUploadSettings(loadUploadSettings());
    
    setLoaded(true);
  }, []);
  
  // Save all settings to localStorage
  const saveSettings = () => {
    localStorage.setItem("chatbot_model_settings", JSON.stringify(modelSettings));
    localStorage.setItem("chatbot_format_settings", JSON.stringify(formatSettings));
    localStorage.setItem("chatbot_noinfo_settings", JSON.stringify(noInfoSettings));
    
    // Also save upload settings
    saveUploadSettings(uploadSettings);
    
    toast({
      title: "Settings saved",
      description: "Your settings have been saved and will be applied to future chats.",
      duration: 3000,
    });
  };
  
  // Reset settings to defaults
  const resetSettings = () => {
    if (confirm("Are you sure you want to reset all settings to defaults?")) {
      setModelSettings({
        ollama_model: "llama3",
        ollama_embedding_model: "mxbai-embed-large",
        temperature: 0.7,
        top_p: 1.0,
        top_k_per_file: 5,
        similarity_threshold: 0.5,
        system_prompt: ""
      });
      
      setFormatSettings({
        format: "html",
        clean_response: true,
        remove_hedging: true,
        remove_references: true,
        remove_disclaimers: true,
        ensure_html_structure: true,
        custom_hedging_patterns: [],
        custom_reference_patterns: [],
        custom_disclaimer_patterns: [],
        html_tags_to_fix: []
      });
      
      setNoInfoSettings({
        no_info_title: "Information Not Found",
        no_info_message: "I couldn't find specific information about that in the selected document(s).",
        include_suggestions: true,
        custom_suggestions: [],
        no_info_html_format: true
      });
      
      setUploadSettings({
        processOnUpload: true,
        autoAddToChat: true,
        maxFileSize: 10,
        allowedFileTypes: []
      });
      
      localStorage.removeItem("chatbot_model_settings");
      localStorage.removeItem("chatbot_format_settings");
      localStorage.removeItem("chatbot_noinfo_settings");
      
      toast({
        title: "Settings reset",
        description: "All settings have been reset to default values.",
        duration: 3000,
      });
    }
  };
  
  // Pattern management functions
// Pattern type definition
type PatternType = "hedging" | "reference" | "disclaimer" | "htmlTag" | "suggestion";

// Function to add patterns to their respective collections
const addPattern = (type: PatternType, value: string): void => {
    if (!value.trim()) return;
    
    switch (type) {
        case "hedging":
            setFormatSettings({
                ...formatSettings,
                custom_hedging_patterns: [...formatSettings.custom_hedging_patterns, value]
            });
            setNewHedgingPattern("");
            break;
        case "reference":
            setFormatSettings({
                ...formatSettings,
                custom_reference_patterns: [...formatSettings.custom_reference_patterns, value]
            });
            setNewReferencePattern("");
            break;
        case "disclaimer":
            setFormatSettings({
                ...formatSettings,
                custom_disclaimer_patterns: [...formatSettings.custom_disclaimer_patterns, value]
            });
            setNewDisclaimerPattern("");
            break;
        case "htmlTag":
            setFormatSettings({
                ...formatSettings,
                html_tags_to_fix: [...formatSettings.html_tags_to_fix, value]
            });
            setNewHtmlTag("");
            break;
        case "suggestion":
            setNoInfoSettings({
                ...noInfoSettings,
                custom_suggestions: [...noInfoSettings.custom_suggestions, value]
            });
            setNewSuggestion("");
            break;
    }
};
  
// Pattern type definition for removePattern function
const removePattern = (type: PatternType, index: number): void => {
    switch (type) {
        case "hedging":
            setFormatSettings({
                ...formatSettings,
                custom_hedging_patterns: formatSettings.custom_hedging_patterns.filter((_, i) => i !== index)
            });
            break;
        case "reference":
            setFormatSettings({
                ...formatSettings,
                custom_reference_patterns: formatSettings.custom_reference_patterns.filter((_, i) => i !== index)
            });
            break;
        case "disclaimer":
            setFormatSettings({
                ...formatSettings,
                custom_disclaimer_patterns: formatSettings.custom_disclaimer_patterns.filter((_, i) => i !== index)
            });
            break;
        case "htmlTag":
            setFormatSettings({
                ...formatSettings,
                html_tags_to_fix: formatSettings.html_tags_to_fix.filter((_, i) => i !== index)
            });
            break;
        case "suggestion":
            setNoInfoSettings({
                ...noInfoSettings,
                custom_suggestions: noInfoSettings.custom_suggestions.filter((_, i) => i !== index)
            });
            break;
    }
};

  if (!loaded) {
    return <div className="p-8 text-center">Loading settings...</div>;
  }
  
  return (
    <div className="container mx-auto py-6 max-w-5xl overflow-y-visible">
      <div className="flex items-center mb-6">
        <Button variant="ghost" onClick={() => router.back()} className="mr-4">
          <ArrowLeft className="h-4 w-4 mr-2" />
          Back
        </Button>
        <h1 className="text-2xl font-bold">Settings</h1>
      </div>
      
      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full ">
        <TabsList className="grid grid-cols-3 mb-6">
          <TabsTrigger value="model">Model & Retrieval</TabsTrigger>
          <TabsTrigger value="formatting">Response Formatting</TabsTrigger>
          <TabsTrigger value="noinfo">No-Information Responses</TabsTrigger>
          <TabsTrigger value="uploads">File Uploads</TabsTrigger>
        </TabsList>
        
        {/* === Model & Retrieval Settings === */}
        <TabsContent value="model">
          <Card>
            <CardHeader>
              <CardTitle>Model & Retrieval Settings</CardTitle>
              <CardDescription>
                Configure the LLM models and retrieval parameters used by the chatbot.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-4">
                <div>
                  <Label htmlFor="ollama_model">LLM Model</Label>
                  <Input 
                    id="ollama_model" 
                    value={modelSettings.ollama_model}
                    onChange={(e) => setModelSettings({...modelSettings, ollama_model: e.target.value})}
                    placeholder="llama3"
                  />
                  <p className="text-sm text-muted-foreground mt-1">
                    The Ollama model to use for chat (e.g., llama3, mistral, gemma)
                  </p>
                </div>
                
                <div>
                  <Label htmlFor="ollama_embedding_model">Embedding Model</Label>
                  <Input 
                    id="ollama_embedding_model" 
                    value={modelSettings.ollama_embedding_model}
                    onChange={(e) => setModelSettings({...modelSettings, ollama_embedding_model: e.target.value})}
                    placeholder="mxbai-embed-large"
                  />
                  <p className="text-sm text-muted-foreground mt-1">
                    The model to use for semantic embeddings
                  </p>
                </div>
                
                <div>
                  <Label htmlFor="temperature">Temperature: {modelSettings.temperature.toFixed(2)}</Label>
                  <Slider 
                    id="temperature"
                    min={0}
                    max={2}
                    step={0.01}
                    value={[modelSettings.temperature]}
                    onValueChange={(values) => setModelSettings({...modelSettings, temperature: values[0]})}
                    className="mt-2"
                  />
                  <p className="text-sm text-muted-foreground mt-1">
                    Controls randomness: lower values are more deterministic, higher values more creative
                  </p>
                </div>
                
                <div>
                  <Label htmlFor="top_p">Top P: {modelSettings.top_p.toFixed(2)}</Label>
                  <Slider 
                    id="top_p"
                    min={0.1}
                    max={1}
                    step={0.01}
                    value={[modelSettings.top_p]}
                    onValueChange={(values) => setModelSettings({...modelSettings, top_p: values[0]})}
                    className="mt-2"
                  />
                  <p className="text-sm text-muted-foreground mt-1">
                    Controls diversity via nucleus sampling
                  </p>
                </div>
                
                <div>
                  <Label htmlFor="top_k_per_file">Top K Results Per File: {modelSettings.top_k_per_file}</Label>
                  <Slider 
                    id="top_k_per_file"
                    min={1}
                    max={15}
                    step={1}
                    value={[modelSettings.top_k_per_file]}
                    onValueChange={(values) => setModelSettings({...modelSettings, top_k_per_file: values[0]})}
                    className="mt-2"
                  />
                  <p className="text-sm text-muted-foreground mt-1">
                    Maximum number of relevant chunks to retrieve per file
                  </p>
                </div>
                
                <div>
                  <Label htmlFor="similarity_threshold">Similarity Threshold: {modelSettings.similarity_threshold.toFixed(2)}</Label>
                  <Slider 
                    id="similarity_threshold"
                    min={0.1}
                    max={0.9}
                    step={0.01}
                    value={[modelSettings.similarity_threshold]}
                    onValueChange={(values) => setModelSettings({...modelSettings, similarity_threshold: values[0]})}
                    className="mt-2"
                  />
                  <p className="text-sm text-muted-foreground mt-1">
                    Minimum similarity score for a document chunk to be considered relevant
                  </p>
                </div>
                
                <div>
                  <Label htmlFor="system_prompt">System Prompt</Label>
                  <Textarea 
                    id="system_prompt" 
                    value={modelSettings.system_prompt}
                    onChange={(e) => setModelSettings({...modelSettings, system_prompt: e.target.value})}
                    placeholder="You are a helpful AI assistant..."
                    rows={5}
                  />
                  <p className="text-sm text-muted-foreground mt-1">
                    Custom system prompt to control assistant behavior
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        
        {/* === Response Formatting Settings === */}
        <TabsContent value="formatting">
          <Card>
            <CardHeader>
              <CardTitle>Response Formatting</CardTitle>
              <CardDescription>
                Configure how responses are formatted and cleaned
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label htmlFor="format">Response Format</Label>
                    <Select 
                      value={formatSettings.format}
                      onValueChange={(value) => setFormatSettings({...formatSettings, format: value})}
                    >
                      <SelectTrigger className="w-[180px]">
                        <SelectValue placeholder="Select format" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="html">HTML</SelectItem>
                        <SelectItem value="markdown">Markdown</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
                
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label htmlFor="clean_response">Clean Response</Label>
                    <p className="text-sm text-muted-foreground">
                      Apply language cleaning to AI responses
                    </p>
                  </div>
                  <Switch 
                    id="clean_response"
                    checked={formatSettings.clean_response}
                    onCheckedChange={(checked) => setFormatSettings({...formatSettings, clean_response: checked})}
                  />
                </div>
                
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label htmlFor="remove_hedging">Remove Hedging Language</Label>
                    <p className="text-sm text-muted-foreground">
                      Remove phrases like "I think", "perhaps", etc.
                    </p>
                  </div>
                  <Switch 
                    id="remove_hedging"
                    checked={formatSettings.remove_hedging}
                    onCheckedChange={(checked) => setFormatSettings({...formatSettings, remove_hedging: checked})}
                  />
                </div>
                
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label htmlFor="remove_references">Remove References</Label>
                    <p className="text-sm text-muted-foreground">
                      Remove "according to the document" and similar phrases
                    </p>
                  </div>
                  <Switch 
                    id="remove_references"
                    checked={formatSettings.remove_references}
                    onCheckedChange={(checked) => setFormatSettings({...formatSettings, remove_references: checked})}
                  />
                </div>
                
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label htmlFor="remove_disclaimers">Remove Disclaimers</Label>
                    <p className="text-sm text-muted-foreground">
                      Remove "I'm just an AI" and similar disclaimers
                    </p>
                  </div>
                  <Switch 
                    id="remove_disclaimers"
                    checked={formatSettings.remove_disclaimers}
                    onCheckedChange={(checked) => setFormatSettings({...formatSettings, remove_disclaimers: checked})}
                  />
                </div>
                
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label htmlFor="ensure_html_structure">Ensure HTML Structure</Label>
                    <p className="text-sm text-muted-foreground">
                      Fix unclosed HTML tags in response
                    </p>
                  </div>
                  <Switch 
                    id="ensure_html_structure"
                    checked={formatSettings.ensure_html_structure}
                    onCheckedChange={(checked) => setFormatSettings({...formatSettings, ensure_html_structure: checked})}
                  />
                </div>
                
                {/* Custom Patterns */}
                <div className="border rounded-md p-4 space-y-4">
                  <h3 className="font-medium">Custom Patterns to Remove</h3>
                  
                  <div className="space-y-2">
                    <Label htmlFor="custom_hedging">Custom Hedging Patterns</Label>
                    <div className="flex space-x-2">
                      <Input 
                        id="custom_hedging"
                        value={newHedgingPattern}
                        onChange={(e) => setNewHedgingPattern(e.target.value)}
                        placeholder="e.g., I'm not sure, but"
                      />
                      <Button 
                        onClick={() => addPattern("hedging", newHedgingPattern)}
                        disabled={!newHedgingPattern.trim()}
                      >
                        Add
                      </Button>
                    </div>
                    <div className="flex flex-wrap gap-2 mt-2">
                      {formatSettings.custom_hedging_patterns.map((pattern, index) => (
                        <Badge key={index} variant="secondary" className="gap-1">
                          {pattern}
                          <button 
                            className="ml-1 text-xs font-semibold" 
                            onClick={() => removePattern("hedging", index)}
                          >
                            ×
                          </button>
                        </Badge>
                      ))}
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    <Label htmlFor="custom_reference">Custom Reference Patterns</Label>
                    <div className="flex space-x-2">
                      <Input 
                        id="custom_reference"
                        value={newReferencePattern}
                        onChange={(e) => setNewReferencePattern(e.target.value)}
                        placeholder="e.g., The text suggests that"
                      />
                      <Button 
                        onClick={() => addPattern("reference", newReferencePattern)}
                        disabled={!newReferencePattern.trim()}
                      >
                        Add
                      </Button>
                    </div>
                    <div className="flex flex-wrap gap-2 mt-2">
                      {formatSettings.custom_reference_patterns.map((pattern, index) => (
                        <Badge key={index} variant="secondary" className="gap-1">
                          {pattern}
                          <button 
                            className="ml-1 text-xs font-semibold" 
                            onClick={() => removePattern("reference", index)}
                          >
                            ×
                          </button>
                        </Badge>
                      ))}
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    <Label htmlFor="custom_disclaimer">Custom Disclaimer Patterns</Label>
                    <div className="flex space-x-2">
                      <Input 
                        id="custom_disclaimer"
                        value={newDisclaimerPattern}
                        onChange={(e) => setNewDisclaimerPattern(e.target.value)}
                        placeholder="e.g., For professional advice, consult"
                      />
                      <Button 
                        onClick={() => addPattern("disclaimer", newDisclaimerPattern)}
                        disabled={!newDisclaimerPattern.trim()}
                      >
                        Add
                      </Button>
                    </div>
                    <div className="flex flex-wrap gap-2 mt-2">
                      {formatSettings.custom_disclaimer_patterns.map((pattern, index) => (
                        <Badge key={index} variant="secondary" className="gap-1">
                          {pattern}
                          <button 
                            className="ml-1 text-xs font-semibold" 
                            onClick={() => removePattern("disclaimer", index)}
                          >
                            ×
                          </button>
                        </Badge>
                      ))}
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    <Label htmlFor="html_tags">HTML Tags to Fix</Label>
                    <div className="flex space-x-2">
                      <Input 
                        id="html_tags"
                        value={newHtmlTag}
                        onChange={(e) => setNewHtmlTag(e.target.value)}
                        placeholder="e.g., p, div, ul"
                      />
                      <Button 
                        onClick={() => addPattern("htmlTag", newHtmlTag)}
                        disabled={!newHtmlTag.trim()}
                      >
                        Add
                      </Button>
                    </div>
                    <div className="flex flex-wrap gap-2 mt-2">
                      {formatSettings.html_tags_to_fix.map((tag, index) => (
                        <Badge key={index} variant="secondary" className="gap-1">
                          {tag}
                          <button 
                            className="ml-1 text-xs font-semibold" 
                            onClick={() => removePattern("htmlTag", index)}
                          >
                            ×
                          </button>
                        </Badge>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        
        {/* === No Information Response === */}
        <TabsContent value="noinfo">
          <Card>
            <CardHeader>
              <CardTitle>No Information Response</CardTitle>
              <CardDescription>
                Configure how the system responds when no relevant information is found
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-4">
                <div>
                  <Label htmlFor="no_info_title">Response Title</Label>
                  <Input 
                    id="no_info_title" 
                    value={noInfoSettings.no_info_title}
                    onChange={(e) => setNoInfoSettings({...noInfoSettings, no_info_title: e.target.value})}
                    placeholder="Information Not Found"
                  />
                  <p className="text-sm text-muted-foreground mt-1">
                    Title for the response when no information is found
                  </p>
                </div>
                
                <div>
                  <Label htmlFor="no_info_message">Response Message</Label>
                  <Textarea 
                    id="no_info_message" 
                    value={noInfoSettings.no_info_message}
                    onChange={(e) => setNoInfoSettings({...noInfoSettings, no_info_message: e.target.value})}
                    placeholder="I couldn't find specific information about that in the selected document(s)."
                    rows={3}
                  />
                  <p className="text-sm text-muted-foreground mt-1">
                    Message displayed when no relevant information is found
                  </p>
                </div>
                
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label htmlFor="include_suggestions">Include Suggestions</Label>
                    <p className="text-sm text-muted-foreground">
                      Show helpful suggestions when no information is found
                    </p>
                  </div>
                  <Switch 
                    id="include_suggestions"
                    checked={noInfoSettings.include_suggestions}
                    onCheckedChange={(checked) => setNoInfoSettings({...noInfoSettings, include_suggestions: checked})}
                  />
                </div>
                
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label htmlFor="no_info_html_format">Use HTML Format</Label>
                    <p className="text-sm text-muted-foreground">
                      Format no-information responses as HTML
                    </p>
                  </div>
                  <Switch 
                    id="no_info_html_format"
                    checked={noInfoSettings.no_info_html_format}
                    onCheckedChange={(checked) => setNoInfoSettings({...noInfoSettings, no_info_html_format: checked})}
                  />
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="custom_suggestions">Custom Suggestions</Label>
                  <div className="flex space-x-2">
                    <Input 
                      id="custom_suggestions"
                      value={newSuggestion}
                      onChange={(e) => setNewSuggestion(e.target.value)}
                      placeholder="e.g., Try rephrasing your question"
                    />
                    <Button 
                      onClick={() => addPattern("suggestion", newSuggestion)}
                      disabled={!newSuggestion.trim()}
                    >
                      Add
                    </Button>
                  </div>
                  <div className="flex flex-wrap gap-2 mt-2">
                    {noInfoSettings.custom_suggestions.map((suggestion, index) => (
                      <Badge key={index} variant="secondary" className="gap-1">
                        {suggestion}
                        <button 
                          className="ml-1 text-xs font-semibold" 
                          onClick={() => removePattern("suggestion", index)}
                        >
                          ×
                        </button>
                      </Badge>
                    ))}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        
        {/* === File Upload Settings === */}
        <TabsContent value="uploads">
          <Card>
            <CardHeader>
              <CardTitle>File Upload Settings</CardTitle>
              <CardDescription>
                Configure how files are uploaded and processed in chat
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label htmlFor="process_on_upload">Process on Upload</Label>
                    <p className="text-sm text-muted-foreground">
                      Automatically process files when uploaded
                    </p>
                  </div>
                  <Switch 
                    id="process_on_upload"
                    checked={uploadSettings.processOnUpload}
                    onCheckedChange={(checked) => setUploadSettings({...uploadSettings, processOnUpload: checked})}
                  />
                </div>
                
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label htmlFor="auto_add_to_chat">Auto-Add to Chat</Label>
                    <p className="text-sm text-muted-foreground">
                      Automatically add processed files to the current chat
                    </p>
                  </div>
                  <Switch 
                    id="auto_add_to_chat"
                    checked={uploadSettings.autoAddToChat}
                    onCheckedChange={(checked) => setUploadSettings({...uploadSettings, autoAddToChat: checked})}
                  />
                </div>
                
                <div>
                  <Label htmlFor="max_file_size">Maximum File Size (MB): {uploadSettings.maxFileSize}</Label>
                  <Slider 
                    id="max_file_size"
                    min={1}
                    max={50}
                    step={1}
                    value={[uploadSettings.maxFileSize]}
                    onValueChange={(values) => setUploadSettings({...uploadSettings, maxFileSize: values[0]})}
                    className="mt-2"
                  />
                  <p className="text-sm text-muted-foreground mt-1">
                    Maximum allowed file size for uploads
                  </p>
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="allowed_file_types">Allowed File Types</Label>
                  <div className="flex space-x-2">
                    <Input 
                      id="allowed_file_types"
                      value={newFileType}
                      onChange={(e) => setNewFileType(e.target.value)}
                      placeholder="e.g., .pdf, .txt"
                    />
                    <Button 
                      onClick={() => {
                        if (newFileType.trim()) {
                          setUploadSettings({
                            ...uploadSettings, 
                            allowedFileTypes: [...uploadSettings.allowedFileTypes, newFileType]
                          });
                          setNewFileType("");
                        }
                      }}
                      disabled={!newFileType.trim()}
                    >
                      Add
                    </Button>
                  </div>
                  <div className="flex flex-wrap gap-2 mt-2">
                    {uploadSettings.allowedFileTypes.map((type, index) => (
                      <Badge key={index} variant="secondary" className="gap-1">
                        {type}
                        <button 
                          className="ml-1 text-xs font-semibold" 
                          onClick={() => {
                            setUploadSettings({
                              ...uploadSettings,
                              allowedFileTypes: uploadSettings.allowedFileTypes.filter((_, i) => i !== index)
                            });
                          }}
                        >
                          ×
                        </button>
                      </Badge>
                    ))}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
      
      <div className="flex justify-between mt-6">
        <Button variant="outline" onClick={resetSettings} className="flex items-center">
          <RefreshCw className="h-4 w-4 mr-2" />
          Reset to Defaults
        </Button>
        <Button onClick={saveSettings} className="flex items-center">
          <Save className="h-4 w-4 mr-2" />
          Save Settings
        </Button>
      </div>
    </div>
  );
}