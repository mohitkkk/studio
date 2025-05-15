
"use client";

import { useState, useEffect, FormEvent, ChangeEvent } from 'react';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { ScrollArea } from "@/components/ui/scroll-area";
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle, AlertDialogTrigger } from "@/components/ui/alert-dialog";
import { Settings2, PlusCircle, Trash2, Save, RotateCcw } from "lucide-react";
import LoadingSpinner from "@/components/ui/loading-spinner";
import type { EnvVariable } from "@/types";
import { useToast } from "@/hooks/use-toast";
import { v4 as uuidv4 } from 'uuid';

const ENV_CONFIG_KEY = "nebulaChatEnvConfig";

const loadEnvConfigFromLocalStorage = (): EnvVariable[] => {
  try {
    if (typeof window === 'undefined') return [];
    const configJson = localStorage.getItem(ENV_CONFIG_KEY);
    return configJson ? JSON.parse(configJson) : [];
  } catch (error) {
    console.error("Error loading env config:", error);
    return [];
  }
};

const saveEnvConfigToLocalStorage = (config: EnvVariable[]) => {
  try {
    if (typeof window === 'undefined') return;
    localStorage.setItem(ENV_CONFIG_KEY, JSON.stringify(config));
  } catch (error) {
    console.error("Error saving env config:", error);
  }
};

export default function ConfigPage() {
  const [envVariables, setEnvVariables] = useState<EnvVariable[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [newVarKey, setNewVarKey] = useState("");
  const [newVarValue, setNewVarValue] = useState("");
  const { toast } = useToast();

  useEffect(() => {
    setEnvVariables(loadEnvConfigFromLocalStorage());
    setIsLoading(false);
  }, []);

  const handleAddVariable = (e: FormEvent) => {
    e.preventDefault();
    if (!newVarKey.trim() || !newVarValue.trim()) {
      toast({ title: "Missing Fields", description: "Both key and value are required.", variant: "destructive" });
      return;
    }
    if (envVariables.some(v => v.key === newVarKey.trim())) {
      toast({ title: "Duplicate Key", description: `Variable with key "${newVarKey.trim()}" already exists.`, variant: "destructive" });
      return;
    }

    const newVar: EnvVariable = {
      id: uuidv4(),
      key: newVarKey.trim(),
      value: newVarValue.trim(),
    };
    const updatedVars = [...envVariables, newVar];
    setEnvVariables(updatedVars);
    // Note: This saves to localStorage. Real backend config is not affected.
    saveEnvConfigToLocalStorage(updatedVars); 
    toast({ title: "Variable Added", description: `Variable "${newVar.key}" added.` });
    setNewVarKey("");
    setNewVarValue("");
  };

  const handleDeleteVariable = (varId: string) => {
    const updatedVars = envVariables.filter(v => v.id !== varId);
    setEnvVariables(updatedVars);
    saveEnvConfigToLocalStorage(updatedVars);
    toast({ title: "Variable Deleted", description: "The variable has been removed." });
  };
  
  const handleSaveConfiguration = () => {
    // In a real app, this would likely send to a backend. Here, it's already saved on change.
    // This button can serve as a user confirmation or trigger for other actions.
    saveEnvConfigToLocalStorage(envVariables); // Explicit save, though it happens on add/delete
    toast({ title: "Configuration Saved", description: "Current environment variables saved to local storage." });
  };

  const handleResetConfiguration = () => {
    setEnvVariables([]);
    saveEnvConfigToLocalStorage([]);
    toast({ title: "Configuration Reset", description: "All local environment variables have been cleared." });
  };
  
  const handleVariableChange = (id: string, field: 'key' | 'value', value: string) => {
    setEnvVariables(prevVars => 
      prevVars.map(v => v.id === id ? { ...v, [field]: value } : v)
    );
    // Note: For live editing, you might want to debounce saving or save on blur/submit
  };


  if (isLoading) {
    return (
      <div className="flex justify-center items-center h-full">
        <LoadingSpinner size="lg" />
        <p className="ml-2">Loading configuration...</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <Card className="glassmorphic animated-smoke-bg">
        <CardHeader>
          <CardTitle className="text-xl flex items-center"><Settings2 className="mr-2 h-6 w-6" />Backend Configuration</CardTitle>
          <CardDescription>
            Manage environment variables for the application. Changes are saved locally.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleAddVariable} className="flex flex-col sm:flex-row gap-2 mb-6 p-4 border rounded-md bg-secondary/30">
            <Input
              type="text"
              placeholder="Variable Key (e.g., API_KEY)"
              value={newVarKey}
              onChange={(e) => setNewVarKey(e.target.value)}
              className="flex-grow"
              required
            />
            <Input
              type="text"
              placeholder="Variable Value"
              value={newVarValue}
              onChange={(e) => setNewVarValue(e.target.value)}
              className="flex-grow"
              required
            />
            <Button type="submit" className="w-full sm:w-auto"><PlusCircle className="mr-2 h-4 w-4" /> Add Variable</Button>
          </form>

          {envVariables.length === 0 ? (
            <div className="text-center py-10">
              <Settings2 className="mx-auto h-12 w-12 text-muted-foreground" />
              <p className="mt-2 text-muted-foreground">No environment variables configured yet.</p>
            </div>
          ) : (
            <ScrollArea className="h-[calc(100vh-30rem)] sm:h-[calc(100vh-28rem)]"> {/* Adjust height */}
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Key</TableHead>
                    <TableHead>Value</TableHead>
                    <TableHead className="text-right">Action</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {envVariables.map((variable) => (
                    <TableRow key={variable.id}>
                      <TableCell>
                        <Input 
                          value={variable.key} 
                          onChange={(e) => handleVariableChange(variable.id, 'key', e.target.value)}
                          className="bg-transparent border-0 focus:ring-1 focus:ring-primary px-1"
                        />
                      </TableCell>
                      <TableCell>
                        <Input 
                          type="password"
                          value={variable.value} 
                          onChange={(e) => handleVariableChange(variable.id, 'value', e.target.value)}
                          className="bg-transparent border-0 focus:ring-1 focus:ring-primary px-1"
                        />
                      </TableCell>
                      <TableCell className="text-right">
                        <Button variant="ghost" size="icon" onClick={() => handleDeleteVariable(variable.id)} className="text-destructive hover:text-destructive-foreground hover:bg-destructive" title="Delete Variable">
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </ScrollArea>
          )}
        </CardContent>
        <CardFooter className="flex flex-col sm:flex-row justify-end gap-2 border-t pt-4">
          <AlertDialog>
            <AlertDialogTrigger asChild>
              <Button variant="outline" className="w-full sm:w-auto"><RotateCcw className="mr-2 h-4 w-4" /> Reset All</Button>
            </AlertDialogTrigger>
            <AlertDialogContent className="glassmorphic">
              <AlertDialogHeader>
                <AlertDialogTitle>Are you sure you want to reset?</AlertDialogTitle>
                <AlertDialogDescription>
                  This will delete all locally stored environment variables. This action cannot be undone.
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>Cancel</AlertDialogCancel>
                <AlertDialogAction onClick={handleResetConfiguration} className="bg-destructive hover:bg-destructive/90">
                  Reset Configuration
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
          <Button onClick={handleSaveConfiguration} className="w-full sm:w-auto"><Save className="mr-2 h-4 w-4" /> Save Configuration</Button>
        </CardFooter>
      </Card>
    </div>
  