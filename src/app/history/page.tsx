
"use client";
import { useState, useEffect, FormEvent, ChangeEvent } from 'react';
import { useRouter } from 'next/navigation';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { ScrollArea } from "@/components/ui/scroll-area";
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle, AlertDialogTrigger } from "@/components/ui/alert-dialog";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { History, Search, Trash2, Edit3, RefreshCcw, MessageSquarePlus, Eye, SortAsc, SortDesc } from "lucide-react";
import LoadingSpinner from "@/components/ui/loading-spinner";
import type { ChatSession } from "@/types";
import { useToast } from "@/hooks/use-toast";
import { formatDistanceToNow } from 'date-fns';

const CHAT_HISTORY_KEY = "nebulaChatHistory";

const loadChatHistoryFromLocalStorage = (): ChatSession[] => {
  try {
    if (typeof window === 'undefined') return [];
    const historyJson = localStorage.getItem(CHAT_HISTORY_KEY);
    return historyJson ? JSON.parse(historyJson) : [];
  } catch (error) {
    console.error("Error loading chat history:", error);
    return [];
  }
};

const saveChatHistoryToLocalStorage = (history: ChatSession[]) => {
  try {
    if (typeof window === 'undefined') return;
    localStorage.setItem(CHAT_HISTORY_KEY, JSON.stringify(history));
  } catch (error) {
    console.error("Error saving chat history:", error);
  }
};


export default function ChatHistoryPage() {
  const [chatHistory, setChatHistory] = useState<ChatSession[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState("");
  const [editingSession, setEditingSession] = useState<ChatSession | null>(null);
  const [newName, setNewName] = useState("");
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc'); // 'desc' for newest first
  const router = useRouter();
  const { toast } = useToast();

  useEffect(() => {
    setChatHistory(loadChatHistoryFromLocalStorage());
    setIsLoading(false);
  }, []);

  const handleSearch = (event: ChangeEvent<HTMLInputElement>) => {
    setSearchTerm(event.target.value);
  };

  const handleDelete = (sessionId: string) => {
    const updatedHistory = chatHistory.filter(session => session.id !== sessionId);
    setChatHistory(updatedHistory);
    saveChatHistoryToLocalStorage(updatedHistory);
    toast({ title: "Chat Deleted", description: "The chat session has been removed." });
  };

  const handleRename = (e: FormEvent) => {
    e.preventDefault();
    if (editingSession && newName.trim()) {
      const updatedHistory = chatHistory.map(session =>
        session.id === editingSession.id ? { ...session, name: newName.trim(), lastModified: Date.now() } : session
      );
      setChatHistory(updatedHistory);
      saveChatHistoryToLocalStorage(updatedHistory);
      toast({ title: "Chat Renamed", description: `Session renamed to "${newName.trim()}".` });
      setEditingSession(null);
      setNewName("");
    }
  };

  const openRenameDialog = (session: ChatSession) => {
    setEditingSession(session);
    setNewName(session.name);
  };

  const handleReloadChat = (sessionId: string) => {
    // In a real app, this would navigate to the chat page with the session ID
    // For this example, it's more of a placeholder as we don't have session loading by ID implemented in chat page.
    // router.push(`/?sessionId=${sessionId}`);
    toast({ title: "Reload Chat", description: `Navigating to chat session (not fully implemented in demo). ID: ${sessionId}` });
    // To truly reload, the chat page needs to accept a session ID and load it.
    // For now, let's copy ID to clipboard or similar action.
    navigator.clipboard.writeText(sessionId).then(() => {
       toast({ title: "Session ID Copied", description: "Session ID copied to clipboard. You can use this to manually restore a session if needed (feature TBD)." });
    }).catch(err => {
       toast({ title: "Copy Failed", description: "Could not copy Session ID.", variant: "destructive" });
    });
  };
  
  const handleNewChat = () => {
    router.push('/'); // Navigate to the main chat page to start a new chat
  };

  const toggleSortOrder = () => {
    setSortOrder(prev => prev === 'asc' ? 'desc' : 'asc');
  };

  const filteredAndSortedHistory = chatHistory
    .filter(session =>
      session.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      session.messages.some(msg => msg.content.toLowerCase().includes(searchTerm.toLowerCase()))
    )
    .sort((a, b) => {
      if (sortOrder === 'asc') {
        return a.lastModified - b.lastModified;
      }
      return b.lastModified - a.lastModified;
    });

  if (isLoading) {
    return (
      <div className="flex justify-center items-center h-full">
        <LoadingSpinner size="lg" />
        <p className="ml-2">Loading chat history...</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <Card className="glassmorphic animated-smoke-bg">
        <CardHeader>
          <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2">
            <div>
              <CardTitle className="text-xl flex items-center"><History className="mr-2 h-6 w-6" />Chat History</CardTitle>
              <CardDescription>
                Review, manage, and reload your past conversations.
              </CardDescription>
            </div>
            <Button onClick={handleNewChat}><MessageSquarePlus className="mr-2 h-4 w-4" /> New Chat</Button>
          </div>
          <div className="mt-4 flex flex-col sm:flex-row gap-2 items-center">
            <div className="relative flex-grow w-full sm:w-auto">
              <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                type="search"
                placeholder="Search chats by name or content..."
                value={searchTerm}
                onChange={handleSearch}
                className="pl-8 w-full"
              />
            </div>
            <Button variant="outline" onClick={toggleSortOrder} className="w-full sm:w-auto">
              {sortOrder === 'asc' ? <SortAsc className="mr-2 h-4 w-4" /> : <SortDesc className="mr-2 h-4 w-4" />}
              Sort ({sortOrder === 'asc' ? 'Oldest' : 'Newest'} First)
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {filteredAndSortedHistory.length === 0 ? (
            <div className="text-center py-10">
              <History className="mx-auto h-12 w-12 text-muted-foreground" />
              <p className="mt-2 text-muted-foreground">
                {searchTerm ? "No chats match your search." : "No chat history yet."}
              </p>
              {!searchTerm && <Button onClick={handleNewChat} className="mt-4">Start a New Chat</Button>}
            </div>
          ) : (
            <ScrollArea className="h-[calc(100vh-25rem)] sm:h-[calc(100vh-22rem)]"> {/* Adjust height as needed */}
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Name</TableHead>
                    <TableHead>Last Modified</TableHead>
                    <TableHead className="text-right">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filteredAndSortedHistory.map((session) => (
                    <TableRow key={session.id}>
                      <TableCell className="font-medium">{session.name}</TableCell>
                      <TableCell>{formatDistanceToNow(new Date(session.lastModified), { addSuffix: true })}</TableCell>
                      <TableCell className="text-right space-x-1">
                        <Button variant="ghost" size="icon" onClick={() => handleReloadChat(session.id)} title="Reload Chat (copies ID)">
                          <Eye className="h-4 w-4" />
                        </Button>
                        <DialogTrigger asChild>
                          <Button variant="ghost" size="icon" onClick={() => openRenameDialog(session)} title="Rename Chat">
                            <Edit3 className="h-4 w-4" />
                          </Button>
                        </DialogTrigger>
                        <AlertDialog>
                          <AlertDialogTrigger asChild>
                            <Button variant="ghost" size="icon" className="text-destructive hover:text-destructive-foreground hover:bg-destructive" title="Delete Chat">
                              <Trash2 className="h-4 w-4" />
                            </Button>
                          </AlertDialogTrigger>
                          <AlertDialogContent className="glassmorphic">
                            <AlertDialogHeader>
                              <AlertDialogTitle>Are you sure?</AlertDialogTitle>
                              <AlertDialogDescription>
                                This action cannot be undone. This will permanently delete the chat session "{session.name}".
                              </AlertDialogDescription>
                            </AlertDialogHeader>
                            <AlertDialogFooter>
                              <AlertDialogCancel>Cancel</AlertDialogCancel>
                              <AlertDialogAction onClick={() => handleDelete(session.id)} className="bg-destructive hover:bg-destructive/90">
                                Delete
                              </AlertDialogAction>
                            </AlertDialogFooter>
                          </AlertDialogContent>
                        </AlertDialog>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </ScrollArea>
          )}
        </CardContent>
        {filteredAndSortedHistory.length > 0 && (
          <CardFooter className="text-sm text-muted-foreground">
            Showing {filteredAndSortedHistory.length} chat session(s).
          </CardFooter>
        )}
      </Card>

      {editingSession && (
        <Dialog open={!!editingSession} onOpenChange={(open) => !open && setEditingSession(null)}>
          <DialogContent className="glassmorphic">
            <DialogHeader>
              <DialogTitle>Rename Chat Session</DialogTitle>
              <DialogDescription>
                Enter a new name for "{editingSession.name}".
              </DialogDescription>
            </DialogHeader>
            <form onSubmit={handleRename} className="space-y-4 py-2">
              <Input
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                placeholder="New chat name"
                autoFocus
              />
              <DialogFooter>
                <Button type="button" variant="outline" onClick={() => setEditingSession(null)}>Cancel</Button>
                <Button type="submit">Save Name</Button>
              </DialogFooter>
            </form>
          </DialogContent>
        </Dialog>
      )}
    </div>
  );
}
