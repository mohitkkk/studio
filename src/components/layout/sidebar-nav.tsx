"use client";

import { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';
import { usePathname, useRouter } from 'next/navigation';
import { cn } from '@/lib/utils';
import { MessageSquareText, Home, Settings, PlusCircle, Upload, Trash2, Pencil } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { v4 as uuidv4 } from 'uuid';
import { 
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog';
import { Input } from '@/components/ui/input';

// List of navigation items (excluding New Chat which will be handled separately)
const navItems = [
	{
		title: 'Home',
		href: '/',
		icon: Home,
	},
	{
		title: 'Upload Files',
		href: '/upload',
		icon: Upload,
	},
	{
		title: 'Settings',
		href: '/settings',
		icon: Settings,
	},
];

// Constants for local storage keys
const CHAT_HISTORY_KEY = "nebulaChatHistory";
const CHAT_INDEX_KEY = "nebulaChatIndex";
const CHAT_COUNTER_KEY = "nebulaChatCounter";

// Interface for chat history items
interface ChatHistoryItem {
  id: string;
  name: string;
  lastModified: number;
  selectedFiles: string[];
}

export default function SidebarNav() {
	const pathname = usePathname();
	const router = useRouter();
	const [expanded, setExpanded] = useState(true);
	const [chats, setChats] = useState<ChatHistoryItem[]>([]);
	// Add state to prevent multiple clicks
	const [isCreatingChat, setIsCreatingChat] = useState(false);
	// Add state for delete confirmation dialog
	const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
	const [chatToDelete, setChatToDelete] = useState<string | null>(null);
	// Add state for rename dialog
	const [renameDialogOpen, setRenameDialogOpen] = useState(false);
	const [chatToRename, setChatToRename] = useState<string | null>(null);
	const [newChatName, setNewChatName] = useState("");

	// Load chats from local storage
	useEffect(() => {
		const loadChats = () => {
			if (typeof window === 'undefined') return;
			
			try {
				const indexJson = localStorage.getItem(CHAT_INDEX_KEY);
				const chatIndex = indexJson ? JSON.parse(indexJson) : [];
				// Sort by lastModified (newest first)
				chatIndex.sort((a: ChatHistoryItem, b: ChatHistoryItem) => b.lastModified - a.lastModified);
				setChats(chatIndex);
			} catch (error) {
				console.error("Error loading chat index:", error);
				setChats([]);
			}
		};
		
		loadChats();
		
		// Add event listener to update chats when storage changes
		window.addEventListener('storage', loadChats);
		
		// Create custom event for chat updates
		window.addEventListener('chatUpdated', loadChats);
		
		return () => {
			window.removeEventListener('storage', loadChats);
			window.removeEventListener('chatUpdated', loadChats);
		};
	}, []);

	// Helper function to get next chat number
	const getNextChatNumber = useCallback((): number => {
		try {
			if (typeof window === 'undefined') return 1;
			const counter = localStorage.getItem(CHAT_COUNTER_KEY);
			const nextNumber = counter ? parseInt(counter) + 1 : 1;
			localStorage.setItem(CHAT_COUNTER_KEY, nextNumber.toString());
			return nextNumber;
		} catch (error) {
			console.error("Error getting next chat number:", error);
			return 1;
		}
	}, []);

	// Create a new chat with safeguards to prevent multiple creation
	const createNewChat = useCallback(() => {
		try {
			// Prevent multiple rapid clicks
			if (isCreatingChat) return null;
			setIsCreatingChat(true);
			
			if (typeof window === 'undefined') {
				setIsCreatingChat(false);
				return null;
			}
			
			// Generate chat ID and name
			const chatId = `chat_${uuidv4()}`;
			const chatNumber = getNextChatNumber();
			const chatName = `Chat ${chatNumber}`;
			
			// Create welcome message
			const welcomeMessage = {
				id: uuidv4(),
				sender: "assistant",
				content: "<p>Hello! How can I help you today? You can ask me questions about any files you select.</p>",
				timestamp: Date.now()
			};
			
			// Create new session
			const newSession = {
				id: chatId,
				name: chatName,
				messages: [welcomeMessage],
				selectedFiles: [],
				lastModified: Date.now()
			};
			
			// Save session
			localStorage.setItem(`${CHAT_HISTORY_KEY}_${chatId}`, JSON.stringify(newSession));
			
			// Update chat index
			const chatIndex = getChatIndex();
			chatIndex.unshift({
				id: chatId,
				name: chatName,
				lastModified: Date.now(),
				selectedFiles: []
			});
			localStorage.setItem(CHAT_INDEX_KEY, JSON.stringify(chatIndex));
			
			// Create a custom event that includes the new chat ID
			const chatCreatedEvent = new CustomEvent('chatCreated', { 
				detail: { chatId: chatId } 
			});
			
			// Dispatch both events - chatUpdated for the list, chatCreated for the chat interface
			window.dispatchEvent(new Event('chatUpdated'));
			window.dispatchEvent(chatCreatedEvent);
			
			// Update current chat in localStorage so the main interface knows which chat to display
			localStorage.setItem('currentChatId', chatId);
			
			// Reset creating state after a short delay
			setTimeout(() => setIsCreatingChat(false), 100);
			
			return chatId;
		} catch (error) {
			console.error("Error creating new chat:", error);
			setIsCreatingChat(false);
			return null;
		}
	}, [getNextChatNumber, isCreatingChat]);

	// Helper function to get chat index
	const getChatIndex = (): ChatHistoryItem[] => {
		try {
			if (typeof window === 'undefined') return [];
			const indexJson = localStorage.getItem(CHAT_INDEX_KEY);
			return indexJson ? JSON.parse(indexJson) : [];
		} catch (error) {
			console.error("Error loading chat index:", error);
			return [];
		}
	};

	// Modified handleChatClick to not navigate, just update localStorage and dispatch event
	const handleChatClick = useCallback((e: React.MouseEvent<HTMLAnchorElement>, chatId: string) => {
		e.preventDefault();
		
		// Check if chat exists in local storage
		const chatData = localStorage.getItem(`${CHAT_HISTORY_KEY}_${chatId}`);
		if (!chatData) {
			console.warn(`Chat ${chatId} not found in local storage, creating new chat instead`);
			createNewChat();
			return;
		}
		
		// Update current chat in localStorage so the main interface knows which chat to display
		localStorage.setItem('currentChatId', chatId);
		
		// Create and dispatch a custom event to notify chat interface that the selected chat has changed
		const chatSelectedEvent = new CustomEvent('chatSelected', { 
			detail: { chatId: chatId } 
		});
		window.dispatchEvent(chatSelectedEvent);
	}, [createNewChat]);

	// New function to handle chat deletion
	const deleteChat = useCallback((chatId: string) => {
		try {
			if (typeof window === 'undefined') return;
			
			// Open confirmation dialog
			setChatToDelete(chatId);
			setDeleteDialogOpen(true);
		} catch (error) {
			console.error("Error preparing chat deletion:", error);
		}
	}, []);

	// Function to confirm and execute chat deletion
	const confirmDeleteChat = useCallback(() => {
		try {
			if (typeof window === 'undefined' || !chatToDelete) return;
			
			// Remove chat from localStorage
			localStorage.removeItem(`${CHAT_HISTORY_KEY}_${chatToDelete}`);
			
			// Update chat index
			const chatIndex = getChatIndex();
			const updatedIndex = chatIndex.filter(chat => chat.id !== chatToDelete);
			localStorage.setItem(CHAT_INDEX_KEY, JSON.stringify(updatedIndex));
			
			// Update state
			setChats(prevChats => prevChats.filter(chat => chat.id !== chatToDelete));
			
			// Check if current chat is the one being deleted
			if (pathname === `/chat/${chatToDelete}`) {
				// If we deleted the current chat, navigate to a different one or home
				const remainingChats = getChatIndex().filter(chat => chat.id !== chatToDelete);
				if (remainingChats.length > 0) {
					router.push(`/chat/${remainingChats[0].id}`);
				} else {
					router.push('/');
				}
			}
			
			// Notify about chat update
			window.dispatchEvent(new Event('chatUpdated'));
			
			// Close dialog
			setDeleteDialogOpen(false);
			setChatToDelete(null);
			
		} catch (error) {
			console.error("Error deleting chat:", error);
			setDeleteDialogOpen(false);
			setChatToDelete(null);
		}
	}, [chatToDelete, pathname, router]);

	// New function to handle chat renaming
	const renameChat = useCallback((chatId: string) => {
		try {
			if (typeof window === 'undefined') return;
			
			// Find chat to get current name
			const chat = chats.find(c => c.id === chatId);
			if (chat) {
				setChatToRename(chatId);
				setNewChatName(chat.name);
				setRenameDialogOpen(true);
			}
		} catch (error) {
			console.error("Error preparing chat rename:", error);
		}
	}, [chats]);

	// Function to confirm and execute chat rename
	const confirmRenameChat = useCallback(() => {
		try {
			if (typeof window === 'undefined' || !chatToRename || !newChatName.trim()) return;
			
			// Get the chat from localStorage
			const chatData = localStorage.getItem(`${CHAT_HISTORY_KEY}_${chatToRename}`);
			if (!chatData) {
				console.warn(`Chat ${chatToRename} not found for renaming.`);
				setRenameDialogOpen(false);
				setChatToRename(null);
				return;
			}
			
			// Update chat name
			const parsedChat = JSON.parse(chatData);
			parsedChat.name = newChatName.trim();
			
			// Save updated chat
			localStorage.setItem(`${CHAT_HISTORY_KEY}_${chatToRename}`, JSON.stringify(parsedChat));
			
			// Update chat index
			const chatIndex = getChatIndex();
			const chatToUpdate = chatIndex.find(chat => chat.id === chatToRename);
			if (chatToUpdate) {
				chatToUpdate.name = newChatName.trim();
				localStorage.setItem(CHAT_INDEX_KEY, JSON.stringify(chatIndex));
			}
			
			// Update state
			setChats(prevChats => 
				prevChats.map(chat => 
					chat.id === chatToRename 
						? { ...chat, name: newChatName.trim() } 
						: chat
				)
			);
			
			// Notify about chat update
			window.dispatchEvent(new Event('chatUpdated'));
			
			// Close dialog
			setRenameDialogOpen(false);
			setChatToRename(null);
			
		} catch (error) {
			console.error("Error renaming chat:", error);
			setRenameDialogOpen(false);
			setChatToRename(null);
		}
	}, [chatToRename, newChatName]);

	return (
		<div className="flex flex-col space-y-6">
			<nav className="grid gap-1">
				{/* Regular nav items */}
				{navItems.map((item, index) => (
					<Link
						key={index}
						href={item.href}
						className={cn(
							'flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors',
							pathname === item.href 
								? 'bg-gradient-to-l from-[#FF297F] to-[#4B8FFF] text-white' 
								: 'text-muted-foreground hover:bg-gradient-to-l hover:from-[#FF297F] hover:to-[#4B8FFF] hover:opacity-70 hover:text-white'
						)}
					>
						<div className="flex items-center gap-3 w-full">
							<item.icon className="h-4 w-4" />
							<span>{item.title}</span>
						</div>
					</Link>
				))}
				
				{/* New Chat button with handler to create chat and navigate */}
				<Button
					variant="ghost"
					onClick={(e) => {
						e.preventDefault();
						createNewChat();
					}}
					disabled={isCreatingChat}
					className={cn(
						'flex items-center justify-start gap-3 rounded-lg px-3 py-2 text-sm transition-colors',
						'text-muted-foreground hover:bg-gradient-to-l hover:from-[#FF297F] hover:to-[#4B8FFF] hover:opacity-70 hover:text-white',
						isCreatingChat && 'opacity-50 cursor-not-allowed'
					)}
				>
					<div className="flex items-center gap-3 w-full">
						<PlusCircle className="h-4 w-4" />
						<span>{isCreatingChat ? "Creating..." : "New Chat"}</span>
					</div>
				</Button>
			</nav>

			{expanded && chats.length > 0 && (
				<div className="space-y-1.5">
					<h4 className="px-3 text-sm font-semibold">Recent Chats</h4>
					<nav className="grid gap-1 text-sm text-muted-foreground max-h-[calc(77vh-280px)] overflow-y-auto pr-1 mac-scrollbar">
						{chats.map((chat) => (
							<div
								key={chat.id}
								className="relative group"
							>
								<a
									href={`/chat/${chat.id}`}
									onClick={(e) => handleChatClick(e, chat.id)}
									className={cn(
										'flex items-center gap-3 rounded-lg px-3 py-2 hover:bg-accent/50 hover:text-accent-foreground cursor-pointer',
										pathname === `/chat/${chat.id}` ? 'bg-accent/50 text-accent-foreground' : 'text-muted-foreground'
									)}
								>
									<div className="flex items-center gap-3 w-full">
										<MessageSquareText className="h-4 w-4" />
										<span className="truncate">{chat.name}</span>
									</div>
								</a>
								
								{/* Action buttons - visible on hover */}
								<div className="absolute right-2 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 transition-opacity flex items-center gap-1">
									{/* Rename button */}
									<button
										onClick={(e) => {
											e.preventDefault();
											e.stopPropagation();
											renameChat(chat.id);
										}}
										className="p-1 h-6 w-6 rounded-sm hover:bg-accent text-muted-foreground hover:text-accent-foreground"
										title="Rename chat"
									>
										<Pencil className="h-3.5 w-3.5" />
									</button>
									
									{/* Delete button */}
									<button
										onClick={(e) => {
											e.preventDefault();
											e.stopPropagation();
											deleteChat(chat.id);
										}}
										className="p-1 h-6 w-6 rounded-sm hover:bg-destructive/20 text-muted-foreground hover:text-destructive"
										title="Delete chat"
									>
										<Trash2 className="h-3.5 w-3.5" />
									</button>
								</div>
							</div>
						))}
					</nav>
				</div>
			)}

			{/* Delete confirmation dialog */}
			<AlertDialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
				<AlertDialogContent className="bg-card text-card-foreground">
					<AlertDialogHeader>
						<AlertDialogTitle>Delete Chat</AlertDialogTitle>
						<AlertDialogDescription>
							Are you sure you want to delete this chat? This action cannot be undone.
						</AlertDialogDescription>
					</AlertDialogHeader>
					<AlertDialogFooter>
						<AlertDialogCancel onClick={() => setChatToDelete(null)}>Cancel</AlertDialogCancel>
						<AlertDialogAction 
							onClick={confirmDeleteChat}
							className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
						>
							Delete
						</AlertDialogAction>
					</AlertDialogFooter>
				</AlertDialogContent>
			</AlertDialog>
			
			{/* Rename dialog */}
			<AlertDialog open={renameDialogOpen} onOpenChange={setRenameDialogOpen}>
				<AlertDialogContent className="bg-card text-card-foreground">
					<AlertDialogHeader>
						<AlertDialogTitle>Rename Chat</AlertDialogTitle>
						<AlertDialogDescription>
							Enter a new name for this chat.
						</AlertDialogDescription>
					</AlertDialogHeader>
					
					<Input 
						value={newChatName}
						onChange={(e) => setNewChatName(e.target.value)}
						placeholder="Enter chat name"
						className="mt-2"
						autoFocus
						onKeyDown={(e) => {
							if (e.key === 'Enter') {
								e.preventDefault();
								confirmRenameChat();
							}
						}}
					/>
					
					<AlertDialogFooter>
						<AlertDialogCancel onClick={() => {
							setChatToRename(null);
							setRenameDialogOpen(false);
						}}>
							Cancel
						</AlertDialogCancel>
						<AlertDialogAction 
							onClick={confirmRenameChat}
							disabled={!newChatName.trim()}
							className={!newChatName.trim() ? 'opacity-50 cursor-not-allowed' : ''}
						>
							Rename
						</AlertDialogAction>
					</AlertDialogFooter>
				</AlertDialogContent>
			</AlertDialog>
		</div>
	);
}

