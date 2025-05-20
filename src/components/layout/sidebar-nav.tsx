"use client";

import { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { cn } from '@/lib/utils';
import { MessageSquareText, Home, Settings, PlusCircle, Upload } from 'lucide-react';
import { Button } from '@/components/ui/button';

// List of navigation items
const navItems = [
	{
		title: 'Home',
		href: '/',
		icon: Home,
	},
	{
		title: 'New Chat',
		href: '/chat/new',
		icon: PlusCircle,
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

// Sample chat history items (would be fetched from server in real app)
const recentChats = [
	{ id: 'chat-1', title: 'Project Documentation', href: '/chat/chat-1' },
	{ id: 'chat-2', title: 'Sales Report Analysis', href: '/chat/chat-2' },
	{ id: 'chat-3', title: 'User Manual Query', href: '/chat/chat-3' },
];

export default function SidebarNav() {
	const pathname = usePathname();
	const [expanded, setExpanded] = useState(true);

	return (
		<div className="flex flex-col space-y-6">
			<nav className="grid gap-1">
				{navItems.map((item, index) => (
					<Link
						key={index}
						href={item.href}
						className={cn(
							'flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors hover:bg-accent/50',
							pathname === item.href ? 'bg-accent text-accent-foreground' : 'text-muted-foreground'
						)}
					>
						<div className="flex items-center gap-3 w-full">
							<item.icon className="h-4 w-4" />
							<span>{item.title}</span>
						</div>
					</Link>
				))}
			</nav>

			{expanded && (
				<div className="space-y-1.5">
					<h4 className="px-3 text-sm font-semibold">Recent Chats</h4>
					<nav className="grid gap-1 text-sm text-muted-foreground">
						{recentChats.map((chat) => (
							<Link
								key={chat.id}
								href={chat.href}
								className={cn(
									'flex items-center gap-3 rounded-lg px-3 py-2 hover:bg-accent/50 hover:text-accent-foreground',
									pathname === chat.href ? 'bg-accent/50 text-accent-foreground' : 'text-muted-foreground'
								)}
							>
								<div className="flex items-center gap-3 w-full">
									<MessageSquareText className="h-4 w-4" />
									<span className="truncate">{chat.title}</span>
								</div>
							</Link>
						))}
					</nav>
				</div>
			)}
		</div>
	);
}

