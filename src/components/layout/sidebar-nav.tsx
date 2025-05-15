
"use client";

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { SidebarMenu, SidebarMenuItem, SidebarMenuButton } from '@/components/ui/sidebar';
import { MessageSquare, UploadCloud, History, Settings2, FileText } from 'lucide-react';
import { cn } from '@/lib/utils';

const navItems = [
  { href: '/', label: 'Chat', icon: MessageSquare },
  { href: '/upload', label: 'File Upload', icon: UploadCloud },
  { href: '/history', label: 'Chat History', icon: History },
  { href: '/config', label: 'Configuration', icon: Settings2 },
];

export default function SidebarNav() {
  const pathname = usePathname();

  return (
    <SidebarMenu>
      {navItems.map((item) => (
        <SidebarMenuItem key={item.href}>
          <Link href={item.href} legacyBehavior passHref>
            <SidebarMenuButton
              asChild
              isActive={pathname === item.href}
              tooltip={{ children: item.label, side: 'right', className: 'bg-popover text-popover-foreground' }}
              className={cn(
                "w-full justify-start",
                pathname === item.href && "bg-sidebar-accent text-sidebar-accent-foreground font-semibold"
              )}
            >
              <a>
                <item.icon className="h-5 w-5" />
                <span className="group-data-[collapsible=icon]:hidden">{item.label}</span>
              </a>
            </SidebarMenuButton>
          </Link>
        </SidebarMenuItem>
      ))}
    </SidebarMenu>
  );
}

    