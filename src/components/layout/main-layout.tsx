
"use client";

import type { PropsWithChildren } from 'react';
import { Sidebar, SidebarInset, SidebarTrigger, SidebarHeader, SidebarContent, SidebarFooter } from '@/components/ui/sidebar';
import SidebarNav from './sidebar-nav';
import { Button } from '@/components/ui/button';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { LogOut, Settings } from 'lucide-react';
import Link from 'next/link';

export default function MainLayout({ children }: PropsWithChildren) {
  return (
    <div className="flex min-h-screen">
      <Sidebar collapsible="icon" variant="floating" side="left" className="">
        <SidebarHeader className="p-4 flex items-center justify-between">
           <Link href="/" className="flex items-center gap-2">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-8 h-8 text-primary">
              <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm-5-9h2v2H7v-2zm4 0h2v2h-2v-2zm4 0h2v2h-2v-2z" />
            </svg>
            <h1 className="text-2xl font-semibold text-foreground group-data-[collapsible=icon]:hidden">NebulaChat</h1>
          </Link>
          <div className="group-data-[collapsible=icon]:hidden">
            <SidebarTrigger />
          </div>
        </SidebarHeader>
        <SidebarContent className="flex-1 p-4">
          <SidebarNav />
        </SidebarContent>
        <SidebarFooter className="p-4 mt-auto"> {/* Removed border-t */}
           <div className="p-3 rounded-lg bg-card text-card-foreground shadow-md group-data-[collapsible=icon]:p-2">
            <div className="flex items-center gap-3 group-data-[collapsible=icon]:flex-col group-data-[collapsible=icon]:items-center group-data-[collapsible=icon]:gap-2">
              <Avatar className="h-10 w-10">
                <AvatarImage src="https://placehold.co/100x100.png" alt="User Avatar" data-ai-hint="user avatar" />
                <AvatarFallback>NC</AvatarFallback>
              </Avatar>
              <div className="group-data-[collapsible=icon]:hidden">
                <p className="text-sm font-medium">User Name</p>
                <p className="text-xs text-muted-foreground">user@example.com</p>
              </div>
            </div>
            <Button variant="ghost" className="w-full justify-start mt-3 group-data-[collapsible=icon]:justify-center group-data-[collapsible=icon]:mt-2 group-data-[collapsible=icon]:w-auto group-data-[collapsible=icon]:p-1 group-data-[collapsible=icon]:h-auto">
              <LogOut className="mr-2 h-4 w-4 group-data-[collapsible=icon]:mr-0" />
              <span className="group-data-[collapsible=icon]:hidden">Log Out</span>
            </Button>
          </div>
        </SidebarFooter>
      </Sidebar>
      <SidebarInset className="flex-1 flex flex-col overflow-hidden">
        {/* Header for mobile view or consistent header */}
        <header className="p-4 md:hidden flex items-center justify-between border-b border-border"> {/* Added border for mobile header clarity */}
          <Link href="/" className="flex items-center gap-2">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-6 h-6 text-primary">
              <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm-5-9h2v2H7v-2zm4 0h2v2h-2v-2zm4 0h2v2h-2v-2z" />
            </svg>
            <h1 className="text-xl font-semibold text-foreground">NebulaChat</h1>
          </Link>
          <SidebarTrigger />
        </header>
        <main className="flex-1 overflow-y-auto p-6 bg-background animated-smoke-bg">
           {children}
        </main>
      </SidebarInset>
    </div>
  );
}
