"use client";

import type { PropsWithChildren } from 'react';
import { Sidebar, SidebarInset, SidebarTrigger, SidebarHeader, SidebarContent, SidebarFooter } from '@/components/ui/sidebar';
import SidebarNav from './sidebar-nav';
import { Button } from '@/components/ui/button';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { LogOut, Settings, FileText, FolderOpen, X } from 'lucide-react'; 
import Link from 'next/link';
import { Logo } from '@/components/ui/logo';
import { cn } from '@/lib/utils';

export default function MainLayout({ children }: PropsWithChildren) {
  return (
    <div className="flex min-h-screen w-full overflow-hidden">
      <Sidebar collapsible="icon" variant="floating" side="left" className="">
        <SidebarHeader className="p-4 flex items-center justify-between">
          {/* Remove legacyBehavior prop and move className to the div */}
          <Link href="/">
            <div className="flex items-center gap-2">
              <Logo className="h-6 w-6" />
              <span className="font-bold">NebulaChat</span>
            </div>
          </Link>
          <SidebarTrigger className="flex-shrink-0 group-data-[collapsible=icon]:hidden" />
        </SidebarHeader>
        
        {/* Only show in collapsed mode with clear styling */}
        <div className="hidden group-data-[collapsible=icon]:block px-2 mb-2">
          <SidebarTrigger className="w-full flex justify-center p-1.5 rounded-md bg-muted/50 hover:bg-accent/60 transition-colors" />
        </div>
        
        <SidebarContent className="flex-1 p-4"> {/* SidebarNav goes here */}
          <SidebarNav />
        </SidebarContent>

        {/* Files Box Section - Placed after SidebarContent, before SidebarFooter */}
        <div className="px-4 pb-4 group-data-[collapsible=icon]:px-2 group-data-[collapsible=icon]:pb-2">
          <div className="p-3 rounded-lg bg-card text-card-foreground shadow-md group-data-[collapsible=icon]:p-2">
            {/* Title visible when expanded */}
            <h3 className="text-sm font-semibold mb-2 group-data-[collapsible=icon]:hidden">
              Selected Files
            </h3>
            
            {/* File list area or placeholder text visible when expanded */}
            <div className="space-y-1 text-xs mb-3 group-data-[collapsible=icon]:hidden min-h-[60px] max-h-[150px] overflow-y-hidden">
              <p className="text-muted-foreground italic px-1 py-2">
                Use 'Manage Files' to add documents for chat context.
              </p>
              
              Example of file items to be added later:
              <div className="flex items-center justify-between p-1.5 rounded hover:bg-accent/50">
                <FileText className="h-4 w-4 mr-2 shrink-0 text-muted-foreground" />
                <span className="truncate flex-grow">document_example_one.pdf</span>
                <Button variant="ghost" size="icon" className="h-5 w-5 shrink-0 ml-1 p-0 hover:bg-destructive/20">
                  <X className="h-3 w-3 text-muted-foreground group-hover:text-destructive" />
                </Button>
              </div>
             
            </div>
            
            {/* Icon and text for collapsed state */}
            <div className="hidden group-data-[collapsible=icon]:flex group-data-[collapsible=icon]:flex-col group-data-[collapsible=icon]:items-center group-data-[collapsible=icon]:justify-center py-3">
              <FileText className="h-5 w-5 text-muted-foreground" />
              <p className="text-[10px] text-muted-foreground mt-1 leading-tight">Files</p>
            </div>
            
            {/* Remove legacyBehavior prop and update Link for "Manage Files" button */}
            <Link href="/upload" className="w-full group-data-[collapsible=icon]:hidden">
              <Button variant="outline" size="sm" className="w-full">
                <FolderOpen className="h-3.5 w-3.5 mr-1.5" />
                Manage Files
              </Button>
            </Link>
            
            {/* Remove legacyBehavior prop and update Link for collapsed icon button */}
            <Link
              href="/upload"
              className="hidden group-data-[collapsible=icon]:flex w-full h-8 items-center justify-center mt-1 hover:bg-accent/80"
              title="Manage Files">
              <Button variant="ghost" size="icon">
                <FolderOpen className="h-4 w-4" />
              </Button>
            </Link>
          </div>
        </div>

        <SidebarFooter className="p-4 pt-0 mt-auto">
           {/* User Profile Card */}
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
      <SidebarInset className="min-h-screen w-full flex flex-col overflow-y-hidden">
        {/* Header for mobile view or consistent header */}
        <header className="p-4 md:hidden flex items-center justify-between border-b border-border">
          {/* Remove legacyBehavior prop from mobile Link as well */}
          <Link href="/">
            <div className="flex items-center gap-2">
              <Logo className="h-6 w-6" />
              <span className="font-bold">NebulaChat</span>
            </div>
          </Link>
          <SidebarTrigger />
        </header>
        {/* This div is the direct parent of the page content (children) */}
        <div className="min-h-screen flex flex-col justify-between items-baseline overflow-hidden bg-background">
          {children}
        </div>
      </SidebarInset>
    </div>
  );
}
