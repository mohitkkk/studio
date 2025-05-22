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
        <SidebarHeader className="p-4 ">
          <Link href="/">
          <SidebarTrigger className="flex-shrink-0  group-data-[collapsible=icon]:hidden mb-2" />
            <div className="flex flex-col items-center justify-center">
              {/* <Logo className="h-15 w-12" /> */}

                <span className="font-bold text-lg text-center tracking-[14px] bg-gradient-to-r from-[#00c853] via-[#8B5CF6] to-[#4B8FFF] bg-clip-text text-transparent animate-[gradient_8s_linear_infinite] relative before:absolute before:inset-0 before:bg-gradient-to-r before:from-[#4B8FFF] before:via-[#FF297F] before:to-[#00c853] before:bg-clip-text before:text-transparent before:animate-[gradient_8s_linear_infinite] before:content-['LUNEXA'] before:opacity-0 hover:before:opacity-100 before:transition-opacity"
                  style={{
                    backgroundSize: '200% auto'
                  }}
                >LUNEXA</span>
              <p className="text-[90%] justify-center ">Lunar Next-gen Assistant</p>
            </div>
            
          </Link>
        </SidebarHeader>
        
        {/* Only show in collapsed mode with clear styling */}
        <div className="hidden group-data-[collapsible=icon]:block px-2 mb-2">
          <SidebarTrigger className="w-full flex justify-center p-1.5 rounded-md bg-muted/50 hover:bg-accent/60 transition-colors" />
        </div>
        
        <SidebarContent className="flex-1 p-4"> {/* SidebarNav goes here */}
          <SidebarNav />
        </SidebarContent>


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
      <SidebarInset className="min-h-screen w-full flex flex-col overflow-y-auto">
        {/* Header for mobile view or consistent header */}
        <header className="p-4 md:hidden flex items-center justify-between border-b border-border">
          <Link href="/">
            <div className="flex items-center gap-2">
              <Logo className="h-6 w-6" />
              <span className="font-bold">LUNEXA</span>
            </div>
          </Link>
          <SidebarTrigger />
        </header>
        {/* This div is the direct parent of the page content (children) */}
        <div className="min-h-screen flex flex-col justify-between items-baseline overflow-y-auto bg-background">
          {children}
        </div>
      </SidebarInset>
    </div>
  );
}
