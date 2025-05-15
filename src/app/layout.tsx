
import type { Metadata } from 'next';
import { ibmPlexMono } from '@/fonts';
import './globals.css';
import { SidebarProvider } from '@/components/ui/sidebar';
import { Toaster } from '@/components/ui/toaster';
import MainLayout from '@/components/layout/main-layout';

export const metadata: Metadata = {
  title: 'NebulaChat',
  description: 'Futuristic AI Chat Application',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className={`${ibmPlexMono.variable} antialiased font-mono`}>
        <SidebarProvider defaultOpen={true}>
          <MainLayout>{children}</MainLayout>
        </SidebarProvider>
        <Toaster />
      </body>
    </html>
  );
}

    