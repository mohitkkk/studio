"use client";

import { useState } from 'react';
import { Button } from './button';
import { Copy, Check } from 'lucide-react';
import { copyToClipboard } from '@/utils/clipboard';

interface CopyButtonProps {
  text: string;
  className?: string;
}

export function CopyButton({ text, className = '' }: CopyButtonProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    const success = await copyToClipboard(text);
    
    if (success) {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <Button 
      variant="ghost" 
      size="icon" 
      onClick={handleCopy} 
      className={className}
      title="Copy to clipboard"
    >
      {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
    </Button>
  );
}
