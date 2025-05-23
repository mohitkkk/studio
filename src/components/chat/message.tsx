
"use client";

import React, { useEffect, useRef } from 'react';
// import DOMPurify from 'dompurify'; // TODO: Uncomment this line after ensuring 'dompurify' is installed (e.g., by running `npm install dompurify`)
// import ReactMarkdown from 'react-markdown'; // TODO: Install and uncomment
// import remarkGfm from 'remark-gfm'; // TODO: Install and uncomment
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { cn } from '@/lib/utils';
import { User, Bot } from 'lucide-react';

interface MessageProps {
  content: string;
  isUser?: boolean;
  format?: 'html' | 'markdown';
  timestamp?: string;
}

export function ChatMessage({ content, isUser = false, format = 'html', timestamp }: MessageProps) {
  // Choose styling based on message sender
  const containerClass = cn(
    "flex gap-3 p-4 rounded-lg",
    isUser ? "bg-primary/10" : "bg-muted/50"
  );

  const messageContentRef = useRef<HTMLDivElement>(null);

  // Apply syntax highlighting to code blocks in HTML after render
  useEffect(() => {
    // if (format === 'html' && messageContentRef.current) {
    //   // Find all pre > code elements and enhance them
    //   const codeBlocks = messageContentRef.current.querySelectorAll('pre code');
    //   if (codeBlocks.length > 0) {
    //     // import('highlight.js').then((hljs) => { // TODO: Install and uncomment
    //     //   codeBlocks.forEach((block) => {
    //     //     hljs.default.highlightElement(block as HTMLElement);
    //     //   });
    //     // });
    //   }
    // }
  }, [content, format]);

  // Function to render content based on format
  const renderContent = () => {
    if (!content) return <p className="text-muted-foreground italic">Empty message</p>;

    if (format === 'markdown') {
      // TODO: Markdown rendering is temporarily disabled. Install 'react-markdown' and 'remark-gfm' and uncomment this section.
      // return (
      //   <div className="prose prose-sm dark:prose-invert max-w-none">
      //     <ReactMarkdown
      //       remarkPlugins={[remarkGfm]}
      //       components={{
      //         code({node, inline, className, children, ...props}) {
      //           const match = /language-(\w+)/.exec(className || '');
      //           return !inline && match ? (
      //             <SyntaxHighlighter
      //               {...props}
      //               style={oneDark}
      //               language={match[1]}
      //               PreTag="div"
      //             >
      //               {String(children).replace(/\n$/, '')}
      //             </SyntaxHighlighter>
      //           ) : (
      //             <code {...props} className={className}>
      //               {children}
      //             </code>
      //           );
      //         }
      //       }}
      //     >
      //       {content}
      //     </ReactMarkdown>
      //   </div>
      // );
      // Fallback for when markdown is disabled:
      return (
        <div
          ref={messageContentRef}
          className="prose prose-sm dark:prose-invert max-w-none html-content"
          dangerouslySetInnerHTML={{
            __html: content.replace(/</g, "&lt;").replace(/>/g, "&gt;") // Basic HTML escaping as a fallback
          }}
        />
      );
    } else {
      // Default to HTML with enhanced sanitization and styling
      // TODO: Re-enable DOMPurify once the 'dompurify' package is correctly installed.
      // Using raw content directly in dangerouslySetInnerHTML is a SECURITY RISK if the content is not trusted.
      // For DOMPurify to work, you'd uncomment its import and use:
      // const sanitizedHtml = DOMPurify.sanitize(content, {
      //   USE_PROFILES: { html: true },
      //   ALLOWED_TAGS: [
      //     'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'br', 'ul', 'ol', 'li',
      //     'strong', 'em', 'pre', 'code', 'a', 'blockquote', 'table', 'tr',
      //     'th', 'td', 'hr', 'img', 'span', 'div', 'details', 'summary'
      //   ],
      //   ALLOWED_ATTR: [
      //     'href', 'src', 'alt', 'title', 'class', 'target', 'rel',
      //     'id', 'style', 'data-language', 'data-line'
      //   ]
      // });
      // For now, using raw content due to the missing module:
      const htmlToRender = content; // WARNING: SECURITY RISK. Install and use DOMPurify.

      return (
        <div
          ref={messageContentRef}
          className="prose prose-sm dark:prose-invert max-w-none html-content"
          dangerouslySetInnerHTML={{
            __html: htmlToRender
          }}
        />
      );
    }
  };

  return (
    <div className={containerClass}>
      <div className="flex-shrink-0 mt-1">
        {isUser ? (
          <div className="h-8 w-8 rounded-full bg-primary/10 flex items-center justify-center">
            <User className="h-5 w-5 text-primary" />
          </div>
        ) : (
          <div className="h-8 w-8 rounded-full bg-muted flex items-center justify-center">
            <Bot className="h-5 w-5 text-primary" />
          </div>
        )}
      </div>

      <div className="flex-1 space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium">
            {isUser ? 'You' : 'LUNEXA Assistant'}
          </span>
          {timestamp && (
            <span className="text-xs text-muted-foreground">
              {timestamp}
            </span>
          )}
        </div>

        <div className="message-content">
          {renderContent()}
        </div>
      </div>
    </div>
  );
}
