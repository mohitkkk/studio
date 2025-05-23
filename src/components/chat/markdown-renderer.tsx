"use client";

import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkBreaks from 'remark-breaks';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

interface MarkdownRendererProps {
  content: string;
  className?: string;
}

export default function MarkdownRenderer({ content, className = '' }: MarkdownRendererProps) {
  return (
    <div className={`markdown-content ${className}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkBreaks]}
        components={{
          // Headings
          h1: ({node, ...props}) => <h1 className="text-xl font-bold mt-3 mb-2" {...props} />,
          h2: ({node, ...props}) => <h2 className="text-lg font-bold mt-2.5 mb-1.5" {...props} />,
          h3: ({node, ...props}) => <h3 className="text-md font-bold mt-2 mb-1" {...props} />,
          h4: ({node, ...props}) => <h4 className="font-bold mt-1.5 mb-0.5" {...props} />,
          
          // Lists
          ul: ({node, ...props}) => <ul className="list-disc pl-5 my-2" {...props} />,
          ol: ({node, ...props}) => <ol className="list-decimal pl-5 my-2" {...props} />,
          li: ({node, ...props}) => <li className="mb-1" {...props} />,
          
          // Text formatting
          p: ({node, ...props}) => <p className="mb-2" {...props} />,
          em: ({node, ...props}) => <em className="italic" {...props} />,
          strong: ({node, ...props}) => <strong className="font-bold" {...props} />,
          
          // Code
          code: ({node, inline, className, children, ...props}: {
            node?: any;
            inline?: boolean;
            className?: string;
            children?: React.ReactNode;
            [key: string]: any;
          }) => {
            const match = /language-(\w+)/.exec(className || '');
            return !inline && match ? (
              <SyntaxHighlighter
                style={vscDarkPlus}
                language={match[1]}
                PreTag="div"
                className="rounded-md my-2 overflow-x-auto"
                {...props}
              >
                {String(children || '').replace(/\n$/, '')}
              </SyntaxHighlighter>
            ) : (
              <code className="bg-gray-800 px-1 py-0.5 rounded text-sm font-mono" {...props}>
                {children || ''}
              </code>
            );
          },
          
          // Blockquotes
          blockquote: ({node, ...props}) => (
            <blockquote className="border-l-4 border-gray-500 pl-4 py-1 my-2 text-gray-300" {...props} />
          ),
          
          // Links
          a: ({node, ...props}) => (
            <a 
              className="text-blue-400 hover:underline" 
              target="_blank" 
              rel="noopener noreferrer" 
              {...props} 
            />
          ),
          
          // Tables
          table: ({node, ...props}) => (
            <div className="overflow-x-auto my-2">
              <table className="min-w-full border-collapse" {...props} />
            </div>
          ),
          thead: ({node, ...props}) => <thead className="bg-gray-800" {...props} />,
          tbody: ({node, ...props}) => <tbody className="divide-y divide-gray-700" {...props} />,
          tr: ({node, ...props}) => <tr className="border-b border-gray-700" {...props} />,
          th: ({node, ...props}) => (
            <th className="px-3 py-2 text-left text-xs font-medium uppercase tracking-wider" {...props} />
          ),
          td: ({node, ...props}) => <td className="px-3 py-2 text-sm" {...props} />,
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
