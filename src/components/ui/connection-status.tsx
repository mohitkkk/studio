import React from 'react';
import { Alert } from "@/components/ui/alert";
import { Tooltip } from "@/components/ui/tooltip";

interface ConnectionStatusProps {
  status: 'connected' | 'disconnected' | 'connecting';
  showText?: boolean;
  className?: string;
}

export default function ConnectionStatus({ 
  status, 
  showText = false,
  className = ''
}: ConnectionStatusProps) {
  const getStatusInfo = () => {
    switch(status) {
      case 'connected':
        return {
          color: 'success',
          text: 'Connected',
          tooltip: 'Backend is connected'
        };
      case 'disconnected':
        return {
          color: 'danger',
          text: 'Disconnected',
          tooltip: 'Backend is disconnected'
        };
      case 'connecting':
        return {
          color: 'warning',
          text: 'Connecting',
          tooltip: 'Attempting to connect to backend'
        };
      default:
        return {
          color: 'warning',
          text: 'Unknown',
          tooltip: 'Connection status unknown'
        };
    }
  };

  const { color, text, tooltip } = getStatusInfo();

  // Simple indicator without Alert
  if (!showText) {
    return (
      <div 
        className={`h-3 w-3 rounded-full ${
          status === 'connected' 
            ? 'bg-green-500' 
            : status === 'connecting' 
              ? 'bg-yellow-500 animate-pulse' 
              : 'bg-red-500'
        } ${className}`}
        title={tooltip}
      />
    );
  }

  // Use the HeroUI Alert component for more detailed status
  return (
    <Alert 
      color={color as 'success' | 'danger' | 'warning'} 
      variant={status === 'connected' ? 'default' : status === 'disconnected' ? 'destructive' : 'warning'} 
      className={`py-1 px-3 text-xs rounded-full ${className}`}
    >
      {text}
    </Alert>
  );
}
