import React from 'react';
import { TooltipWrapper as Tooltip } from "@/components/ui/tooltip";
import { Button } from "@/components/ui/button";
import { AlertTriangle, Play, CheckCircle } from "lucide-react";
import LoadingSpinner from "@/components/ui/loading-spinner";

interface ProcessingStatusProps {
  status: string;
  isProcessing: boolean;
  onProcess: () => void;
  isReady: boolean;
}

/**
 * Component to display the processing status of a document
 */
export function ProcessingStatus({
  status,
  isProcessing,
  onProcess,
  isReady
}: ProcessingStatusProps) {
  // Determine status color and icon
  const getStatusDisplay = () => {
    if (isReady) {
      return {
        color: "text-green-500",
        icon: <CheckCircle className="h-3.5 w-3.5" />,
        text: "Ready"
      };
    }
    
    if (status.includes("failed")) {
      return {
        color: "text-red-500",
        icon: <AlertTriangle className="h-3.5 w-3.5" />,
        text: status
      };
    }
    
    return {
      color: "text-amber-500",
      icon: <AlertTriangle className="h-3.5 w-3.5" />,
      text: status
    };
  };
  
  const statusDisplay = getStatusDisplay();
  
  return (
    <div className="flex items-center">
      {!isReady && !status.includes("failed") && (
        <Tooltip content="Process document">
          <Button
            size="icon"
            variant="ghost"
            className="h-6 w-6 mr-1"
            onClick={onProcess}
            disabled={isProcessing}
          >
            {isProcessing ? (
              <LoadingSpinner size="sm" />
            ) : (
              <Play className="h-3.5 w-3.5" />
            )}
          </Button>
        </Tooltip>
      )}
      
      <Tooltip content={statusDisplay.text} className="max-w-[300px]">
        <div className={`text-xs flex items-center ${statusDisplay.color}`}>
          {statusDisplay.icon}
          <span className="ml-1 hidden md:inline truncate max-w-[120px]">
            {statusDisplay.text}
          </span>
        </div>
      </Tooltip>
    </div>
  );
}
