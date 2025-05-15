
import { cn } from "@/lib/utils";

interface LoadingSpinnerProps {
  className?: string;
  dotClassName?: string;
  size?: 'sm' | 'md' | 'lg';
}

export default function LoadingSpinner({ className, dotClassName, size = 'md' }: LoadingSpinnerProps) {
  const dotSizeClasses = {
    sm: 'w-1.5 h-1.5',
    md: 'w-2 h-2',
    lg: 'w-2.5 h-2.5',
  };

  return (
    <div className={cn("flex items-center justify-center space-x-1", className)}>
      <div className={cn("pulsing-dot", dotSizeClasses[size], dotClassName)} style={{ animationDelay: '0s' }}></div>
      <div className={cn("pulsing-dot", dotSizeClasses[size], dotClassName)} style={{ animationDelay: '0.2s' }}></div>
      <div className={cn("pulsing-dot", dotSizeClasses[size], dotClassName)} style={{ animationDelay: '0.4s' }}></div>
    </div>
  );
}

    