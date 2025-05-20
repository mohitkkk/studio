import React from 'react';

/**
 * Helper utility to safely update React state only when values have changed,
 * to prevent infinite update loops.
 * 
 * @param currentValue The current state value
 * @param newValue The potential new state value
 * @param setter The React setState function
 * @param compareFunction Optional custom compare function
 * @returns boolean indicating if state was updated
 */
export function safeSetState<T>(
  currentValue: T, 
  newValue: T, 
  setter: (value: T) => void,
  compareFunction?: (a: T, b: T) => boolean
): boolean {
  // Use custom compare function or default deep comparison
  const areEqual = compareFunction 
    ? compareFunction(currentValue, newValue)
    : JSON.stringify(currentValue) === JSON.stringify(newValue);
  
  if (!areEqual) {
    setter(newValue);
    return true;
  }
  
  return false;
}

/**
 * Create a stable reference to a function that doesn't change on re-renders
 * but can still access the latest state values.
 * 
 * @param callback The function to stabilize
 * @returns A stable function reference
 */
export function useStableCallback<T extends (...args: any[]) => any>(callback: T): T {
  const callbackRef = React.useRef(callback);
  
  React.useEffect(() => {
    callbackRef.current = callback;
  }, [callback]);
  
  return React.useCallback((...args: any[]) => {
    return callbackRef.current(...args);
  }, []) as T;
}
