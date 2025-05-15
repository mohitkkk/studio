/**
 * Safely copy text to clipboard with environment checks
 */
export const copyToClipboard = async (text: string): Promise<boolean> => {
  // First check if we're in a browser environment
  if (typeof window === 'undefined' || typeof navigator === 'undefined') {
    console.warn('Clipboard API not available: not in browser environment');
    return false;
  }

  // Check if the clipboard API is available
  if (!navigator.clipboard) {
    console.warn('Clipboard API not available in this browser');
    return false;
  }

  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch (error) {
    console.error('Failed to copy to clipboard:', error);
    return false;
  }
};
