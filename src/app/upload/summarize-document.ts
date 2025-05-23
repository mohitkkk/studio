'use server';

import axios from 'axios';

// Get API URL from environment variables with fallback
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://13.202.208.115:3000';

/**
 * Process and summarize a document using our backend API instead of Gemini
 */
export async function summarizeDocument(fileId: string, fileName: string): Promise<{summary: string}> {
  try {
    console.log(`Sending document to API for processing: ${fileName} to ${API_URL}/process`);
    
    // Use our backend API for processing, not Gemini
    const response = await axios.post(`${API_URL}/process`, {
      filename: fileName,
      fileId: fileId
    });
    
    if (!response.data || !response.data.summary) {
      throw new Error(`No summary returned from processing service: ${JSON.stringify(response.data)}`);
    }
    
    return {
      summary: response.data.summary
    };
  } catch (error) {
    console.error('Error processing document:', error);
    throw new Error(`Error processing or summarizing document: ${error instanceof Error ? error.message : String(error)}`);
  }
}
