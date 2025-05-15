
import {NextRequest, NextResponse} from 'next/server';
import {ragBasedChat, RagBasedChatInput} from '@/ai/flows/rag-based-chat';
import {z} from 'zod';

// Define input schema for the API request body
const ApiChatInputSchema = z.object({
  message: z.string().min(1, {message: 'Message cannot be empty.'}),
  documentContent: z.string().optional(),
  // In the future, we could add other parameters like chatSessionId, LLM configurations, etc.
});

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const validationResult = ApiChatInputSchema.safeParse(body);

    if (!validationResult.success) {
      return NextResponse.json({error: 'Invalid request body', details: validationResult.error.format()}, {status: 400});
    }

    const {message, documentContent} = validationResult.data;

    // Prepare input for the RAG flow
    // If documentContent is not provided, an empty string is passed.
    // The ragBasedChat flow's prompt is designed to work with documentContent,
    // so if it's empty, the AI will likely respond based on its general knowledge
    // or indicate it needs a document for specific queries.
    const ragInput: RagBasedChatInput = {
      query: message,
      documentContent: documentContent || "", // Pass empty string if undefined
    };

    const ragOutput = await ragBasedChat(ragInput);

    // The problem description mentions "client ID" in response,
    // for now, we are just returning the answer. This can be extended.
    return NextResponse.json({answer: ragOutput.answer}, {status: 200});

  } catch (error: any) {
    console.error('Error in /api/chat endpoint:', error);
    
    let errorMessage = 'An internal server error occurred.';
    let errorDetails: any = undefined;

    if (error instanceof z.ZodError) {
      // This case should ideally be caught by validationResult.success check,
      // but as a fallback.
      errorMessage = 'Invalid request structure.';
      errorDetails = error.format();
      return NextResponse.json({error: errorMessage, details: errorDetails}, {status: 400});
    }
    
    // General error
    if (error.message) {
      errorMessage = error.message;
    }
    
    return NextResponse.json({error: errorMessage}, {status: 500});
  }
}
