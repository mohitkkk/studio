'use server';

/**
 * @fileOverview RAG (Retrieval-Augmented Generation) flow for document summarization and question answering.
 *
 * - ragBasedChat - A function that handles the RAG process.
 * - RagBasedChatInput - The input type for the ragBasedChat function.
 * - RagBasedChatOutput - The return type for the ragBasedChat function.
 */

import {ai} from '@/ai/genkit';
import {z} from 'genkit';

const RagBasedChatInputSchema = z.object({
  documentContent: z
    .string()
    .describe('Content of the uploaded document.'),
  query: z.string().describe('User query related to the document content.'),
});
export type RagBasedChatInput = z.infer<typeof RagBasedChatInputSchema>;

const RagBasedChatOutputSchema = z.object({
  answer: z.string().describe('Answer to the user query based on the document content, with citations.'),
});
export type RagBasedChatOutput = z.infer<typeof RagBasedChatOutputSchema>;

export async function ragBasedChat(input: RagBasedChatInput): Promise<RagBasedChatOutput> {
  return ragBasedChatFlow(input);
}

const ragBasedChatPrompt = ai.definePrompt({
  name: 'ragBasedChatPrompt',
  input: {schema: RagBasedChatInputSchema},
  output: {schema: RagBasedChatOutputSchema},
  prompt: `You are a chatbot that answers questions based on the content of uploaded documents.  You must cite the specific sections of the document used to answer the question.

Document Content:
{{{documentContent}}}

User Query:
{{{query}}}

Answer:`,
});

const ragBasedChatFlow = ai.defineFlow(
  {
    name: 'ragBasedChatFlow',
    inputSchema: RagBasedChatInputSchema,
    outputSchema: RagBasedChatOutputSchema,
  },
  async input => {
    const {output} = await ragBasedChatPrompt(input);
    return output!;
  }
);
