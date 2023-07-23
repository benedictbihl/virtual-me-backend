import { serve } from "http/server.ts";

import { RetrievalQAChain } from "langchain/chains";
import { BufferMemory, ChatMessageHistory } from "langchain/memory";
import { ChatOpenAI } from "langchain/chat_models/openai";
import { CallbackManager } from "langchain/callbacks";
import {
  ChatPromptTemplate,
  HumanMessagePromptTemplate,
  SystemMessagePromptTemplate,
  MessagesPlaceholder,
} from "langchain/prompts";
import { SupabaseVectorStore } from "langchain/vectorstores/supabase";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";

import { corsHeaders } from "../_shared/cors.ts";
import { createClient } from "../_shared/supabase-client.ts";

const chatPrompt = ChatPromptTemplate.fromPromptMessages([
  SystemMessagePromptTemplate.fromTemplate(
    "My Name is Benedict and I want you to act as my virtual representation. Answer every question as if you were me, but do not make up any information."
  ),
  new MessagesPlaceholder("history"),
  HumanMessagePromptTemplate.fromTemplate("{question}"),
]);

serve(async (req) => {
  // This is needed if you're planning to invoke your function from a browser.
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders });
  }

  try {
    const client = createClient(req);

    const vectorStore = await SupabaseVectorStore.fromExistingIndex(
      new OpenAIEmbeddings(),
      {
        client,
        tableName: "documents",
        queryName: "match_documents",
      }
    );
    const { input, history } = await req.json();
    // For a streaming response we need to use a TransformStream to
    // convert the LLM's callback-based API into a stream-based API.
    const encoder = new TextEncoder();
    const stream = new TransformStream();
    const writer = stream.writable.getWriter();

    const llm = new ChatOpenAI({
      streaming: true,
      callbackManager: CallbackManager.fromHandlers({
        handleLLMNewToken: async (token) => {
          await writer.ready;
          await writer.write(encoder.encode(`data: ${token}\n\n`));
        },
        handleLLMEnd: async () => {
          await writer.ready;
          await writer.close();
        },
        handleLLMError: async (e) => {
          await writer.ready;
          await writer.abort(e);
        },
      }),
    });

    const chain = RetrievalQAChain.fromLLM(llm, vectorStore.asRetriever(), {});
    const chatHistory = new ChatMessageHistory(history);
    chain.memory = new BufferMemory({
      returnMessages: true,
      memoryKey: "history",
      chatHistory,
    });

    chain
      .call({
        query: await chatPrompt.format({
          question: input,
          history: chatHistory.messages,
        }),
      })
      .catch((e) => console.error(e));

    return new Response(stream.readable, {
      headers: { ...corsHeaders, "Content-Type": "text/event-stream" },
    });
  } catch (e) {
    return new Response(JSON.stringify({ error: e.message }), {
      status: 500,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }
});
