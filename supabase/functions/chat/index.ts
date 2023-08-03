import { serve } from "http/server.ts";

import { ConversationalRetrievalQAChain } from "langchain/chains";
import { BufferMemory, ChatMessageHistory } from "langchain/memory";
import { ChatOpenAI } from "langchain/chat_models/openai";
import { CallbackManager } from "langchain/callbacks";
import { PromptTemplate } from "langchain/prompts";

import { SupabaseVectorStore } from "langchain/vectorstores/supabase";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";

import { corsHeaders } from "../_shared/cors.ts";
import { createClient } from "../_shared/supabase-client.ts";

serve(async (req) => {
  const CUSTOM_QA_PROMPT = `You are presented the following pieces of context with information about Benedict, a 27 Year old Developer living in Munich. Use them to answer the question at the end. If you don't know the answer, suggest the user asks the real Benedict, don't try to make up an answer.
  {context}

  Question: {question}
  Helpful Answer:`;

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
      }
    );
    const { input, history, conversationID } = await req.json();
    // For a streaming response we need to use a TransformStream to
    // convert the LLM's callback-based API into a stream-based API.
    const encoder = new TextEncoder();
    const stream = new TransformStream();
    const writer = stream.writable.getWriter();
    const chatHistory = new ChatMessageHistory(history);

    let answer = "";

    const streamingModel = new ChatOpenAI({
      streaming: true,
      callbackManager: CallbackManager.fromHandlers({
        handleLLMNewToken: async (token) => {
          answer += token;
          await writer.ready;
          await writer.write(encoder.encode(`data: ${token}\n\n`));
        },
        handleLLMEnd: async () => {
          console.log("Writing to DB");
          const chats = chatHistory.messages.map((m) => {
            if (m.kwargs) return m.kwargs.content;
          });
          chats.push(answer);

          const { error } = await client.from("chats").upsert([
            {
              id: conversationID,
              created_at: new Date().toLocaleString("en-US", {
                timeZone: "Europe/Berlin",
              }),
              chat: chats,
            },
          ]);
          if (error) {
            console.error(error);
          }
          await writer.ready;
          await writer.close();
        },
        handleLLMError: async (e) => {
          await writer.ready;
          await writer.abort(e);
        },
      }),
    });

    const chain = ConversationalRetrievalQAChain.fromLLM(
      streamingModel,
      vectorStore.asRetriever(),
      {
        returnSourceDocuments: true,
        memory: new BufferMemory({
          memoryKey: "chat_history",
          inputKey: "question", // The key for the input to the chain
          outputKey: "text", // The key for the final conversational output of the chain
          returnMessages: true, // If using with a chat model
        }),
        questionGeneratorChainOptions: {
          llm: new ChatOpenAI({}),
        },
        qaChainOptions: {
          type: "stuff",
          prompt: PromptTemplate.fromTemplate(CUSTOM_QA_PROMPT),
        },
      }
    );

    chain
      .call({
        question: input,
        chat_history: chatHistory.messages,
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
