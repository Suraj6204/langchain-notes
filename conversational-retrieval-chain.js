// Load data from documents and answer questions using a retriever-augmented LLM
import { ChatOpenAI } from "@langchain/openai";
import { HuggingFaceInferenceEmbeddings } from "@langchain/community/embeddings/hf";

import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import { createStuffDocumentsChain } from "@langchain/classic/chains/combine_documents";

import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { createRetrievalChain } from "@langchain/classic/chains/retrieval";
// import { HumanMessage, AIMessage } from "langchain";
// import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever";

import { HumanMessage, AIMessage } from "@langchain/core/messages";
import { createHistoryAwareRetriever } from "@langchain/classic/chains/history_aware_retriever";

import { config } from "dotenv";
config();

//Load data and store in vector store
const createVectorStore = async () => {
  // get text from document via cheerio
  const loader = new CheerioWebBaseLoader(
    "https://docs.langchain.com/oss/javascript/langchain/models",
  );
  const docs = await loader.load();

  //Split the documents by text spillter (RecursiveCharacterTextSplitter)
  const spilitter = new RecursiveCharacterTextSplitter({
    chunkSize: 200,
    chunkOverlap: 20,
  });
  const splitDocs = await spilitter.splitDocuments(docs);

  // ##Vector store##
  //store
  const embeddings = new HuggingFaceInferenceEmbeddings({
    model: "sentence-transformers/all-MiniLM-L6-v2",
  });

  const vectorStore = await MemoryVectorStore.fromDocuments(
    splitDocs,
    embeddings,
  );
  return vectorStore;
};

//create retrieval chain(model,prompt,vector store)
const createChain = async (vectorStore) => {
  const model = new ChatOpenAI({
    model: "nex-agi/nex-n2-pro:free",
    temperature: 0.7,
    maxOutputTokens: 1000,
    apiKey: process.env.OPENROUTER_API_KEY,
    configuration: {
      baseURL: "https://openrouter.ai/api/v1",
    },
  });

  //in case of retrieval chain(vector store) , {context} and {input} is required
  // const prompt = ChatPromptTemplate.fromTemplate(`
  //     Answer the user's question.
  //     Context: {context}
  //     Question: {input}
  // `);
  const prompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      "You are a helpful assistant that answers questions based on the provided context: {context}",
    ],
    new MessagesPlaceholder("chat_history"), //converts array to string
    ["user", "Question: {input}"],
  ]);

  //createStuffDocumentsChain - allows to pass array of documents which will be used as context
  const chain = await createStuffDocumentsChain({
    llm: model,
    prompt,
  });

  //fetch relevant data from vector store
  const retriever = vectorStore.asRetriever({ k: 2 });

  //this is the prompt used in history aware retriever
  const retrievalPrompt = ChatPromptTemplate.fromMessages([
    new MessagesPlaceholder("chat_history"),
    ["user", "{input}"],
    [
      "user",
      "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation",
    ],
  ]);

  // it converts user input and chat history into a single query to fetch relevant data from vector store
  let historyAwareRetriever;
  try {
    historyAwareRetriever = await createHistoryAwareRetriever({
      llm: model,
      retriever,
      rephrasePrompt: retrievalPrompt,
    });
  } catch (e) {
    console.warn(
      "History-aware retriever failed, falling back to simple retriever:",
      e?.message ?? e,
    );
    historyAwareRetriever = retriever;
  }

  // Create the retrieval chain using the history-aware retriever.
//   const rawConversationChain = await createRetrievalChain({
//     combineDocsChain: chain,
//     retriever: historyAwareRetriever,
//   });

  const conversationChain = {
    invoke: async ({ input, chat_history }) => {
      const docs = await retriever.invoke(input);

      // call combine-docs chain which expects `context` key with documents
      return chain.invoke({ input, context: docs, chat_history });
    },
  };

  return conversationChain;
};

const vectorStore = await createVectorStore();
const chain = await createChain(vectorStore);

const chatHistory = [
  new HumanMessage("Hello"),
  new AIMessage("Hi, how can I help you?"),
  new HumanMessage("My name is suraj"),
  new AIMessage("Nice to meet you, Suraj! How can I assist you today?"),
];
const response = await chain.invoke({
  input:"how to install and import initChatModel in langchain? Start with my name!",
  chat_history: chatHistory, //we cant pass array here(we have to convert it to string by MessagesPlaceholder)
});

console.log(response);
