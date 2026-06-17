//we will learn how to get feed from data from document to our llm and get answer according to documentimport { ChatOpenAI } from "@langchain/openai";
import { ChatOpenAI , OpenAIEmbeddings } from "@langchain/openai"

import { ChatPromptTemplate } from "@langchain/core/prompts";
import { createStuffDocumentsChain } from "@langchain/classic/chains/combine_documents";   

import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";   
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { createRetrievalChain } from "@langchain/classic/chains/retrieval";   

import { config } from "dotenv";
config();

//setup model
const model = new ChatOpenAI({
  model: "poolside/laguna-xs.2:free",
  temperature: 0.7,
  maxOutputTokens: 1000,
  apiKey: process.env.OPENROUTER_API_KEY,
  configuration: {
    baseURL: "https://openrouter.ai/api/v1",
  },
});


//in case of retrieval chain(vector store) , {context} and {input} is required 
const prompt = ChatPromptTemplate.fromTemplate(`
    Answer the user's question.
    Context: {context}  
    Question: {input}
`);
  
//create chain using createStuffDocumentsChain
const chain = await createStuffDocumentsChain({
    llm: model,
    prompt
});

// get text from document via cheerio
const loader = new CheerioWebBaseLoader("https://docs.langchain.com/oss/javascript/langchain/short-term-memory")
const docs = await loader.load();

//Split the documents by text spillter (RecursiveCharacterTextSplitter)
const spilitter = new RecursiveCharacterTextSplitter({
    chunkSize: 200,    
    chunkOverlap: 20,
})

const splitDocs = await spilitter.splitDocuments(docs);

//Vector store 
//store 
const embeddings = new OpenAIEmbeddings({
  apiKey: process.env.OPENROUTER_API_KEY,
  configuration: {
    baseURL: "https://openrouter.ai/api/v1",
  },
});

const vectorStore = await MemoryVectorStore.fromDocuments(
    splitDocs, 
    embeddings
);

//retrieve data
const retriever = vectorStore.asRetriever({ k: 2 });

// Create a retrieval chain
const retrievalChain = await createRetrievalChain({
  combineDocsChain: chain,
  retriever,
});

const response = await retrievalChain.invoke({
  input: "What do Chat models accepts?",
});
console.log(response);

