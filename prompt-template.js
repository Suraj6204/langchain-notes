import {ChatOpenAI} from "@langchain/openai";
import {ChatPromptTemplate} from "@langchain/core/prompts";

import {config} from "dotenv";
config();

//create the model 
const model = new ChatOpenAI({
    model: "poolside/laguna-xs.2:free",
    maxOutputTokens: 1000,
    temperature:0.7,
    apiKey: process.env.OPENROUTER_API_KEY,
    configuration:{
        baseURL: "https://openrouter.ai/api/v1",
    },
});

//create prompt template
// const prompt = ChatPromptTemplate.fromTemplate(`
//     You are a hr recruiter . write a email regarging following topic . use headings and bold words {input}
// `);

//second way 
const prompt = ChatPromptTemplate.fromMessages([
    ["system" , "Generate joke based on word given by user"],
    ["human", "{input}"],
]);


//creat chain using LCEL
const chain = prompt.pipe(model); // prompt -> model 

//call chain or print response
const res = await chain.invoke({ 
    input: "dog"
});

console.log(res.content);

//prompt template is useful when you want to pass some pre prompts to llm with user input prompt