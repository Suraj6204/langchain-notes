import { ChatOpenAI } from "@langchain/openai";
import {config} from "dotenv";
config();

const model = new ChatOpenAI({
    model: "nex-agi/nex-n2-pro:free",
    maxOutputTokens: 1000,
    temperature:0.7,
    apiKey: process.env.OPENROUTER_API_KEY,
    configuration: {
        baseURL: "https://openrouter.ai/api/v1",
    },
});

const res = await model.invoke("how will world end");
console.log(res.content);


//can't giver user input - will use prompt template
// cant see output markdown in format