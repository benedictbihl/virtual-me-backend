# Virtual Me Backend

Contains the all the code necessary to create a chatbot backend that uses personal documents to answer questions. In my case, I use it for a section on my homepage where folks can ask questions about me and my work and ideally get a response that is as close to what I would say as possible.

Loosely based on the backend part of the [langchain supabase template](https://github.com/langchain-ai/langchain-template-supabase), it houses some python scripts to

- transcribe audio to text, using a local whisper model (so instead of typing a bunch of stuff, I can just record it)
- to transform docs into a q and a format, using the doctran lib + openAI ([see here why](https://python.langchain.com/docs/modules/data_connection/document_transformers/integrations/doctran_interrogate_document))
- to embed the docs into a vector space, using openAI and a supabase vector store

as well as a supabase function to query the vector space for the most similar document to a given query and use the openai api to generate a response.

## Supabase Local Setup

0. Make sure you have [Docker](https://www.docker.com/) installed and running

1. Install dependencies, including the Supabase CLI

```bash
yarn
```

**Note**: If you install the Supabase CLI using a different method you have to make sure you are on version 1.49.4 as more recent versions currently suffer from an issue which prevents this from working correctly.

3. Create supabase functions env file

```bash
echo "OPENAI_API_KEY=sk-xxx" > supabase/.env
```

4. If not already running, start Docker. Learn how to do this for your OS [here](https://docs.docker.com/desktop/).

5. Start the supabase project.

```bash
npx supabase start
```

6. Start the supabase functions locally

```bash
yarn supabase:dev
```

7. (Not part of this repo) Start the frontend locally or invoke the function using curl

```bash
curl -i --location --request POST 'http://localhost:50321/functions/v1/chat' \
  --header 'Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6ImFub24iLCJleHAiOjE5ODM4MTI5OTZ9.CRXP1A7WOeoJeXxjNni43kdQwgnWNReilDMblYTn_I0' \
  --header 'Content-Type: application/json' \
  --data '{"input":"Tell me a joke"}'
```

## Deploy

1. Create a new project on [Supabase](https://supabase.io)

2. To deploy the supabase functions, first login to Supabase:

```bash
npx supabase login
```

Then, link your project:

```bash
npx supabase link --project-ref <project-ref>
```

Then, deploy the functions:

```bash
yarn supabase:deploy
```

Push the schema to the database:

```bash
supabase db push
```

## Provide your own data

1. Record some audio files and put them in the folder you specified in the `ROOT_DIR` env variable
2. Create a virtual environment and install the dependencies from the `requirements.txt` file
3. Run the `transcribe.py` script to transcribe the audio files to text
4. Run the `transform.py` script to transform the text files to a q and a format
5. Run the `embed.py` script to embed the q and a files into a vector space
