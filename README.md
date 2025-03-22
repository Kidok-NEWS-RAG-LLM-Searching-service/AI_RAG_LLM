## AI_RAG_LLM: LLM 기반 뉴스 검색 서비스

A Retrieval-Augmented Generation (RAG) based intelligent QA system designed to search and answer questions from news content using Large Language Models. Built for the Kidok News service.


## Features
- Integrated OpenAI ChatCompletion + LangChain + Pinecone for hybrid retrieval
- Developed prompt-based routing system for dynamic QA flows
- Implemented multiple retrievers (dense-sparse hybrid, time-based, journalist-specific)
- Constructed rich metadata-aware citation system (source formatting, hallucination filtering)
- Used async/await patterns for fast, non-blocking LLM queries
