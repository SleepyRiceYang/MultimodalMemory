import asyncio
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import numpy as np
from transformers import AutoModel, AutoTokenizer
from PIL import Image
from typing import TypedDict, Union, Literal, Generic, TypeVar, Optional, Dict, Any, List
from tqdm.asyncio import tqdm_asyncio

from ..lightrag import WFLightRAG, QueryParam
from ..lightrag.utils import EmbeddingFunc, compute_args_hash
from ..lightrag.llm import (
    openai_complete_if_cache, 
    openai_embedding, 
    hf_model_complete,
    hf_embedding,
    initialize_hf_model
)

RetrievedDoc = TypedDict(
    "RetrievedDoc",
    {
        "doc_id": str,
        "chunks": str,
        "content": str, 
        "image": Union[Image.Image,None], 
        "attribute": Union[Dict,str,None], 
        "confidence": float
    }
)


class ProceduralMemory:
    """ 
    Procedural memory that stores UI knowledge.
    """
    def __init__(self, model_type, llm_model, embedding_config, threshold:float=0.5,
                 api_key:str=None, base_url:str=None,
                 model_dir="./model_zoo", rag_dir='./memory/procedural',
                 inherit_counter: bool = True
    ):
        """
        Initialize the ProceduralMemory class with the given parameters.

        Args:
            model_type: Source of the model, either 'hf' for Hugging Face or 'api' for OpenAI.
            llm_model: Language model.
            embedding_config: Configuration for the embedding model.
            api_key: OpenAI API key.
            base_url: OpenAI API base URL.
            model_dir: Directory to store model files.
            rag_dir: Directory to store RAG files.
        """
        self.model_type = model_type
        assert self.model_type in ["hf", "api"], "model_type must be either 'hf' or 'api'"

        self.llm_model = llm_model
        self.embedding_config = embedding_config
        self.embed_dim = embedding_config.embed_dim
        self.max_token_size = embedding_config.max_token_size

        if self.model_type == "openai":
            assert api_key is not None, "api_key must be provided for OpenAI models"
            assert base_url is not None, "base_url must be provided for OpenAI models"
            self.api_key = api_key
            self.base_url = base_url
            self.embedding_model = embedding_config.model_name
        elif self.model_type == "hf":
            self.llm_model = f"{model_dir}/{llm_model}"
            embedding_model_name = f"{model_dir}/{embedding_config.model_name}"
            self.embedding_model, self.embedding_tokenizer, _ = initialize_hf_model(embedding_model_name)

        if not os.path.exists(rag_dir): os.makedirs(rag_dir)
        self.rag = WFLightRAG(
            working_dir=rag_dir,
            llm_model_func=self.llm_model_func,
            llm_model_name=self.llm_model,
            embedding_func=EmbeddingFunc(
                embedding_dim=self.embed_dim,
                max_token_size=self.max_token_size,
                func=self.embedding_func
            ),
            threshold=threshold,
            inherit_counter=inherit_counter,
        )

    async def llm_model_func(
        self, prompt, system_prompt=None, history_messages=[], **kwargs
    ) -> str:
        """
        Generate a response from the language model based on the given prompt.

        Args:
            prompt: The input prompt.
            system_prompt: Optional system prompt.
            history_messages: Optional history messages.
        
        Returns:
            The generated response as a string.
        """
        if self.model_type == "openai":
            return await openai_complete_if_cache(
                self.llm_model,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=self.api_key,
                base_url=self.base_url,
                **kwargs
            )
        elif self.model_type == "hf":
            return await hf_model_complete(
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                keyword_extraction=False,
                **kwargs
            )
        else: raise ValueError(f"Invalid model type: {self.model_type}")
    

    async def embedding_func(self, texts: list[str]) -> np.ndarray:
        """
        Generate embeddings for the given texts.

        Args:
            texts: List of input texts.
        
        Returns:
            Numpy array of embeddings.
        """
        if self.model_type == "openai":
            return await openai_embedding(
                texts,
                model= self.embedding_model,
                api_key= self.api_key,
                base_url= self.base_url
            )
        elif self.model_type == "hf":
            return await hf_embedding(
                texts,
                tokenizer=self.embedding_tokenizer,
                embed_model=self.embedding_model
            )
    

    def retrieve_and_instantiate(self, input, top_n=1) -> RetrievedDoc:
        """
        Retrieve and instantiate documents from the memory based on the input.

        Args:
            input: The input query.
            top_n: Number of top documents to retrieve.
        
        Returns:
            Retrieved documents as a TypedDict.
        """
        documents = self.rag.query(input, param=QueryParam(mode="naive", top_n=top_n, top_k=top_n))
        return documents


    async def dynamic_update(self, action_string):
        """
        Update the memory with new information.

        Args:
            action_string: The action description.
            action_attribute: The action attributes.
        """
        await self.rag.ainsert(action_string)

        # # 1. Retrieve the corresponding document in memory
        # doc = await self.rag.aquery(action_string, param=QueryParam(mode="naive", top_n=1, top_k=1))
        
        # # 2. Dynamic Update the memory
        # if not doc:  # No document found, insert new document
        #     print('No need to Update!')
        #     await self.rag.ainsert(action_string)
        # else:  # Document found, update the document
        #     print('Stationary Memory Updated!')
        #     doc_id = doc[0]['doc_id']
        #     await self.rag.aupdate(doc_id, action_string)
