import asyncio
from tqdm.asyncio import tqdm as tqdm_async
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import cast, Dict
import PIL
import json

from memory.config.settings import get_settings
from memory.core.paths import timestamped_work_dir

from .llm import (
    gpt_4o_mini_complete,
    openai_embedding,
    hf_embedding,
    hf_model_complete
)
from .operate import (
    chunking_by_token_size,
    extract_entities,
    # local_query,global_query,hybrid_query,
    kg_query,
    naive_query,
    naive_workflow_query,
    mix_kg_vector_query,
)

from .runtime import NamespaceConfig, configure_runtime
from .utils import (
    EmbeddingFunc,
    compute_mdhash_id,
    convert_response_to_json,
    logger,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    StorageNameSpace,
    QueryParam,
    DocStatus,
)

from .storage import (
    JsonKVStorage,
    NanoVectorDBStorage,
    NetworkXStorage,
    JsonDocStatusStorage,
)

from .prompt import GRAPH_FIELD_SEP
from .memory_rank import (
    ensure_doc_status,
    load_global_counter,
    rank_candidates,
    reset_global_counter,
    save_global_counter,
    update_doc_access,
)

# future KG integrations

# from .kg.ArangoDB_impl import (
#     GraphStorage as ArangoDBStorage
# )


def lazy_external_import(module_name: str, class_name: str):
    """Lazily import a class from an external module based on the package of the caller."""

    # Get the caller's module and package
    import inspect

    caller_frame = inspect.currentframe().f_back
    module = inspect.getmodule(caller_frame)
    package = module.__package__ if module else None

    def import_class(*args, **kwargs):
        import importlib

        # Import the module using importlib
        module = importlib.import_module(module_name, package=package)

        # Get the class from the module and instantiate it
        cls = getattr(module, class_name)
        return cls(*args, **kwargs)

    return import_class


Neo4JStorage = lazy_external_import(".kg.neo4j_impl", "Neo4JStorage")
OracleKVStorage = lazy_external_import(".kg.oracle_impl", "OracleKVStorage")
OracleGraphStorage = lazy_external_import(".kg.oracle_impl", "OracleGraphStorage")
OracleVectorDBStorage = lazy_external_import(".kg.oracle_impl", "OracleVectorDBStorage")
MilvusVectorDBStorge = lazy_external_import(".kg.milvus_impl", "MilvusVectorDBStorge")
MongoKVStorage = lazy_external_import(".kg.mongo_impl", "MongoKVStorage")
ChromaVectorDBStorage = lazy_external_import(".kg.chroma_impl", "ChromaVectorDBStorage")
TiDBKVStorage = lazy_external_import(".kg.tidb_impl", "TiDBKVStorage")
TiDBVectorDBStorage = lazy_external_import(".kg.tidb_impl", "TiDBVectorDBStorage")
TiDBGraphStorage = lazy_external_import(".kg.tidb_impl", "TiDBGraphStorage")
AGEStorage = lazy_external_import(".kg.age_impl", "AGEStorage")
GremlinStorage = lazy_external_import(".kg.gremlin_impl", "GremlinStorage")


_SETTINGS = get_settings()


"""
Python 库 asyncio 是一个用于编写并发代码的库，主要用于处理异步 I/O 操作。它提供了事件循环、协程、任务和各种用于并发编程的同步原语。

主要功能
- 事件循环： asyncio 提供了一个事件循环，用于调度和执行异步任务。
- 协程： 使用 async def 定义的函数是协程，可以使用 await 关键字等待其他协程的完成。
- 任务： asyncio 提供了 Task 类，用于调度协程的执行。
- 同步原语： 提供了多种同步原语，如 Event、Lock、Semaphore 等，用于在异步代码中进行同步操作。
- 异步 I/O 操作： 提供了多种异步 I/O 操作，如网络通信、文件读写等。
"""

def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    """
    Ensure that there is always an event loop available.

    This function tries to get the current event loop. If the current event loop is closed or does not exist,
    it creates a new event loop and sets it as the current event loop.

    Returns:
        asyncio.AbstractEventLoop: The current or newly created event loop.
    """
    try:
        # Try to get the current event loop
        current_loop = asyncio.get_event_loop()
        if current_loop.is_closed():
            raise RuntimeError("Event loop is closed.")
        return current_loop

    except RuntimeError:
        # If no event loop exists or it is closed, create a new one
        logger.info("Creating a new event loop in main thread.")
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        return new_loop


@dataclass
class WFLightRAG:
    working_dir: str = field(
        default_factory=lambda: str(timestamped_work_dir("workflow_lightrag"))
    )
    # Default not to use embedding cache
    embedding_cache_config: dict = field(
        default_factory=lambda: {
            "enabled": False,
            "similarity_threshold": 0.95,
            "use_llm_check": False,
        }
    )
    kv_storage: str = field(default="JsonKVStorage")
    vector_storage: str = field(default="NanoVectorDBStorage")
    graph_storage: str = field(default="NetworkXStorage")
    threshold: float = _SETTINGS.workflow_defaults.cosine_threshold
    current_log_level = logger.level
    log_level: str = field(default=current_log_level)

    # text chunking
    chunk_token_size: int = _SETTINGS.workflow_defaults.chunk_token_size
    chunk_overlap_token_size: int = _SETTINGS.workflow_defaults.chunk_overlap_token_size
    tiktoken_model_name: str = _SETTINGS.workflow_defaults.tiktoken_model_name

    # entity extraction
    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 500

    # node embedding
    node_embedding_algorithm: str = "node2vec"
    node2vec_params: dict = field(
        default_factory=lambda: {
            "dimensions": 1536,
            "num_walks": 10,
            "walk_length": 40,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        }
    )

    # embedding_func: EmbeddingFunc = field(default_factory=lambda:hf_embedding)
    embedding_func: EmbeddingFunc = field(default_factory=lambda: openai_embedding)
    embedding_batch_num: int = _SETTINGS.workflow_defaults.embedding_batch_num
    embedding_func_max_async: int = _SETTINGS.workflow_defaults.embedding_func_max_async

    # LLM
    llm_model_func: callable = gpt_4o_mini_complete  # hf_model_complete#
    llm_model_name: str = _SETTINGS.workflow_defaults.llm_model_name
    llm_model_max_token_size: int = _SETTINGS.workflow_defaults.llm_max_token_size
    llm_model_max_async: int = _SETTINGS.workflow_defaults.llm_max_async
    llm_model_kwargs: dict = field(default_factory=dict)

    # storage
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)

    enable_llm_cache: bool = True

    # extension
    addon_params: dict = field(default_factory=dict)
    convert_response_to_json_func: callable = convert_response_to_json

    # Add new field for document status storage type
    doc_status_storage: str = field(default="JsonDocStatusStorage")
    inherit_counter: bool = True

    def __post_init__(self):
        base_config = lambda: {"working_dir": self.working_dir}
        vector_config = lambda: {
            "working_dir": self.working_dir,
            "embedding_batch_num": self.embedding_batch_num,
            "cosine_better_than_threshold": self.threshold,
        }

        configure_runtime(
            self,
            log_filename=_SETTINGS.logging.rag_file,
            base_global_config=base_config,
            kv_namespaces=[
                NamespaceConfig(attr_name="full_docs", namespace="full_docs"),
                NamespaceConfig(
                    attr_name="workflow_chunks", namespace="workflow_chunks"
                ),
            ],
            vector_namespaces=[
                NamespaceConfig(
                    attr_name="workflow_chucks_vdb",
                    namespace="workflow_chucks",
                    global_config_factory=vector_config,
                )
            ],
            doc_status_namespace=NamespaceConfig(
                attr_name="doc_status",
                namespace="doc_status",
                use_embedding=False,
            ),
        )

        logger.info(f"Logger initialized for working directory: {self.working_dir}")
        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self).items()])
        logger.debug(f"WFLightRAG init with param:\n  {_print_config}\n")
        if not self.inherit_counter:
            reset_global_counter(self.working_dir)

    def _get_storage_class(self) -> dict:
        return {
            # kv storage
            "JsonKVStorage": JsonKVStorage,
            "OracleKVStorage": OracleKVStorage,
            "MongoKVStorage": MongoKVStorage,
            "TiDBKVStorage": TiDBKVStorage,
            # vector storage
            "NanoVectorDBStorage": NanoVectorDBStorage,
            "OracleVectorDBStorage": OracleVectorDBStorage,
            "MilvusVectorDBStorge": MilvusVectorDBStorge,
            "ChromaVectorDBStorage": ChromaVectorDBStorage,
            "TiDBVectorDBStorage": TiDBVectorDBStorage,
            # graph storage
            "NetworkXStorage": NetworkXStorage,
            "Neo4JStorage": Neo4JStorage,
            "OracleGraphStorage": OracleGraphStorage,
            "AGEStorage": AGEStorage,
            "TiDBGraphStorage": TiDBGraphStorage,
            "GremlinStorage": GremlinStorage,
            # "ArangoDBStorage": ArangoDBStorage
            "JsonDocStatusStorage": JsonDocStatusStorage,
        }

    def insert(self, string_or_strings):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert(string_or_strings))

    async def ainsert(self, string_or_strings):
        """Insert documents with checkpoint support

        Args:
            string_or_strings: Single document string or list of document strings
        """
        if isinstance(string_or_strings, str):
            string_or_strings = [string_or_strings]

        # 1. Remove duplicate contents from the list
        unique_contents = list(set(doc.strip() for doc in string_or_strings))

        # 2. Generate document IDs and initial status
        new_docs = {
            compute_mdhash_id(content, prefix="doc-"): {
                "title": eval(content)['title'],
                "actions": "\n".join(eval(content)['plan']),
                "full_doc_id": compute_mdhash_id(content, prefix="doc-"),
                "status": DocStatus.PENDING,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }
            for content in unique_contents
        }

        # 3. Filter out already processed documents
        _add_doc_keys = await self.doc_status.filter_keys(list(new_docs.keys()))
        new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}

        if not new_docs:
            logger.info("All documents have been processed or are duplicates")
            return

        logger.info(f"Processing {len(new_docs)} new unique documents")

        # Process documents in batches
        batch_size = self.addon_params.get("insert_batch_size", 10)
        for i in range(0, len(new_docs), batch_size):
            batch_docs = dict(list(new_docs.items())[i : i + batch_size])

            for doc_id, doc in tqdm_async(
                batch_docs.items(), desc=f"Processing batch {i//batch_size + 1}"
            ):
                try:
                    # Update status to processing
                    doc_status = {
                        "action_num": len(doc["actions"].split("\n")),
                        "status": DocStatus.PROCESSING,
                        "created_at": doc["created_at"],
                        "updated_at": datetime.now().isoformat(),
                        "metadata": {"last_access": 0, "access_count": 0},
                    }
                    await self.doc_status.upsert({doc_id: doc_status})

                    # Generate chunks from document
                    workflow_chunks = {
                        compute_mdhash_id(dp["content"], prefix="chunk-"): {
                            **dp,
                            "full_doc_id": doc_id,
                        }
                        for dp in chunking_by_token_size(
                            doc["title"],
                            overlap_token_size=self.chunk_overlap_token_size,
                            max_token_size=self.chunk_token_size,
                            tiktoken_model=self.tiktoken_model_name,
                        )
                    }

                    # Update status with chunks information
                    doc_status.update(
                        {
                            "chunks_count": len(workflow_chunks),
                            "updated_at": datetime.now().isoformat(),
                        }
                    )
                    await self.doc_status.upsert({doc_id: doc_status})

                    try:
                        # Store chunks in vector database
                        await self.workflow_chucks_vdb.upsert(workflow_chunks)

                        # Store original document and chunks
                        await self.full_docs.upsert(
                            # {doc_id: {"content": doc["content"]}}
                            new_docs
                        )
                        await self.workflow_chunks.upsert(workflow_chunks)

                        # Update status to processed
                        doc_status.update(
                            {
                                "status": DocStatus.PROCESSED,
                                "updated_at": datetime.now().isoformat(),
                            }
                        )
                        await self.doc_status.upsert({doc_id: doc_status})

                    except Exception as e:
                        # Mark as failed if any step fails
                        doc_status.update(
                            {
                                "status": DocStatus.FAILED,
                                "error": str(e),
                                "updated_at": datetime.now().isoformat(),
                            }
                        )
                        await self.doc_status.upsert({doc_id: doc_status})
                        raise e

                except Exception as e:
                    import traceback

                    error_msg = f"Failed to process document {doc_id}: {str(e)}\n{traceback.format_exc()}"
                    logger.error(error_msg)
                    continue

                finally:
                    # Ensure all indexes are updated after each document
                    await self._insert_done()

    async def _insert_done(self):
        tasks = []
        for storage_inst in [
            self.full_docs,
            self.workflow_chunks,
            self.llm_response_cache,
            self.workflow_chucks_vdb
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    def query(self, task: str, param: QueryParam = QueryParam()):
        loop = always_get_an_event_loop()
        results = loop.run_until_complete(self.aquery(task, self.workflow_chunks, self.workflow_chucks_vdb, param))
        return results

    async def aquery(self, query: str, chunks_db, chunks_vdb, param: QueryParam = QueryParam()):
        if param.mode == "naive":
            response = await naive_workflow_query(
                query,
                chunks_vdb,
                chunks_db,
                self.full_docs,
                param,
                asdict(self),
                hashing_kv=self.llm_response_cache
                if self.llm_response_cache
                and hasattr(self.llm_response_cache, "global_config")
                else self.key_string_value_json_storage_cls(
                    namespace="llm_response_cache",
                    global_config={"working_dir": self.working_dir},
                    embedding_func=None,
                ),
            )
        else:
            raise ValueError(f"Unknown mode {param.mode}")
        if isinstance(response, list) and response:
            doc_ids = [item.get("full_doc_id") or item.get("doc_id") for item in response]
            status_map = await self.doc_status.get_by_ids(doc_ids)
            counter = load_global_counter(self.working_dir)
            ranked = rank_candidates(response, status_map, counter)
            now_iso = datetime.now().isoformat()
            updated_status = {}
            for item in ranked:
                doc_id = item.get("full_doc_id") or item.get("doc_id")
                status = ensure_doc_status(status_map.get(doc_id), now_iso)
                counter += 1
                status = update_doc_access(status, counter, now_iso)
                updated_status[doc_id] = status
            if updated_status:
                await self.doc_status.upsert(updated_status)
            save_global_counter(self.working_dir, counter)
            response = ranked
        await self._query_done()
        return response

    async def _query_done(self):
        tasks = []
        for storage_inst in [self.llm_response_cache]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)


    def _get_content_summary(self, content: str, max_length: int = 100) -> str:
        """Get summary of document content

        Args:
            content: Original document content
            max_length: Maximum length of summary

        Returns:
            Truncated content with ellipsis if needed
        """
        content = content.strip()
        if len(content) <= max_length:
            return content
        return content[:max_length] + "..."

    async def get_processing_status(self) -> Dict[str, int]:
        """Get current document processing status counts

        Returns:
            Dict with counts for each status
        """
        return await self.doc_status.get_status_counts()

    async def adelete_by_doc_id(self, doc_id: str):
        """Delete a document and all its related data

        Args:
            doc_id: Document ID to delete
        """
        try:
            # 1. Get the document status and related data
            doc_status = await self.doc_status.get(doc_id)
            if not doc_status:
                logger.warning(f"Document {doc_id} not found")
                return

            logger.debug(f"Starting deletion for document {doc_id}")

            # 2. Get all related chunks
            chunks = await self.workflow_chunks.filter(
                lambda x: x.get("full_doc_id") == doc_id
            )
            chunk_ids = list(chunks.keys())
            logger.debug(f"Found {len(chunk_ids)} chunks to delete")

            # Continue with the original deletion process...

            # 4. Delete chunks from vector database
            if chunk_ids:
                await self.workflow_chucks_vdb.delete(chunk_ids)
                await self.workflow_chunks.delete(chunk_ids)


            # Track which entities and relationships need to be deleted or updated
            entities_to_delete = set()
            entities_to_update = {}  # entity_name -> new_source_id
            relationships_to_delete = set()
            relationships_to_update = {}  # (src, tgt) -> new_source_id

            # 6. Delete original document and status
            await self.full_docs.delete([doc_id])
            await self.doc_status.delete([doc_id])

            # 7. Ensure all indexes are updated
            await self._insert_done()

            logger.info(
                f"Successfully deleted document {doc_id} and related data. "
                f"Deleted {len(entities_to_delete)} entities and {len(relationships_to_delete)} relationships. "
                f"Updated {len(entities_to_update)} entities and {len(relationships_to_update)} relationships."
            )

            # Add verification step
            async def verify_deletion():
                # Verify if the document has been deleted
                if await self.full_docs.get_by_id(doc_id):
                    logger.error(f"Document {doc_id} still exists in full_docs")

                # Verify if chunks have been deleted
                remaining_chunks = await self.text_chunks.filter(
                    lambda x: x.get("full_doc_id") == doc_id
                )
                if remaining_chunks:
                    logger.error(f"Found {len(remaining_chunks)} remaining chunks")

            await verify_deletion()

        except Exception as e:
            logger.error(f"Error while deleting document {doc_id}: {e}")

    def delete_by_doc_id(self, doc_id: str):
        """Synchronous version of adelete"""
        return asyncio.run(self.adelete_by_doc_id(doc_id))
