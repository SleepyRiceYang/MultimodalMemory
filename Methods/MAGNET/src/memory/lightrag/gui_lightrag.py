import asyncio
import json

from tqdm.asyncio import tqdm as tqdm_async
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import cast, Dict, List
from PIL import Image

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
    naive_gui_query,
)

from .runtime import NamespaceConfig, configure_runtime
from .utils import (
    EmbeddingFunc,
    compute_mdhash_id,
    convert_response_to_json,
    logger,
)
from .base import (
    BaseKVStorage,
    BaseVectorStorage,
    StorageNameSpace,
    QueryParam,
    DocStatus,
)

from .storage import (
    JsonKVStorage,
    NanoVectorDBStorage,
    JsonDocStatusStorage,
)

from .lightrag import (
    lazy_external_import,
    always_get_an_event_loop
)
from .gui_prompt import PROMPTS as GUI_PROMPTS
from .memory_rank import (
    ensure_doc_status,
    load_global_counter,
    normalize_image_entries,
    rank_candidates,
    rank_image_entries,
    reset_global_counter,
    save_global_counter,
    update_doc_access,
)


_SETTINGS = get_settings()


"""
@dataclass 是 Python 3.7 引入的一个装饰器，
用于简化类的定义，特别是那些主要用于存储数据的类。
它自动为类生成一些常用的特殊方法，例如 __init__、__repr__、__eq__ 等。
这个自动生成的 __init__ 方法会根据类定义中的字段和默认值进行初始化，
并在最后调用 __post_init__ 方法进行进一步的初始化操作。
"""
@dataclass
class GUILightRAG:
    """ Remove the graph_storage, entity extraction, and node embedding"""
    working_dir: str = field(
        default_factory=lambda: str(timestamped_work_dir("gui_lightrag"))
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
    
    current_log_level = logger.level
    log_level: str = field(default=current_log_level)

    # text chunking
    chunk_token_size: int = _SETTINGS.gui_defaults.chunk_token_size
    chunk_overlap_token_size: int = _SETTINGS.gui_defaults.chunk_overlap_token_size
    tiktoken_model_name: str = _SETTINGS.gui_defaults.tiktoken_model_name

    embedding_func: EmbeddingFunc = field(default_factory=lambda:hf_embedding)
    embedding_batch_num: int = _SETTINGS.gui_defaults.embedding_batch_num
    embedding_func_max_async: int = _SETTINGS.gui_defaults.embedding_func_max_async
    threshold: float = _SETTINGS.gui_defaults.cosine_threshold

    # LLM
    llm_model_func: callable = hf_model_complete  # hf_model_complete#
    llm_model_name: str = _SETTINGS.gui_defaults.llm_model_name
    llm_model_max_token_size: int = _SETTINGS.gui_defaults.llm_max_token_size
    llm_model_max_async: int = _SETTINGS.gui_defaults.llm_max_async
    llm_model_kwargs: dict = field(default_factory=dict)

    # storage
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)

    enable_llm_cache: bool = True

    # extension
    addon_params: dict = field(default_factory=dict)
    convert_response_to_json_func: callable = convert_response_to_json

    # Add new field for document status storage type
    doc_status_storage: str = field(default="JsonDocStatusStorage")

    # Add new field for beta value
    beta: float = 0.9
    prompt_version = "v1"
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
            log_filename=_SETTINGS.logging.gui_file,
            base_global_config=base_config,
            kv_namespaces=[
                NamespaceConfig(attr_name="full_docs", namespace="full_docs"),
                NamespaceConfig(attr_name="text_chunks", namespace="text_chunks"),
            ],
            vector_namespaces=[
                NamespaceConfig(
                    attr_name="chunks_vdb",
                    namespace="chunks",
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
        logger.debug(f"GUILightRAG init with param:\n  {_print_config}\n")
        if not self.inherit_counter:
            reset_global_counter(self.working_dir)

    def _get_storage_class(self) -> dict:
        return {
            # kv storage
            "JsonKVStorage": JsonKVStorage,
            # vector storage
            "NanoVectorDBStorage": NanoVectorDBStorage,
            # status storage
            "JsonDocStatusStorage": JsonDocStatusStorage,
        }
    
    def insert(self, string_or_strings, image_or_images=None, attribute_or_attributes=None, conf_or_confidences=None):
        # 尝试获取当前的事件循环
        # 如果当前事件循环已关闭或不存在，则创建一个新的事件循环并将其设置为当前事件循环。
        loop = always_get_an_event_loop()

        # 调用 ainsert 方法
        return loop.run_until_complete(self.ainsert(
            string_or_strings, image_or_images, attribute_or_attributes, conf_or_confidences))
    
    async def ainsert(self, string_or_strings, image_or_images_path, attribute_or_attributes, conf_or_confidences=None):
        """ Insert documents with checkpoint support

        Args:
            string_or_strings: Single document string or list of document strings
            image_or_images: Single PIL image or list of PIL image tuples
            attribute_or_attributes: Single attribute string/dict or list of attribute strings/dicts
        """
        if isinstance(string_or_strings, str):
            string_or_strings = [string_or_strings]
        
        # if image_or_images is not None:
        #     if isinstance(image_or_images, Image.Image):
        #         image_or_images = [(image_or_images,)]
        #     elif isinstance(image_or_images, tuple):
        #         image_or_images = [image_or_images]
        # else: image_or_images = [None] * len(string_or_strings)

        if image_or_images_path is not None:
            if isinstance(image_or_images_path, str):
                image_or_images_path = [json.dumps([image_or_images_path])]
            elif isinstance(image_or_images_path, list):
                image_or_images_path = [json.dumps(image_or_images_path)]
            elif isinstance(image_or_images_path, tuple):
                image_or_images_path = [json.dumps(list(image_or_images_path))]
        else: image_or_images_path = [None] * len(string_or_strings)

        if attribute_or_attributes is not None:
            if isinstance(attribute_or_attributes, str):
                attribute_or_attributes = [attribute_or_attributes]
            elif isinstance(attribute_or_attributes, dict):
                attribute_or_attributes = [json.dumps(attribute_or_attributes, sort_keys=True)]
        else: attribute_or_attributes = [None] * len(string_or_strings)

        if conf_or_confidences is not None:
            if isinstance(conf_or_confidences, float):
                conf_or_confidences = [conf_or_confidences]
        else: conf_or_confidences = [1.0] * len(string_or_strings)

        assert len(image_or_images_path) == len(string_or_strings), "Length of strings and images must be the same"
        assert len(attribute_or_attributes) == len(string_or_strings), "Length of strings and attributes must be the same"
        assert len(conf_or_confidences) == len(string_or_strings), "Length of strings and confidences must be the same"
        
        # 0. Before add to RAG, make sure the strings and images are not duplicate
        # 1. Remove duplicate contents from the list
        # unique_contents = list(set(doc.strip() for doc in string_or_strings))
        
        # unique_contents = [(doc.strip().lower(), image, attr, conf) for doc, image, attr, conf in \
        #                    zip(string_or_strings, image_or_images, attribute_or_attributes, conf_or_confidences)]
        
        unique_contents = list(set((doc.strip().lower(), image_path, attr, conf) for doc, image_path, attr, conf in \
                           zip(string_or_strings, image_or_images_path, attribute_or_attributes, conf_or_confidences)))
        now_iso = datetime.now().isoformat()
        for (content, image_path, attribute, confidence) in unique_contents:
            image_entries = normalize_image_entries(json.loads(image_path), now_iso)
            item = {
                "content": content,
                "image_path": image_entries,
                "attribute": json.loads(attribute) if attribute is not None else None,
                "confidence": confidence,
                "content_summary": self._get_content_summary(content),
                "content_length": len(content),
                "status": DocStatus.PENDING,
                "created_at": now_iso,
                "updated_at": now_iso,
            }
            print(item)
            print(compute_mdhash_id(content, prefix="doc-"))
            

        # 2. Generate document IDs and initial status
        new_docs = {
            compute_mdhash_id(content, prefix="doc-"): {
                "content": content,
                "image_path": normalize_image_entries(json.loads(image_path), now_iso),
                "attribute": json.loads(attribute) if attribute is not None else None,
                "confidence": confidence,
                "content_summary": self._get_content_summary(content),
                "content_length": len(content),
                "status": DocStatus.PENDING,
                "created_at": now_iso,
                "updated_at": now_iso,
            }
            for (content, image_path, attribute, confidence) in unique_contents
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
                        "content_summary": doc["content_summary"],
                        "content_length": doc["content_length"],
                        "status": DocStatus.PROCESSING,
                        "created_at": doc["created_at"],
                        "updated_at": datetime.now().isoformat(),
                        "metadata": {"last_access": 0, "access_count": 0},
                    }
                    await self.doc_status.upsert({doc_id: doc_status})

                    # Generate chunks from document
                    chunks = {
                        compute_mdhash_id(dp["content"], prefix="chunk-"): {
                            **dp,
                            "full_doc_id": doc_id,
                        } 
                        for dp in chunking_by_token_size(
                            doc["content"],
                            overlap_token_size=self.chunk_overlap_token_size,
                            max_token_size=self.chunk_token_size,
                            tiktoken_model=self.tiktoken_model_name,
                        )
                        # for dp in [{
                        #     "tokens": len(doc["content"]),
                        #     "content": doc["content"],
                        #     "chunk_order_index": 0,
                        # }]
                    }

                    # Update status with chunks information
                    doc_status.update(
                        {
                            "chunks_count": len(chunks),
                            "updated_at": datetime.now().isoformat(),
                        }
                    )
                    await self.doc_status.upsert({doc_id: doc_status})

                    try:
                        # Store chunks in vector database
                        await self.chunks_vdb.upsert(chunks)
                        # Store original chunks in json database
                        await self.text_chunks.upsert(chunks)

                        # Store original document in json database
                        # Ensure to store document with corresponding image
                        await self.full_docs.upsert({
                            doc_id: {
                                "content": doc["content"], 
                                "image_path": doc["image_path"], 
                                "attribute": doc["attribute"],
                                "confidence": doc["confidence"]
                            }
                        })

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

    def _get_content_summary(self, content:str, max_length:int=100) -> str:
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

    async def _insert_done(self):
        tasks = []
        for storage_inst in [
            self.full_docs,
            self.text_chunks,
            self.llm_response_cache,
            self.chunks_vdb,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    # async def multi_aquery(self, queries: List[str], param: QueryParam = QueryParam()):
    #     tasks = [self.aquery(query, param) for query in queries]
    #     return await asyncio.gather(*tasks)

    # def multi_query(self, queries: List[str], param: QueryParam = QueryParam()):
    #     loop = always_get_an_event_loop()
    #     return loop.run_until_complete(self.multi_aquery(queries, param))

    def query(self, query:str, param:QueryParam=QueryParam()):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, param))
    
    async def aquery(self, query:str, param:QueryParam=QueryParam()):
        if param.mode == "naive":
            response = await naive_gui_query(
                query,
                self.full_docs,
                self.chunks_vdb,
                self.text_chunks,
                param,
                {"chunk_token_size":self.chunk_token_size, "chunk_overlap_token_size":self.chunk_overlap_token_size},
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
            doc_ids = [item.get("doc_id") or item.get("full_doc_id") for item in response]
            status_map = await self.doc_status.get_by_ids(doc_ids)
            counter = load_global_counter(self.working_dir)
            ranked = rank_candidates(response, status_map, counter)
            now_iso = datetime.now().isoformat()
            updated_status = {}
            updated_docs = {}
            for item in ranked:
                doc_id = item.get("doc_id") or item.get("full_doc_id")
                status = ensure_doc_status(status_map.get(doc_id), now_iso)
                counter += 1
                status = update_doc_access(status, counter, now_iso)
                updated_status[doc_id] = status

                full_doc = await self.full_docs.get_by_id(doc_id)
                if full_doc:
                    image_entries = normalize_image_entries(full_doc.get("image_path"), now_iso)
                    if image_entries:
                        ranked_images = rank_image_entries(image_entries, counter)
                        top_entry = ranked_images[0]
                        top_entry["access_count"] = int(top_entry.get("access_count", 0)) + 1
                        top_entry["last_access"] = counter
                        full_doc["image_path"] = ranked_images
                        updated_docs[doc_id] = full_doc
                        item["image"] = top_entry.get("path")
            if updated_status:
                await self.doc_status.upsert(updated_status)
            if updated_docs:
                await self.full_docs.upsert(updated_docs)
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
    
    async def get_processing_status(self) -> Dict[str, int]:
        """Get current document processing status counts

        Returns:
            Dict with counts for each status
        """
        return await self.doc_status.get_status_counts()

    async def adelete_by_doc_id(self, doc_id:str):
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
            chunks = await self.text_chunks.filter(
                lambda x: x.get("full_doc_id") == doc_id
            )
            chunk_ids = list(chunks.keys())
            logger.debug(f"Found {len(chunk_ids)} chunks to delete")

            # 3. Delete chunks from vector database
            if chunk_ids:
                await self.chunks_vdb.delete(chunk_ids)
                await self.text_chunks.delete(chunk_ids)

            # 4. Delete original document and status
            await self.full_docs.delete([doc_id])
            await self.doc_status.delete([doc_id])

            # 5. Ensure all indexes are updated
            await self._insert_done()

            logger.info(
                f"Successfully deleted document {doc_id} and related data. "
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

    def delete_by_doc_id(self, doc_id:str):
        """Synchronous version of adelete"""
        return asyncio.run(self.adelete_by_doc_id(doc_id))

    def update(self, doc_id:str, content:str, image:Image.Image=None, attribute:dict=None):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aupdate(doc_id, content, image, attribute))
    
    async def aupdate(self, doc_id, content, image, attribute):
        """Update a document with new content

        Args:
            doc_id: Document ID to update
            content: New content for the document
            image: New image for the document
            attribute: New attribute for the document
            reward: Reward value for the document, used to update confidence
        """
        try:
        # if True:
            # 1. Get the document status and related data
            doc_status = await self.doc_status.get(doc_id)
            if not doc_status:
                logger.warning(f"Document {doc_id} not found")
                await self.ainsert(content, image, attribute, 1.0)
            
            # 2. Get the previous document content
            prev_doc = await self.full_docs.get_by_id(doc_id)
            now_iso = datetime.now().isoformat()
            image_entries = normalize_image_entries(prev_doc.get("image_path"), now_iso)
            if image:
                image_entries.append(
                    {
                        "path": image,
                        "created_at": now_iso,
                        "last_access": 0,
                        "access_count": 0,
                    }
                )
            new_image = image_entries
            # new_image = (image,)
            
            # 3. Use LLM to merge the new document with the previous document
            doc_1_string = json.dumps({"function": prev_doc["content"], "attributes": prev_doc["attribute"]})
            doc_2_string = json.dumps({"function": content, "attributes": attribute})
            merge_doc_prompt = GUI_PROMPTS[f"merge_document_{self.prompt_version}"].format(
                examples="\n".join(GUI_PROMPTS[f"merge_document_examples_{self.prompt_version}"]), 
                first_ui_element=doc_1_string, 
                second_ui_element=doc_2_string
            )
            # result = await self.llm_model_func(merge_doc_prompt)
            
            # result = json.loads(result)
            # new_content, new_attribute = result["function"], result["attributes"]

            # 4. Use LLM to generate the reward for the new document
            reward_prompt = GUI_PROMPTS[f"judge_document_{self.prompt_version}"].format(
                examples="\n".join(GUI_PROMPTS[f"judge_document_examples_{self.prompt_version}"]), 
                first_ui_element=doc_1_string, 
                second_ui_element=doc_2_string
            )
            # result = await self.llm_model_func(reward_prompt)
            # reward = 1 if "True" in result else 0

            # # 5. Update the confidence score
            # new_confidence = prev_doc["confidence"] * (1-self.beta) + reward * self.beta

            # 6. Delete the previous document and insert the new document
            await self.adelete_by_doc_id(doc_id)
            # await self.ainsert(new_content, new_image, new_attribute, new_confidence)
            await self.ainsert(prev_doc["content"], new_image, prev_doc["attribute"], prev_doc["confidence"])

        except Exception as e:
            logger.error(f"Error while updating document {doc_id}: {e}")
