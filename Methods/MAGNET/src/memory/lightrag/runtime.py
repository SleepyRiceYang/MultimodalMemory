from __future__ import annotations

import os
from dataclasses import dataclass
from functools import partial
from typing import Callable, Iterable, Optional

from memory.core.paths import log_path

from .utils import limit_async_func_call, logger, set_logger


@dataclass
class NamespaceConfig:
    attr_name: str
    namespace: str
    global_config_factory: Optional[Callable[[], dict]] = None
    meta_fields: Optional[set[str]] = None
    use_embedding: bool = True


def configure_runtime(
    owner,
    *,
    log_filename: str,
    base_global_config: Callable[[], dict],
    kv_namespaces: Iterable[NamespaceConfig] = (),
    vector_namespaces: Iterable[NamespaceConfig] = (),
    doc_status_namespace: NamespaceConfig,
    graph_namespace: Optional[NamespaceConfig] = None,
):
    """Shared runtime bootstrapper for LightRAG implementations."""

    log_file = log_path(log_filename)
    set_logger(str(log_file))
    logger.setLevel(owner.log_level)

    if not os.path.exists(owner.working_dir):
        os.makedirs(owner.working_dir, exist_ok=True)

    storage_classes = owner._get_storage_class()
    owner.key_string_value_json_storage_cls = storage_classes[owner.kv_storage]
    owner.vector_db_storage_cls = storage_classes[owner.vector_storage]
    if graph_namespace is not None:
        owner.graph_storage_cls = storage_classes[owner.graph_storage]

    owner.llm_response_cache = owner.key_string_value_json_storage_cls(
        namespace="llm_response_cache",
        global_config=base_global_config(),
        embedding_func=None,
    )

    owner.embedding_func = limit_async_func_call(owner.embedding_func_max_async)(
        owner.embedding_func
    )

    for namespace in kv_namespaces:
        config = namespace.global_config_factory() if namespace.global_config_factory else base_global_config()
        setattr(
            owner,
            namespace.attr_name,
            owner.key_string_value_json_storage_cls(
                namespace=namespace.namespace,
                global_config=config,
                embedding_func=owner.embedding_func if namespace.use_embedding else None,
            ),
        )

    for namespace in vector_namespaces:
        config = namespace.global_config_factory() if namespace.global_config_factory else base_global_config()
        setattr(
            owner,
            namespace.attr_name,
            owner.vector_db_storage_cls(
                namespace=namespace.namespace,
                global_config=config,
                embedding_func=owner.embedding_func,
                meta_fields=namespace.meta_fields or set(),
            ),
        )

    if graph_namespace is not None:
        config = graph_namespace.global_config_factory() if graph_namespace.global_config_factory else base_global_config()
        setattr(
            owner,
            graph_namespace.attr_name,
            owner.graph_storage_cls(
                namespace=graph_namespace.namespace,
                global_config=config,
                embedding_func=owner.embedding_func,
            ),
        )

    owner.llm_model_func = limit_async_func_call(owner.llm_model_max_async)(
        partial(
            owner.llm_model_func,
            hashing_kv=owner.llm_response_cache,
            **owner.llm_model_kwargs,
        )
    )

    owner.doc_status_storage_cls = storage_classes[owner.doc_status_storage]
    config = doc_status_namespace.global_config_factory() if doc_status_namespace.global_config_factory else base_global_config()
    owner.doc_status = owner.doc_status_storage_cls(
        namespace=doc_status_namespace.namespace,
        global_config=config,
        embedding_func=None,
    )
