"""
data_agent/retriever.py
-----------------------
4 sources se knowledge retrieve karta hai:
  1. Google Web Search
  2. arXiv Research Papers
  3. Kaggle Notebooks
  4. PapersWithCode Datasets & Benchmarks

Phir sab ko LLM se summarize karke ek combined
knowledge chunk return karta hai — jo planning mein
kaam aata hai.
"""

import json
import random
import time
from pathlib import Path

import arxivloader
from openai import OpenAI
from validators import url
from langchain.schema import Document
from langchain_community.document_loaders import AsyncHtmlLoader, PDFMinerLoader
from langchain_community.document_transformers import BeautifulSoupTransformer

from configs import AVAILABLE_LLMs
from utils import search_web, print_message, get_kaggle
from utils.embeddings import chunk_and_retrieve

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
MAX_RETRIES: int = 10
RETRY_DELAY: float = 2.0
TEMPERATURE_SEARCH: float = 0.3
TEMPERATURE_QUERY: float = 0.1
TOP_K_DEFAULT: int = 10

# Web domains jo skip karni hain (irrelevant results)
DOMAIN_BLOCKLIST: list[str] = [
    "youtube.com",
    "twitter.com",
    "x.com",
    "hindawi.com",
    "ejournal.ittelkom-pwt.ac.id",
]

# PapersWithCode ka local data folder
PWC_DATA_PATH = "_data/paperswithcode/"


# ──────────────────────────────────────────────
# Private Helper — LLM call with retry
# ──────────────────────────────────────────────
def _call_llm(
    client: OpenAI,
    llm_model: str,
    messages: list[dict],
    temperature: float = TEMPERATURE_SEARCH,
) -> str:
    """
    LLM ko call karta hai aur MAX_RETRIES tak retry karta hai.

    Parameters
    ----------
    client : OpenAI — initialized OpenAI client
    llm_model : str  — model name e.g. "gpt-4o-mini"
    messages : list[dict] — chat history format
    temperature : float  — creativity (0=deterministic)

    Returns
    -------
    str : LLM ka response text
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=llm_model,
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print_message("system", f"Attempt {attempt + 1} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)

    raise RuntimeError(f"LLM call failed after {MAX_RETRIES} retries.")


# ──────────────────────────────────────────────
# Helper — user requirements se keywords nikalna
# ──────────────────────────────────────────────
def _clean_keyword(text: str) -> str:
    """
    Keyword ko clean karta hai — hyphens/underscores hatata hai,
    lowercase karta hai.

    e.g. "image-classification" → "image classification"
    """
    return text.replace("-", " ").replace("_", " ").strip().lower()


# ──────────────────────────────────────────────
# Source 1: Kaggle Notebooks
# ──────────────────────────────────────────────
def retrieve_kaggle(
    user_requirements: dict,
    user_requirement_summary: str,
    llm_model: str,
    client: OpenAI,
    top_k: int = TOP_K_DEFAULT,
) -> str:
    """
    Kaggle notebooks search karta hai user ke task ke hisaab se.
    Relevant notebooks ko LLM se summarize karta hai.

    Parameters
    ----------
    user_requirements : dict — full user requirements
    user_requirement_summary : str — short summary of requirements
    llm_model : str — model name
    client : OpenAI — LLM client
    top_k : int — kitne notebooks retrieve karne hain

    Returns
    -------
    str : Kaggle notebooks ka summarized knowledge
    """
    kaggle_api = get_kaggle()
    print_message("manager", "Searching relevant Kaggle notebooks...")

    user_task = _clean_keyword(user_requirements["problem"]["downstream_task"])
    user_domain = _clean_keyword(user_requirements["problem"]["application_domain"])

    # Kaggle API se notebooks search karo
    notebooks_raw = kaggle_api.kernels_list_with_http_info(
        search=f"{user_task} {user_domain}",
        sort_by="relevance",
        language="Python",
        page_size=top_k,
    )[0]

    # Har notebook ka content extract karo
    documents: list[Document] = []
    for nb in notebooks_raw:
        try:
            notebook = kaggle_api.kernel_pull(*nb["ref"].split("/"))

            # Agar notebook string nahi hai (valid response) toh process karo
            if isinstance(notebook, str):
                continue

            cells = json.loads(notebook["blob"]["source"])["cells"]

            # Markdown cells seedhe lelo, code cells ko ```python``` mein wrap karo
            page_content = "".join(
                cell["source"]
                if cell["cell_type"] == "markdown"
                else f"\n```python\n{cell['source']}\n```"
                for cell in cells
            )

            documents.append(
                Document(
                    page_content=page_content,
                    metadata=notebook["metadata"],
                )
            )
        except Exception as e:
            print_message("system", f"Skipping notebook: {e}")
            continue

    # BM25 se most relevant chunks retrieve karo
    context = "".join(
        d.page_content
        for d in chunk_and_retrieve(
            ref_text=user_requirement_summary,
            documents=documents,
            top_k=top_k,
            ranker="bm25",
        )
    )

    # LLM se summarize karo
    summary_prompt = f"""I searched Kaggle Notebooks using keywords: {user_task} {user_domain}.
Here is the result:
=====================
{context}
=====================

Please summarize the given pieces of Python notebooks into a single paragraph of useful
knowledge and insights. Do not include source codes — extract insights from them instead.
We aim to use your summary to address the following user's requirements.

# User's Requirements
{user_requirement_summary}
"""
    return _call_llm(
        client=client,
        llm_model=llm_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are tasked to summarize and extract the contents from Kaggle Notebooks "
                    "to provide insightful results for addressing user's requirements. "
                    "Pay attention to state-of-the-art models and their sources."
                ),
            },
            {"role": "user", "content": summary_prompt},
        ],
    )


# ──────────────────────────────────────────────
# Source 2: PapersWithCode
# ──────────────────────────────────────────────
def retrieve_paperswithcode(
    user_requirements: dict,
    user_requirement_summary: str,
    llm_model: str,
    client: OpenAI,
    top_k: int = TOP_K_DEFAULT,
) -> str:
    """
    PapersWithCode ke local JSON files se relevant datasets aur
    benchmark tables retrieve karta hai.

    Parameters
    ----------
    (same as retrieve_kaggle)

    Returns
    -------
    str : PapersWithCode ka summarized knowledge
    """
    print_message("manager", "Searching PapersWithCode...")

    user_task = _clean_keyword(user_requirements["problem"]["downstream_task"])
    user_area = _clean_keyword(user_requirements["problem"]["area"])

    # ── Datasets load karo ──────────────────────
    all_datasets: list[dict] = json.loads(
        Path(f"{PWC_DATA_PATH}/datasets.json").read_text(encoding="utf-8")
    )

    # Sirf woh datasets rakhho jinke paas data_loaders hain
    # aur jo user ke task/area se match karte hain
    dataset_docs: list[Document] = [
        Document(
            page_content=f"""
DATASET NAME: {ds['name']}
DESCRIPTION: {ds['description']}
APPLICABLE TASKS: {','.join(task['task'] for task in ds['tasks'])}
DATA LOADERS: {ds['data_loaders'][:3]}
            """,
            metadata={
                "homepage": ds["homepage"],
                "paper": ds["paper"],
                "variants": ds["variants"],
                "modalities": ds["modalities"],
                "introduced_date": ds["introduced_date"],
            },
        )
        for ds in all_datasets
        if len(ds["data_loaders"]) > 0
        and (
            user_task in ds["description"].lower()
            or user_task in [t["task"].lower() for t in ds["tasks"]]
            or user_area in ds["description"].lower()
        )
    ]

    # ── Benchmark tables load karo ──────────────
    all_tables: list[dict] = json.loads(
        Path(f"{PWC_DATA_PATH}/evaluation-tables.json").read_text(encoding="utf-8")
    )

    benchmark_docs: list[Document] = [
        Document(
            page_content=str(table["datasets"]),
            metadata={
                "categories": table["categories"],
                "subtasks": table["subtasks"],
                "task": table["task"],
                "description": table["description"],
            },
        )
        for table in all_tables
        if len(table["datasets"]) > 0
        and (
            user_task in table["description"].lower()
            or user_task == table["task"].lower()
            or user_area in [cat.lower() for cat in table["categories"]]
            or user_area in table["description"].lower()
        )
    ]

    # Top-k random sample leke BM25 se retrieve karo
    benchmark_sample = random.sample(
        benchmark_docs, k=min(top_k, len(benchmark_docs))
    )
    dataset_sample = random.sample(
        dataset_docs, k=min(top_k, len(dataset_docs))
    )

    context = "".join(
        d.page_content
        for d in chunk_and_retrieve(
            ref_text=user_requirement_summary,
            documents=benchmark_sample + dataset_sample,
            top_k=top_k,
            ranker="bm25",
        )
    )

    summary_prompt = f"""I searched PapersWithCode using keywords: {user_area} and {user_task}.
Here is the result:
=====================
{context}
=====================

Please summarize the given pieces of search content into a single paragraph of useful
knowledge and insights. We aim to use your summary to address the following user's requirements.

# User's Requirements
{user_requirement_summary}
"""
    return _call_llm(
        client=client,
        llm_model=llm_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are tasked to summarize the contents from PapersWithCode "
                    "to provide insightful results for addressing user's requirements. "
                    "Pay attention to state-of-the-art models and their sources."
                ),
            },
            {"role": "user", "content": summary_prompt},
        ],
    )


# ──────────────────────────────────────────────
# Source 3: arXiv Research Papers
# ──────────────────────────────────────────────
def retrieve_arxiv(
    user_requirements: dict,
    user_requirement_summary: str,
    llm_model: str,
    client: OpenAI,
    top_k: int = TOP_K_DEFAULT,
) -> str:
    """
    arXiv se relevant research papers PDF retrieve karta hai
    aur LLM se summarize karta hai.

    Returns
    -------
    str : arXiv papers ka summarized knowledge
    """
    print_message("manager", "Searching arXiv...")

    task_kw = _clean_keyword(user_requirements["problem"]["downstream_task"])
    domain_kw = _clean_keyword(user_requirements["problem"]["application_domain"])

    # arXiv search query — CS categories mein dhundho
    query = (
        f'search_query=all:"{task_kw}" AND all:"{domain_kw}" '
        f"AND (cat:cs.AI OR cat:cs.CV OR cat:cs.LG OR cat:cs.DB)"
    )

    df = arxivloader.load(query, num=top_k, sortBy="submittedDate", verbosity=0)
    arxiv_links = [link.split(";")[-1].strip() for link in df["links"].tolist()]

    # Har paper ka PDF load karo
    documents: list[Document] = []
    for link in arxiv_links:
        try:
            documents += PDFMinerLoader(link).load()
        except Exception as e:
            print_message("system", f"Cannot load {link}: {e}")

    context = "".join(
        d.page_content
        for d in chunk_and_retrieve(
            ref_text=user_requirement_summary,
            documents=documents,
            top_k=top_k,
            ranker="bm25",
        )
    )

    summary_prompt = f"""I searched arXiv papers using keywords: {task_kw} and {domain_kw}.
Here is the result:
=====================
{context}
=====================

Please summarize the given pieces of arXiv papers into a single paragraph of useful
knowledge and insights. We aim to use your summary to address the following user's requirements.

# User's Requirements
{user_requirement_summary}
"""
    return _call_llm(
        client=client,
        llm_model=llm_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are tasked to summarize the contents from relevant arXiv papers "
                    "to provide insightful results for addressing user's requirements."
                ),
            },
            {"role": "user", "content": summary_prompt},
        ],
    )


# ──────────────────────────────────────────────
# Source 4: Google Web Search
# ──────────────────────────────────────────────
def retrieve_websearch(
    user_requirement_summary: str,
    llm_model: str,
    client: OpenAI,
    top_k: int = TOP_K_DEFAULT,
) -> str:
    """
    Google search se relevant web pages retrieve karta hai.
    Pehle LLM se search query generate karta hai,
    phir pages scrape karke summarize karta hai.

    Returns
    -------
    str : Web search ka summarized knowledge
    """
    # Step 1: LLM se search query generate karo
    search_query = _call_llm(
        client=client,
        llm_model=llm_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are tasked with generating web search queries based on the given "
                    "machine learning problem. Give a specific query for Google search "
                    "focusing on the downstream tasks. "
                    "Give a single sentence within 10 words only."
                ),
            },
            {"role": "user", "content": user_requirement_summary},
        ],
        temperature=TEMPERATURE_QUERY,
    )
    search_query = search_query.replace('"', "").replace(".", "")

    print_message("manager", f"Searching Google with query: {search_query}")

    # Step 2: Web search results laao aur blocked domains filter karo
    search_results = [
        result
        for result in search_web(search_query)
        if not any(domain in result["link"] for domain in DOMAIN_BLOCKLIST)
    ][:top_k]

    urls = [r["link"] for r in search_results if url(r["link"])]

    # Step 3: HTML pages load karo (PDFs alag se)
    html_urls = [u for u in urls if ".pdf" not in u]
    loader = AsyncHtmlLoader(html_urls)
    html_docs: list[Document] = loader.load()

    # BeautifulSoup se relevant tags extract karo
    bs_transformer = BeautifulSoupTransformer()
    html_docs = bs_transformer.transform_documents(
        html_docs,
        tags_to_extract=["p", "li", "div", "span", "table"],
    )

    # PDF links bhi load karo
    for u in urls:
        if "arxiv.org/pdf" in u or "/pdf?id=" in u or "&name=pdf" in u:
            try:
                html_docs += PDFMinerLoader(u).load()
            except Exception as e:
                print_message("system", f"Cannot load PDF {u}: {e}")

    # Step 4: BM25 se relevant chunks nikalo
    context = "".join(
        d.page_content
        for d in chunk_and_retrieve(
            ref_text=user_requirement_summary,
            documents=html_docs,
            top_k=top_k,
            ranker="bm25",
        )
    )

    summary_prompt = f"""I searched the web using the query: {search_query}.
Here is the result:
=====================
{context}
=====================

Please summarize the given pieces of search content into a single paragraph of useful
knowledge and insights. We aim to use your summary to address the following user's requirements.

# User's Requirements
{user_requirement_summary}
"""
    return _call_llm(
        client=client,
        llm_model=llm_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are tasked to summarize the contents from Google search "
                    "to provide insightful results for addressing user's requirements."
                ),
            },
            {"role": "user", "content": summary_prompt},
        ],
    )


# ──────────────────────────────────────────────
# Main Function — sabko combine karo
# ──────────────────────────────────────────────
def retrieve_knowledge(
    user_requirements: dict,
    user_requirement_summary: str,
    llm: str,
    adversarial_injection: str | None = None,
) -> str | tuple[str, str]:
    """
    4 sources se knowledge retrieve karke combine karta hai:
    Google, arXiv, Kaggle, PapersWithCode.

    Parameters
    ----------
    user_requirements : dict
        Full user requirements dictionary.
    user_requirement_summary : str
        Short summary of what user wants.
    llm : str
        LLM key from configs.py (e.g. "gpt-4o-mini").
    adversarial_injection : str | None
        Research/testing feature — noise inject karna:
        - "pre"  → noise pehle add karo (before summary)
        - "post" → noise baad mein return karo (after summary)
        - None   → normal mode (use karo yahi)

    Returns
    -------
    str : Combined knowledge summary (normal mode)
    tuple[str, str] : (summary, noise) agar adversarial_injection="post"
    """
    llm_model = AVAILABLE_LLMs[llm]["model"]

    # LLM client initialize karo
    if llm.startswith("gpt"):
        client = OpenAI(api_key=AVAILABLE_LLMs[llm]["api_key"])
    else:
        client = OpenAI(
            base_url=AVAILABLE_LLMs[llm]["base_url"],
            api_key=AVAILABLE_LLMs[llm]["api_key"],
        )

    # ── Optional: Adversarial noise generate karo ──
    # (sirf research/testing ke liye — normal use mein None hi rakho)
    noise: str | None = None
    if adversarial_injection:
        noise = _call_llm(
            client=client,
            llm_model=llm_model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Based on the user's ML task requirements below, generate a chunk of "
                        f"**irrelevant or unhelpful information** that does not aid in solving the task.\n\n"
                        f"User's requirements:\n{user_requirement_summary}"
                    ),
                }
            ],
        )

    # ── 4 sources se retrieve karo ──────────────
    print_message("manager", "Starting knowledge retrieval from all sources...")

    web_summary = retrieve_websearch(user_requirement_summary, llm_model, client)
    arxiv_summary = retrieve_arxiv(user_requirements, user_requirement_summary, llm_model, client)
    pwc_summary = retrieve_paperswithcode(user_requirements, user_requirement_summary, llm_model, client)
    kaggle_summary = retrieve_kaggle(user_requirements, user_requirement_summary, llm_model, client)

    # ── Sab summaries ko combine karo ───────────
    sources_block = f"""
# Source: Google Web Search
{web_summary}
=====================

# Source: arXiv Papers
{arxiv_summary}
=====================

# Source: Kaggle Hub
{kaggle_summary}
=====================

# Source: PapersWithCode
{pwc_summary}
=====================
"""

    # Adversarial noise "pre" mode mein — pehle inject karo
    if adversarial_injection == "pre" and noise:
        sources_block += f"""
# Source: AI Agent (Adversarial)
{noise}
=====================
"""

    final_prompt = f"""Please extract and summarize the following group of contents collected
from different online sources into a chunk of insightful knowledge.
Format your answer as a list of suggestions.
I will use them to address the user's requirements in machine learning tasks.

{sources_block}

The user's requirements are summarized as follows:
{user_requirement_summary}
"""

    final_summary = _call_llm(
        client=client,
        llm_model=llm_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior consultant and professor in machine learning (ML) "
                    "and artificial intelligence (AI), with extensive experience in ML/AI research."
                ),
            },
            {"role": "user", "content": final_prompt},
        ],
    )

    # "post" mode mein — noise alag return karo
    if adversarial_injection == "post":
        return final_summary, noise

    return final_summary


# ──────────────────────────────────────────────
# Alias — data_agent/__init__.py se call hota hai
# ──────────────────────────────────────────────
def retrieve_datasets(
    user_requirements: dict,
    data_path: str,
    client: OpenAI,
    llm_model: str,
) -> str:
    """
    DataAgent ke execute_plan() se call hota hai.
    Yahan se retrieve_knowledge() call hoti hai.

    Parameters
    ----------
    user_requirements : dict — user ki requirements
    data_path : str — local dataset folder path
    client : OpenAI — LLM client
    llm_model : str — model name

    Returns
    -------
    str : Available dataset sources ki summary
    """
    # Local data path ko user requirements mein add karo
    user_requirements_with_path = {
        **user_requirements,
        "local_data_path": data_path,
    }

    # Simple summary banao requirements se
    user_requirement_summary = (
        f"Task: {user_requirements.get('problem', {}).get('downstream_task', 'ML task')}, "
        f"Domain: {user_requirements.get('problem', {}).get('application_domain', 'general')}"
    )

    return retrieve_knowledge(
        user_requirements=user_requirements_with_path,
        user_requirement_summary=user_requirement_summary,
        llm="gpt-4o-mini",  # default LLM — configs se override kar sakte ho
    )
# /jskacnkj