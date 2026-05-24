"""
data_agent/__init__.py
----------------------
DataAgent — LLM-powered agent jo data retrieval,
preprocessing, augmentation aur analysis ka plan
samajhta hai aur execute karta hai.
"""

import time
from configs import AVAILABLE_LLMs
from data_agent import retriever
from utils import print_message, get_client

# ──────────────────────────────────────────────
# Agent ka system prompt (persona)
# ──────────────────────────────────────────────
AGENT_PROFILE = """You are the world's best data scientist of an automated \
machine learning project (AutoML) that can find the most relevant datasets, \
run useful preprocessing, perform suitable data augmentation, and make \
meaningful visualization to comprehensively understand the data based on the \
user requirements. You have the following main responsibilities to complete:

1. Retrieve a dataset from the user or search for the dataset based on the user instruction.
2. Perform data preprocessing based on the user instruction or best practice based on the given tasks.
3. Perform data augmentation as necessary.
4. Extract useful information and underlying characteristics of the dataset."""

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
MAX_RETRIES: int = 10
RETRY_DELAY: float = 2.0   # seconds between retries
TEMPERATURE: float = 0.3   # LLM creativity (0 = deterministic, 1 = creative)


class DataAgent:
    """
    DataAgent — data science ka kaam karne wala LLM agent.

    Parameters
    ----------
    user_requirements : dict
        User ne jo kaam manga hai uski details (task type, dataset info, etc.)
    llm : str
        Konsa LLM use karna hai — configs.py mein defined hona chahiye.
        Default: "gpt-4o-mini" (sasta aur fast)
    rap : bool
        Retrieval-Augmented Planning on/off. Default: True
    decomp : bool
        Plan ko breakdown (decompose) karna on/off. Default: True
    """

    def __init__(
        self,
        user_requirements: dict,
        llm: str = "gpt-4o-mini",
        rap: bool = True,
        decomp: bool = True,
    ) -> None:
        self.agent_type = "data"
        self.llm = llm
        self.model: str = AVAILABLE_LLMs[llm]["model"]   # e.g. "gpt-4o-mini"
        self.user_requirements = user_requirements
        self.rap = rap
        self.decomp = decomp
        self.money: dict = {}   # token usage track karne ke liye (cost monitoring)

    # ──────────────────────────────────────────
    # Private helper — LLM call with retry
    # ──────────────────────────────────────────
    def _call_llm(self, messages: list[dict]) -> object:
        """
        LLM ko call karta hai — agar error aaye toh MAX_RETRIES tak retry karta hai.

        Parameters
        ----------
        messages : list[dict]
            Chat history format mein messages.
            e.g. [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]

        Returns
        -------
        response object (OpenAI format)
        """
        client = get_client(self.llm)

        for attempt in range(MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=TEMPERATURE,
                )
                return response
            except Exception as e:
                print_message("system", f"Attempt {attempt + 1} failed: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)

        raise RuntimeError(f"LLM call failed after {MAX_RETRIES} retries.")

    # ──────────────────────────────────────────
    # Step 1: Plan samajhna
    # ──────────────────────────────────────────
    def understand_plan(self, plan: str) -> str:
        """
        Senior manager ka plan leke LLM se data science angle se summarize karta hai.

        Parameters
        ----------
        plan : str
            Agent manager ka original ML pipeline plan.

        Returns
        -------
        str : Data-focused summary of the plan
        """
        summary_prompt = f"""As a proficient data scientist, summarize the following plan \
given by the senior AutoML project manager according to the user's requirements and your \
expertise in data science.

# User's Requirements
```json
{self.user_requirements}
```

# Project Plan
{plan}

The summary of the plan should enable you to fulfill your responsibilities as the answers \
to the following questions by focusing on the data manipulation and analysis:
1. How to retrieve or collect the dataset(s)?
2. How to preprocess the retrieved dataset(s)?
3. How to efficiently augment the dataset(s)?
4. How to extract and understand the underlying characteristics of the dataset(s)?

Note: Do not perform data visualization. Make sure another data scientist can exactly \
reproduce the results based on your summary."""

        messages = [
            {"role": "system", "content": AGENT_PROFILE},
            {"role": "user", "content": summary_prompt},
        ]

        res = self._call_llm(messages)

        data_plan: str = res.choices[0].message.content.strip()

        # Token usage save karo (cost tracking ke liye)
        self.money["Data_Plan_Decomposition"] = res.usage.to_dict(mode="json")

        return data_plan

    # ──────────────────────────────────────────
    # Step 2: Plan execute karna
    # ──────────────────────────────────────────
    def execute_plan(self, plan: str, data_path: str, pid: int) -> str:
        """
        Plan ko execute karta hai:
          1. Plan summarize karta hai (agar decomp=True)
          2. Available datasets dhundta hai
          3. LLM se detailed steps explain karta hai

        Parameters
        ----------
        plan : str
            Agent manager ka original ML pipeline plan.
        data_path : str
            Local dataset folder ka path.
        pid : int
            Pipeline ID (multiple runs track karne ke liye).

        Returns
        -------
        str : Detailed data processing steps
        """
        print_message(self.agent_type, "I am working with the given plan!", pid)

        # Step 1: Plan decompose karo (agar decomp=True)
        if self.decomp:
            data_plan = self.understand_plan(plan)
        else:
            data_plan = plan

        # Step 2: Available datasets dhundho (local path se)
        available_sources = retriever.retrieve_datasets(
            self.user_requirements,
            data_path,
            get_client(self.llm),
            self.model,
        )

        # Step 3: LLM se detailed execution steps maango
        exec_prompt = f"""As a proficient data scientist, your task is to explain **detailed** \
steps for data manipulation and analysis parts by executing the following machine learning \
development plan.

# Plan
{data_plan}

# Potential Source of Dataset
{available_sources}

Make sure that your explanation follows these instructions:
- All explanations must be self-contained without placeholders so other data scientists can \
exactly reproduce all steps — but do not include any code.
- Include how and where to retrieve or collect the data.
- Include how to preprocess the data and which tools or libraries to use.
- Include how to do data augmentation with details and names.
- Include how to extract and understand the characteristics of the data.
- Include reasons why each step is essential to complete the plan effectively.

Note: Do not perform data visualization. Focus only on the data part — do not conduct \
anything related to modeling or training.

After completing the explanations, explicitly specify the (expected) outcomes and results \
both quantitative and qualitative."""

        messages = [
            {"role": "system", "content": AGENT_PROFILE},
            {"role": "user", "content": exec_prompt},
        ]

        res = self._call_llm(messages)

        action_result: str = res.choices[0].message.content.strip()

        # Token usage save karo
        self.money[f"Data_Plan_Execution_{pid}"] = res.usage.to_dict(mode="json")

        print_message(self.agent_type, "I have done with my execution!", pid)
        return action_result