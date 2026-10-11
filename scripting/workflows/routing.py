"""Converted from ``routing.ipynb``.

Routing architecture: given free-text input, extract the scientist name, then an LLM
makes a routing decision to produce either just history, just achievements
('accomplishment'), or a full article. For the 'article' case, the workflow invokes
a nested subgraph (article_workflow) that runs history and achievements in parallel
and then generates/refines the article.

ASCII flow diagram (actual node names from build_graph):
    START -> scientist_name_extraction -> llm_routing
                                           ├--> generate_history -> END
                                           ├--> generate_achievements -> END
                                           └--> article_workflow -> END
                                                (article_workflow):
                                                    START --┬--> generate_history ----------┐
                                                           └--> generate_achievements -----+--> generate_article -> generate_refined_article -> END

Nodes:
- scientist_name_extraction: Extracts the scientist name from free-text input (RoutingState['input']).
- llm_routing: LLM decides which branch to take (history / accomplishment / article); stores decision.
- generate_history: Produces history (terminal branch when routed directly).
- generate_achievements: Produces achievements (terminal branch when routed directly).
- article_workflow: Nested subgraph (compiled StateGraph) that generates full article via parallel history+achievements then article+refine.

"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .. import nodes
from ..llms import build_models
from ..prompts import ROUTING_PROMPT
from ..schemas import RoutingState

NAME = "routing"
KIND = "routing"
HAS_FEEDBACK = False
HAS_TOOLS = False
INPUT_KIND = "input"
DESCRIPTION = "LLM decides between history, achievements or a full article."


def initial_state(payload: dict) -> dict:
    """Initialize state from payload. Expects 'input' (str). Returns {'input': ...}
    matching RoutingState shape (free-text input to extract scientist name and route)."""
    return {"input": payload.get("input", "")}


def make_llm_routing(models):
    """Factory returning llm_routing node function. Uses structured router chain
    (ROUTING_PROMPT | models.router) on state['input'] to produce decision.step
    stored as state['decision'] (string)."""
    def llm_routing(state: RoutingState) -> dict:
        # ``models.router`` is a structured chain; compose it with the prompt.
        chain = ROUTING_PROMPT | models.router
        decision = chain.invoke({"input": state["input"]})
        # Store the raw step name; ``route_decision`` maps it to a node.
        return {"decision": decision.step}

    return llm_routing


def route_decision(state: RoutingState) -> str:
    """Conditional-edge router mapping LLM decision to next node name.
    Maps: 'history' -> 'generate_history', 'accomplishment' -> 'generate_achievements',
    'article' -> 'article_workflow'. Raises ValueError if unknown."""
    decision = state.get("decision", "")
    if decision == "history":
        return "generate_history"
    if decision == "accomplishment":
        return "generate_achievements"
    if decision == "article":
        return "article_workflow"
    # Any other value means the router hallucinated: fail loudly.
    raise ValueError(f"Invalid decision: {decision}")


def build_article_workflow(models):
    """Build nested subgraph for full article case (runs as 'article_workflow' node).
    Parallel history+achievements from START, converge at generate_article, then refine.
    Compiled subgraph returned."""
    article_workflow = StateGraph(RoutingState)
    # Node roles inside nested subgraph:
    # generate_history: produce history (parallel)
    article_workflow.add_node("generate_history", nodes.make_generate_history(models))
    # generate_achievements: produce achievements (parallel)
    article_workflow.add_node(
        "generate_achievements", nodes.make_generate_achievements(models)
    )
    # generate_article: compose article from both branches
    article_workflow.add_node("generate_article", nodes.make_generate_article(models))
    # generate_refined_article: refine composed article
    article_workflow.add_node(
        "generate_refined_article", nodes.make_generate_refined_article(models)
    )

    # Inner article pipeline: parallel history + achievements -> article -> refine.
    # Fan-out from START inside subgraph.
    article_workflow.add_edge(START, "generate_history")
    article_workflow.add_edge(START, "generate_achievements")
    # Join branches at generate_article.
    article_workflow.add_edge("generate_history", "generate_article")
    article_workflow.add_edge("generate_achievements", "generate_article")
    # article -> refined -> END.
    article_workflow.add_edge("generate_article", "generate_refined_article")
    article_workflow.add_edge("generate_refined_article", END)

    return article_workflow.compile()


def build_graph(models=None):
    models = models or build_models(routing=True)

    workflow = StateGraph(RoutingState)
    # Top-level graph: extract the scientist, let the LLM route, then branch
    # to history / achievements / the embedded article workflow.
    article_workflow = build_article_workflow(models)

    # Node role: Extract scientist name from free-text input (required first step).
    workflow.add_node(
        "scientist_name_extraction", nodes.make_scientist_name_extraction(models)
    )
    # Node role: Generate history (terminal branch when routed).
    workflow.add_node("generate_history", nodes.make_generate_history(models))
    # Node role: Generate achievements (terminal branch when routed).
    workflow.add_node(
        "generate_achievements", nodes.make_generate_achievements(models)
    )
    # Node role: Nested subgraph that produces full article when routed.
    workflow.add_node("article_workflow", article_workflow)
    # Node role: LLM makes routing decision based on input.
    workflow.add_node("llm_routing", make_llm_routing(models))

    # Main flow: extract the scientist first, then let the router decide.
    workflow.add_edge(START, "scientist_name_extraction")
    workflow.add_edge("scientist_name_extraction", "llm_routing")

    # Conditional edges: branch based on routing decision (history/accomplishment/article).
    workflow.add_conditional_edges(
        "llm_routing",
        route_decision,
        {
            "generate_history": "generate_history",
            "generate_achievements": "generate_achievements",
            "article_workflow": "article_workflow",
        },
    )

    workflow.add_edge("generate_history", END)
    workflow.add_edge("generate_achievements", END)
    workflow.add_edge("article_workflow", END)

    return workflow.compile()


def run(payload: dict, models=None, callbacks=None) -> dict:
    # ``routing=True`` makes the router LLM available on the models bundle.
    models = models or build_models(routing=True)
    graph = build_graph(models)
    config = {"callbacks": callbacks} if callbacks else {}
    return graph.invoke(initial_state(payload), config=config)
