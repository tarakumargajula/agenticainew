from langgraph.graph import StateGraph, END

# ------------------------------
# Define shared state
# ------------------------------
from typing import TypedDict

class MyState(TypedDict):
    x: int


# ------------------------------
# Define node functions
# ------------------------------
def step1(state: MyState) -> MyState:
    print("Running step1...")
    state["x"] = 10
    return state

def step2(state: MyState) -> MyState:
    print("Running step2...")
    print("Value of x:", state["x"])
    return state


# ------------------------------
# Build the graph
# ------------------------------
workflow = StateGraph(MyState)

workflow.add_node("step1", step1)
workflow.add_node("step2", step2)

workflow.set_entry_point("step1")
workflow.add_edge("step1", "step2")
workflow.add_edge("step2", END)

# Compile the graph
app = workflow.compile()

# ------------------------------
# Run the workflow
# ------------------------------
if __name__ == "__main__":
    result = app.invoke({})
    print("Final state:", result)
