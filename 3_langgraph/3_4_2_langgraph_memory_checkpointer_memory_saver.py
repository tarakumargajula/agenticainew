from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# --- Step 1: Define state ---
def greet(state):
    if state.get("first_run", True):
        name = state.get("name", "Guest")
        state["saved_name"] = name
        state["msg"] = f"First run: Hello, {name}! (saved to memory)"
        state["first_run"] = False
    else:
        saved_name = state.get("saved_name", "Unknown")
        state["msg"] = f"Second run: Welcome back, {saved_name}!"
    return state


# --- Step 2: Create graph ---
graph = StateGraph(dict)
graph.add_node("greet", greet)
graph.set_entry_point("greet")
graph.add_edge("greet", END)

# --- Step 3: Add checkpointer ---
# Attach the memory checkpointing mechanism when compiling the graph
memory = MemorySaver()
app = graph.compile(checkpointer=memory)

# Create a unique thread/session ID to isolate the conversation
session_id = "session_1"

# --- Step 4: First run ---
print("\n--- First Run ---")
# Executes the node for the first time
# The user name (Alice) is stored in memory
result1 = app.invoke(
    {
        "name": "Alice",
        "first_run": True
    },
    config={
        "configurable": {"thread_id": session_id}
    },
)
print(result1["msg"])

# --- Step 5: Second run ---
print("\n--- Second Run ---")

# Fetches saved state from memory for the given thread/session
checkpoint = memory.get(config={"configurable": {"thread_id": session_id}})
if checkpoint is None:
    print("No checkpoint found!")
    restored_state = {}
else:
    # Safely extracts the stored state dictionary from checkpoint data
    checkpoint_data = getattr(checkpoint, "checkpoint", checkpoint)
    channel_values = checkpoint_data.get("channel_values", {})
    # restored_state now holds saved values such as "saved_name": "Alice"
    restored_state = channel_values.get("__root__", {})

# Produces message → Second run: Welcome back, Alice!
result2 = app.invoke(
    restored_state,
    config={
        "configurable": {"thread_id": session_id}
    },
)
print(result2["msg"])

# --- Step 6: Inspect stored memory ---
print("\n--- Memory Checkpoint Data ---")
print(dict(memory.storage))
