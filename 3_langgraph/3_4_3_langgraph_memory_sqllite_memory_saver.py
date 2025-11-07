from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

# --- Step 1: Define stateful node ---
def greet(state):
    # Detect first run
    if state.get("first_run", True):
        # First run: save the user's name
        name = state.get("name", "Guest")
        state["saved_name"] = name
        state["msg"] = f"First run: Hello, {name}! (saved to SQLite memory)"
        state["first_run"] = False  # flip flag
    else:
        # Second run: recall the name from memory
        saved_name = state.get("saved_name", "Unknown")
        state["msg"] = f"Second run: Welcome back, {saved_name}!"
    return state


# --- Step 2: Build graph ---
graph = StateGraph(dict)
graph.add_node("greet", greet)
graph.set_entry_point("greet")
graph.add_edge("greet", END)

# --- Step 3: Use Sqlitecheckpointer properly as context manager ---
session_id = "sqlite_session_1"

with SqliteSaver.from_conn_string("c://code//agenticai//3_langgraph//checkpointer.db") as checkpointer:
    app = graph.compile(checkpointer=checkpointer)

    # --- Step 4: First run ---
    print("\n--- First Run ---")
    result1 = app.invoke(
        {"name": "Alice", "first_run": True},
        config={"configurable": {"thread_id": session_id}},
    )
    print(result1["msg"])

    # --- Step 5: Restore state and run again ---
    print("\n--- Second Run ---")
    checkpoint = checkpointer.get(config={"configurable": {"thread_id": session_id}})
    if checkpoint is None:
        print("No checkpoint found!")
        restored_state = {}
    else:
        checkpoint_data = getattr(checkpoint, "checkpoint", checkpoint)
        channel_values = checkpoint_data.get("channel_values", {})
        restored_state = channel_values.get("__root__", {})

    result2 = app.invoke(
        restored_state,
        config={"configurable": {"thread_id": session_id}},
    )
    print(result2["msg"])
    
     