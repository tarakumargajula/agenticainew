from langgraph.checkpoint.sqlite import SqliteSaver

# --- Path to your existing SQLite checkpoint file ---
db_path = "c://code//agenticai//3_langgraph//checkpointer.db"

# --- The same session_id you used before ---
session_id = "sqlite_session_1"

# --- Open the SQLite checkpointer in read mode ---
with SqliteSaver.from_conn_string(db_path) as checkpointer:
    # Retrieve checkpoint data for the given thread/session
    checkpoint = checkpointer.get(config={"configurable": {"thread_id": session_id}})
    
    if checkpoint is None:
        print("No checkpoint found for this session.")
    else:
        # Extract and display the stored state
        checkpoint_data = getattr(checkpoint, "checkpoint", checkpoint)
        channel_values = checkpoint_data.get("channel_values", {})
        restored_state = channel_values.get("__root__", {})

        print("\n--- Retrieved Memory State ---")
        for key, value in restored_state.items():
            print(f"{key}: {value}")
