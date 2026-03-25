def router(state):
    user_msg = state["messages"][-1].content.lower()

    # Date
    if "date" in user_msg:
        route = "date"

    # Weather
    elif "weather" in user_msg:
        route = "search"

    # Player stats → Search
    elif any(word in user_msg for word in [
        "runs", "most runs", "wickets", "top players",
        "highest scorer", "stats", "statistics"
    ]):
        route = "search"

    # General WC info → RAG
    elif any(word in user_msg for word in [
        "world cup", "wc", "t20", "cricket"
    ]):
        route = "rag"

    # News
    elif any(word in user_msg for word in [
        "latest", "news", "current"
    ]):
        route = "search"

    else:
        route = "llm"

    return {
        "messages": state["messages"],
        "next": route
    }
