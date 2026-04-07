class DirectHistory:
    """A simple, non-branching conversation history manager."""
    def __init__(self, system_prompt):
        self.history = [{"role": "system", "content": system_prompt}]

    def add_user_message(self, content):
        """Adds a user message to the history."""
        self.history.append({"role": "user", "content": content})

    def add_assistant_message(self, content):
        """Adds an assistant message to the history."""
        self.history.append({"role": "assistant", "content": content})

    def get_current_history(self):
        """Returns the complete conversation history."""
        return self.history 