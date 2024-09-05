from datetime import datetime


def generate_session_id(raw_prompt: str):
    """Generates a unique session ID based on the provided prompt and current time.

    This function takes a raw prompt string as input and creates a session ID
    that combines the following elements:

    - Shortened and sanitized version of the prompt (lowercase, spaces replaced
      with underscores, quotes removed, truncated to 30 characters).
    - Separator ("__").
    - Current time in the format HH_MM_SS (hours with leading zeros).

    Args:
        raw_prompt: The raw prompt string used to generate the session ID.

    Returns:
        A unique session ID string.
    """

    now = datetime.now()
    hours = now.hour
    minutes = now.minute
    seconds = now.second

    short_time_mm_ss = f"{hours:02}_{minutes:02}_{seconds:02}"

    lower_case = raw_prompt.lower()
    no_space = lower_case.replace(" ", "_")
    no_quotes = no_space.replace("'", "")
    shorter = no_quotes[:30]
    with_uuid = shorter + "__" + short_time_mm_ss
    return with_uuid