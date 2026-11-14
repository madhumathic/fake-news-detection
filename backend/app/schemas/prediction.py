"""Request validation for the prediction endpoint."""
from src.config import MAX_INPUT_CHARS, MIN_WORDS
from src.data.preprocess import clean_text


class ApiError(Exception):
    """An error that is returned to the client as JSON {"error": {"code", "message"}}."""

    def __init__(self, code, message, status=400):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


def validate_prediction_payload(payload):
    """Return the news text from a request body, or raise ApiError."""
    if not isinstance(payload, dict):
        raise ApiError("invalid_json", "Send a JSON body such as {\"news\": \"...\"}.", 400)
    news = payload.get("news", payload.get("text"))
    if news is None:
        raise ApiError("missing_field", "The field 'news' is required.", 400)
    if not isinstance(news, str):
        raise ApiError("invalid_type", "The field 'news' must be text.", 400)
    news = news.strip()
    if not news:
        raise ApiError("empty_input", "Please enter a news article.", 400)
    if len(news) > MAX_INPUT_CHARS:
        raise ApiError("input_too_large",
                       f"The article is too long ({len(news):,} characters). The limit is {MAX_INPUT_CHARS:,}.", 413)
    if len(clean_text(news).split()) < MIN_WORDS:
        raise ApiError("too_short",
                       f"Please enter at least {MIN_WORDS} words of article text (links and HTML are ignored).", 400)
    return news
