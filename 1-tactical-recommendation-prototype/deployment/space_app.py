import os
os.environ["NEWS_ZEROGPU"] = "1"
os.environ["NEWS_EMBEDDING_DEVICE"] = "cpu"
os.environ["NEWS_HOST"] = "0.0.0.0"
os.environ["PORT"] = "7860"
import spaces
from portfolio_app import launch

launch(7860)
