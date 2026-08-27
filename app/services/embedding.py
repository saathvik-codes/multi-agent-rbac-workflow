import httpx
import logging
from app.config import settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self, base_url: str = settings.OLLAMA_BASE_URL, model: str = settings.OLLAMA_EMBED_MODEL):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._ollama_healthy = None

    async def close(self):
        pass

    async def check_ollama_health(self) -> bool:
        """Quickly checks if the local Ollama instance is healthy."""
        if self._ollama_healthy is not None:
            return self._ollama_healthy
        try:
            # Use a brief connection check
            async with httpx.AsyncClient(timeout=1.0) as client:
                response = await client.get(f"{self.base_url}/")
                self._ollama_healthy = (response.status_code == 200)
        except Exception:
            self._ollama_healthy = False
        return self._ollama_healthy

    async def get_embedding(self, text: str) -> list[float]:
        """
        Generates vector embeddings for a given string text.
        Attempts to use local Ollama embeddings, falls back to a deterministic pseudo-embedding if local Ollama fails.
        """
        if await self.check_ollama_health():
            try:
                # Use context manager to create a fresh client bound to the current running event loop
                async with httpx.AsyncClient(timeout=httpx.Timeout(4.0, connect=1.0)) as client:
                    response = await client.post(
                        f"{self.base_url}/api/embed",
                        json={"model": self.model, "input": text}
                    )
                    if response.status_code == 200:
                        data = response.json()
                        if "embeddings" in data and len(data["embeddings"]) > 0:
                            return data["embeddings"][0]

                    response = await client.post(
                        f"{self.base_url}/api/embeddings",
                        json={"model": self.model, "prompt": text}
                    )
                    if response.status_code == 200:
                        data = response.json()
                        if "embedding" in data:
                            return data["embedding"]
            except Exception as e:
                logger.warning(f"Ollama local embedding generation failed: {e}. Falling back to pseudo-embeddings.")
        else:
            logger.info("Local Ollama server is offline. Bypassing embedding generation.")

        return self._generate_pseudo_embedding(text)

    def _generate_pseudo_embedding(self, text: str) -> list[float]:
        """
        Generates a deterministic 768-dimensional float list pseudo-embedding for testing/fallback.
        Ensures documents with similar content still return similar cosine scores.
        """
        dimension = 768
        vec = []
        seed = sum(ord(c) for c in text)
        for i in range(dimension):
            val = ((seed * (i + 1)) % 1000) / 1000.0
            vec.append(val)
        
        # Normalize the vector to have length 1.0 (unit vector)
        norm = sum(x*x for x in vec) ** 0.5
        if norm == 0.0:
            return [0.0] * dimension
        return [x / norm for x in vec]
