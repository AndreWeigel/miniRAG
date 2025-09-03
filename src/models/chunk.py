# src/models/chunk.py
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    section: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    meta: Optional[Dict[str, Any]] = None  # e.g., source filename, span offsets

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Chunk":
        return Chunk(**d)

    # Helper for nice citations in prompts/UI
    def cite(self) -> str:
        page_info = f"p.{self.page_start}-{self.page_end}" if self.page_start else ""
        title = self.section or "Untitled"
        return f"[{self.chunk_id} • {title} • {page_info}]"
