"""SwitchStep multi-branch routing example.

This example demonstrates how to use SwitchStep for multi-branch conditional logic,
which is cleaner than nesting multiple ConditionalStep blocks.

Use cases:
- Routing based on document type, data category, or user role
- Multi-way branching (more than 2 options)
- Complex categorization logic

Requirements:
    pip install agent-orchestration-lib
"""

import asyncio
from typing import Literal

from pydantic import BaseModel

from agent_lib import AgentBlock, EventEmitter, ExecutionContext, SwitchStep


# Define input/output models
class Document(BaseModel):
    """Document to be processed."""

    doc_id: str
    doc_type: str  # pdf, docx, txt, html, json
    content: str
    priority: str  # low, medium, high


class ProcessedDocument(BaseModel):
    """Result of document processing."""

    doc_id: str
    processor: str
    summary: str
    metadata: dict


# Create specialized processors for each document type
class PDFProcessor(AgentBlock[Document, ProcessedDocument]):
    """Processor for PDF documents."""

    async def process(self, input_data: Document) -> ProcessedDocument:
        return ProcessedDocument(
            doc_id=input_data.doc_id,
            processor="PDF Processor",
            summary=f"Extracted text from PDF: {input_data.content[:50]}...",
            metadata={"pages": 10, "has_images": True},
        )


class DOCXProcessor(AgentBlock[Document, ProcessedDocument]):
    """Processor for DOCX documents."""

    async def process(self, input_data: Document) -> ProcessedDocument:
        return ProcessedDocument(
            doc_id=input_data.doc_id,
            processor="DOCX Processor",
            summary=f"Parsed DOCX document: {input_data.content[:50]}...",
            metadata={"paragraphs": 25, "tables": 3},
        )


class TXTProcessor(AgentBlock[Document, ProcessedDocument]):
    """Processor for plain text documents."""

    async def process(self, input_data: Document) -> ProcessedDocument:
        return ProcessedDocument(
            doc_id=input_data.doc_id,
            processor="TXT Processor",
            summary=f"Processed text: {input_data.content[:50]}...",
            metadata={"lines": 100, "encoding": "utf-8"},
        )


class HTMLProcessor(AgentBlock[Document, ProcessedDocument]):
    """Processor for HTML documents."""

    async def process(self, input_data: Document) -> ProcessedDocument:
        return ProcessedDocument(
            doc_id=input_data.doc_id,
            processor="HTML Processor",
            summary=f"Parsed HTML document: {input_data.content[:50]}...",
            metadata={"links": 15, "images": 5},
        )


class JSONProcessor(AgentBlock[Document, ProcessedDocument]):
    """Processor for JSON documents."""

    async def process(self, input_data: Document) -> ProcessedDocument:
        return ProcessedDocument(
            doc_id=input_data.doc_id,
            processor="JSON Processor",
            summary=f"Parsed JSON data: {input_data.content[:50]}...",
            metadata={"keys": 20, "nested": True},
        )


class UnknownFormatProcessor(AgentBlock[Document, ProcessedDocument]):
    """Fallback processor for unknown formats."""

    async def process(self, input_data: Document) -> ProcessedDocument:
        return ProcessedDocument(
            doc_id=input_data.doc_id,
            processor="Unknown Format Handler",
            summary=f"Attempted best-effort processing: {input_data.content[:50]}...",
            metadata={"format": input_data.doc_type, "confidence": "low"},
        )


async def main() -> None:
    """Run SwitchStep examples."""
    # Create context and emitter
    context = ExecutionContext()
    emitter = EventEmitter()

    print("=" * 60)
    print("Example 1: Simple Document Type Routing")
    print("=" * 60)

    # Create processor agents
    pdf_proc = PDFProcessor("pdf_processor", context, emitter)
    docx_proc = DOCXProcessor("docx_processor", context, emitter)
    txt_proc = TXTProcessor("txt_processor", context, emitter)
    html_proc = HTMLProcessor("html_processor", context, emitter)
    json_proc = JSONProcessor("json_processor", context, emitter)
    unknown_proc = UnknownFormatProcessor("unknown_processor", context, emitter)

    # Create SwitchStep with document type selector
    doc_router = SwitchStep(
        name="document_router",
        selector=lambda doc: doc.doc_type.lower(),  # Return lowercase document type
        cases={
            "pdf": pdf_proc,
            "docx": docx_proc,
            "txt": txt_proc,
            "html": html_proc,
            "json": json_proc,
        },
        default_agent=unknown_proc,  # Handle unknown formats
        context=context,
        emitter=emitter,
    )

    # Process different document types
    documents = [
        Document(doc_id="001", doc_type="pdf", content="PDF content here...", priority="high"),
        Document(doc_id="002", doc_type="docx", content="DOCX content here...", priority="medium"),
        Document(doc_id="003", doc_type="txt", content="Plain text here...", priority="low"),
        Document(doc_id="004", doc_type="html", content="<html>...</html>", priority="medium"),
        Document(
            doc_id="005", doc_type="xml", content="<xml>...</xml>", priority="low"
        ),  # Unknown type
    ]

    for doc in documents:
        result = await doc_router.execute(doc)
        print(f"\n{doc.doc_id} ({doc.doc_type}):")
        print(f"  Processor: {result.processor}")
        print(f"  Summary: {result.summary}")
        print(f"  Metadata: {result.metadata}")

    print("\n" + "=" * 60)
    print("Example 2: Complex Categorization Logic")
    print("=" * 60)

    # Define priority-based processors
    class HighPriorityProcessor(AgentBlock[Document, ProcessedDocument]):
        """Fast-track processor for high priority documents."""

        async def process(self, input_data: Document) -> ProcessedDocument:
            return ProcessedDocument(
                doc_id=input_data.doc_id,
                processor="High Priority Fast Track",
                summary=f"[URGENT] Processed {input_data.doc_type} document immediately",
                metadata={"queue_time_ms": 0, "sla": "1_hour"},
            )

    class LargeSizeProcessor(AgentBlock[Document, ProcessedDocument]):
        """Specialized processor for large documents."""

        async def process(self, input_data: Document) -> ProcessedDocument:
            return ProcessedDocument(
                doc_id=input_data.doc_id,
                processor="Large Document Processor",
                summary=f"Processed large {input_data.doc_type} document in chunks",
                metadata={"chunks": 10, "parallel": True},
            )

    class StandardProcessor(AgentBlock[Document, ProcessedDocument]):
        """Standard processor for regular documents."""

        async def process(self, input_data: Document) -> ProcessedDocument:
            return ProcessedDocument(
                doc_id=input_data.doc_id,
                processor="Standard Processor",
                summary=f"Processed {input_data.doc_type} document normally",
                metadata={"queue_time_ms": 5000, "sla": "24_hours"},
            )

    # Complex selector function
    def categorize_document(doc: Document) -> str:
        """Categorize document based on multiple criteria."""
        # High priority always goes to fast track
        if doc.priority == "high":
            return "fast_track"

        # Large documents get special handling
        if len(doc.content) > 10000:
            return "large_document"

        # Everything else is standard
        return "standard"

    # Create complex router
    complex_router = SwitchStep(
        name="complex_router",
        selector=categorize_document,
        cases={
            "fast_track": HighPriorityProcessor("high_priority", context, emitter),
            "large_document": LargeSizeProcessor("large_doc", context, emitter),
            "standard": StandardProcessor("standard", context, emitter),
        },
        context=context,
        emitter=emitter,
    )

    # Test documents with different characteristics
    test_docs = [
        Document(
            doc_id="101",
            doc_type="pdf",
            content="Short urgent doc",
            priority="high",  # Fast track
        ),
        Document(
            doc_id="102",
            doc_type="txt",
            content="x" * 15000,  # Large document
            priority="medium",
        ),
        Document(
            doc_id="103",
            doc_type="docx",
            content="Normal sized doc",
            priority="low",  # Standard
        ),
    ]

    for doc in test_docs:
        result = await complex_router.execute(doc)
        print(f"\n{doc.doc_id} (priority={doc.priority}, size={len(doc.content)}):")
        print(f"  Routed to: {result.processor}")
        print(f"  Summary: {result.summary}")

    print("\n" + "=" * 60)
    print("Example 3: Comparison with Nested ConditionalStep")
    print("=" * 60)

    print("\nWithout SwitchStep, you'd need nested conditions:")
    print(
        """
    # BAD: Nested conditionals for 3+ branches
    if_step1 = ConditionalStep(
        condition=lambda x: x.type == "A",
        true_agent=agent_a,
        false_agent=if_step2  # Another ConditionalStep!
    )

    if_step2 = ConditionalStep(
        condition=lambda x: x.type == "B",
        true_agent=agent_b,
        false_agent=agent_c
    )
    """
    )

    print("\nWith SwitchStep, it's clean and clear:")
    print(
        """
    # GOOD: Clean switch statement
    switch = SwitchStep(
        selector=lambda x: x.type,
        cases={
            "A": agent_a,
            "B": agent_b,
            "C": agent_c,
            "D": agent_d,  # Easy to add more!
        },
        default_agent=fallback
    )
    """
    )

    print("\n" + "=" * 60)
    print("Example 4: Event Tracking")
    print("=" * 60)

    # Track events
    events = []
    emitter.subscribe("progress", lambda event: events.append(event))

    # Execute and observe events
    test_doc = Document(doc_id="999", doc_type="pdf", content="Test content", priority="medium")
    await doc_router.execute(test_doc)

    print("\nProgress events emitted:")
    for event in events:
        if "data" in event and "case" in event.get("data", {}):
            print(f"  - {event.get('stage')}: case={event['data']['case']}")

    print("\n" + "=" * 60)
    print("Summary: SwitchStep Benefits")
    print("=" * 60)
    print("\n✓ Clean multi-branch routing (3+ options)")
    print("✓ Better than nested ConditionalStep blocks")
    print("✓ Default/fallback handling built-in")
    print("✓ Complex selector logic supported")
    print("✓ Observable via progress events")
    print("✓ Easy to add/remove cases")
    print("\nUse SwitchStep when:")
    print("  - You have 3 or more branches")
    print("  - Routing logic is based on a categorical value")
    print("  - You need a default/fallback case")
    print("  - You want clean, maintainable code")


if __name__ == "__main__":
    asyncio.run(main())
