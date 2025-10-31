"""Integration test for ConditionalStep and FlowAdapter (sub-flows).

This test demonstrates a realistic scenario where:
1. A document processing workflow uses conditional logic to route documents
2. Different processing sub-flows are used based on document type
3. Complex nested workflows are composed using FlowAdapter
"""

import pytest
from pydantic import BaseModel
from typing import Literal

from agent_lib import (
    AgentBlock,
    Flow,
    FlowAdapter,
    ConditionalStep,
    ExecutionContext,
    EventEmitter,
)


# ============================================================================
# Data Models
# ============================================================================


class Document(BaseModel):
    """Input document with content and metadata."""

    content: str
    doc_type: Literal["invoice", "resume", "contract"]
    priority: Literal["high", "normal", "low"] = "normal"


class ProcessedDocument(BaseModel):
    """Processed document with extracted data."""

    content: str
    doc_type: str
    priority: str
    extracted_data: dict
    processing_steps: list[str]


# ============================================================================
# Simple Agent Blocks for Testing
# ============================================================================


class CleanTextAgent(AgentBlock[Document, Document]):
    """Cleans and normalizes document text."""

    async def process(self, input_data: Document) -> Document:
        cleaned_content = input_data.content.strip().lower()
        return Document(
            content=cleaned_content,
            doc_type=input_data.doc_type,
            priority=input_data.priority,
        )


class ExtractInvoiceDataAgent(AgentBlock[Document, ProcessedDocument]):
    """Extracts invoice-specific data."""

    async def process(self, input_data: Document) -> ProcessedDocument:
        # Simulate invoice data extraction
        extracted_data = {
            "invoice_number": "INV-001",
            "amount": "$1,000",
            "vendor": "ACME Corp",
        }
        return ProcessedDocument(
            content=input_data.content,
            doc_type=input_data.doc_type,
            priority=input_data.priority,
            extracted_data=extracted_data,
            processing_steps=["clean_text", "extract_invoice_data"],
        )


class ExtractResumeDataAgent(AgentBlock[Document, ProcessedDocument]):
    """Extracts resume-specific data."""

    async def process(self, input_data: Document) -> ProcessedDocument:
        # Simulate resume data extraction
        extracted_data = {
            "name": "John Doe",
            "email": "john@example.com",
            "skills": ["Python", "AI", "ML"],
        }
        return ProcessedDocument(
            content=input_data.content,
            doc_type=input_data.doc_type,
            priority=input_data.priority,
            extracted_data=extracted_data,
            processing_steps=["clean_text", "extract_resume_data"],
        )


class ExtractContractDataAgent(AgentBlock[Document, ProcessedDocument]):
    """Extracts contract-specific data."""

    async def process(self, input_data: Document) -> ProcessedDocument:
        # Simulate contract data extraction
        extracted_data = {
            "parties": ["Company A", "Company B"],
            "start_date": "2025-01-01",
            "end_date": "2026-01-01",
        }
        return ProcessedDocument(
            content=input_data.content,
            doc_type=input_data.doc_type,
            priority=input_data.priority,
            extracted_data=extracted_data,
            processing_steps=["clean_text", "extract_contract_data"],
        )


class PriorityRouterAgent(AgentBlock[ProcessedDocument, ProcessedDocument]):
    """Adds priority-specific processing metadata."""

    async def process(self, input_data: ProcessedDocument) -> ProcessedDocument:
        # Add priority metadata
        updated_data = input_data.extracted_data.copy()
        updated_data["priority_metadata"] = {
            "queue": "urgent" if input_data.priority == "high" else "standard",
            "sla_hours": 1 if input_data.priority == "high" else 24,
        }

        updated_steps = input_data.processing_steps + ["priority_routing"]

        return ProcessedDocument(
            content=input_data.content,
            doc_type=input_data.doc_type,
            priority=input_data.priority,
            extracted_data=updated_data,
            processing_steps=updated_steps,
        )


class ValidateDataAgent(AgentBlock[ProcessedDocument, ProcessedDocument]):
    """Validates extracted data completeness."""

    async def process(self, input_data: ProcessedDocument) -> ProcessedDocument:
        # Add validation metadata
        updated_data = input_data.extracted_data.copy()
        updated_data["validation"] = {
            "is_valid": True,
            "validated_at": "2025-01-24T12:00:00Z",
        }

        updated_steps = input_data.processing_steps + ["validate_data"]

        return ProcessedDocument(
            content=input_data.content,
            doc_type=input_data.doc_type,
            priority=input_data.priority,
            extracted_data=updated_data,
            processing_steps=updated_steps,
        )


# ============================================================================
# Tests
# ============================================================================


@pytest.fixture
def context():
    """Create execution context for tests."""
    return ExecutionContext()


@pytest.fixture
def emitter():
    """Create event emitter for tests."""
    return EventEmitter()


class TestConditionalAndSubFlow:
    """Test conditional logic and sub-flow execution."""

    @pytest.mark.asyncio
    async def test_invoice_processing_with_conditional_and_subflow(
        self, context, emitter
    ):
        """Test invoice processing uses correct sub-flow via conditional."""
        # Create preprocessing sub-flow (shared by all document types)
        preprocess_flow = Flow("preprocessing", context, emitter)
        preprocess_flow.add_agent(CleanTextAgent("clean_text", context, emitter))

        # Wrap preprocessing as an agent block
        preprocess_step = FlowAdapter(preprocess_flow, name="preprocess_step")

        # Create invoice-specific extraction flow
        invoice_flow = Flow("invoice_extraction", context, emitter)
        invoice_flow.add_agent(
            ExtractInvoiceDataAgent("extract_invoice", context, emitter)
        )

        # Create resume-specific extraction flow
        resume_flow = Flow("resume_extraction", context, emitter)
        resume_flow.add_agent(ExtractResumeDataAgent("extract_resume", context, emitter))

        # Wrap extraction flows as agent blocks
        invoice_extraction = FlowAdapter(invoice_flow, name="invoice_extraction_step")
        resume_extraction = FlowAdapter(resume_flow, name="resume_extraction_step")

        # Create conditional step to route based on document type
        def is_invoice(doc: Document) -> bool:
            return doc.doc_type == "invoice"

        type_router = ConditionalStep(
            name="type_router",
            condition=is_invoice,
            true_agent=invoice_extraction,
            false_agent=resume_extraction,
            context=context,
            emitter=emitter,
        )

        # Create main workflow
        main_flow = Flow("document_processing", context, emitter)
        main_flow.add_agent(preprocess_step)  # Sub-flow!
        main_flow.add_agent(type_router)  # Conditional with nested sub-flows!
        main_flow.add_agent(ValidateDataAgent("validate", context, emitter))

        # Test with invoice
        invoice_doc = Document(
            content="  INVOICE #001 - $1,000  ",
            doc_type="invoice",
            priority="high",
        )

        result = await main_flow.execute_sequential(invoice_doc)

        # Verify invoice was processed correctly
        assert isinstance(result, ProcessedDocument)
        assert result.doc_type == "invoice"
        assert result.extracted_data["invoice_number"] == "INV-001"
        assert result.extracted_data["amount"] == "$1,000"
        assert "validation" in result.extracted_data
        assert result.processing_steps == [
            "clean_text",
            "extract_invoice_data",
            "validate_data",
        ]

    @pytest.mark.asyncio
    async def test_resume_processing_with_conditional_and_subflow(
        self, context, emitter
    ):
        """Test resume processing uses correct sub-flow via conditional."""
        # Create preprocessing sub-flow
        preprocess_flow = Flow("preprocessing", context, emitter)
        preprocess_flow.add_agent(CleanTextAgent("clean_text", context, emitter))
        preprocess_step = FlowAdapter(preprocess_flow)

        # Create extraction flows
        invoice_flow = Flow("invoice_extraction", context, emitter)
        invoice_flow.add_agent(
            ExtractInvoiceDataAgent("extract_invoice", context, emitter)
        )

        resume_flow = Flow("resume_extraction", context, emitter)
        resume_flow.add_agent(ExtractResumeDataAgent("extract_resume", context, emitter))

        invoice_extraction = FlowAdapter(invoice_flow)
        resume_extraction = FlowAdapter(resume_flow)

        # Conditional routing
        def is_invoice(doc: Document) -> bool:
            return doc.doc_type == "invoice"

        type_router = ConditionalStep(
            name="type_router",
            condition=is_invoice,
            true_agent=invoice_extraction,
            false_agent=resume_extraction,
            context=context,
            emitter=emitter,
        )

        # Main workflow
        main_flow = Flow("document_processing", context, emitter)
        main_flow.add_agent(preprocess_step)
        main_flow.add_agent(type_router)
        main_flow.add_agent(ValidateDataAgent("validate", context, emitter))

        # Test with resume
        resume_doc = Document(
            content="  JOHN DOE - PYTHON DEVELOPER  ",
            doc_type="resume",
            priority="normal",
        )

        result = await main_flow.execute_sequential(resume_doc)

        # Verify resume was processed correctly
        assert isinstance(result, ProcessedDocument)
        assert result.doc_type == "resume"
        assert result.extracted_data["name"] == "John Doe"
        assert result.extracted_data["skills"] == ["Python", "AI", "ML"]
        assert "validation" in result.extracted_data
        assert result.processing_steps == [
            "clean_text",
            "extract_resume_data",
            "validate_data",
        ]

    @pytest.mark.asyncio
    async def test_nested_conditionals_with_subflows(self, context, emitter):
        """Test nested conditionals and sub-flows for complex routing."""
        # Create preprocessing sub-flow
        preprocess_flow = Flow("preprocessing", context, emitter)
        preprocess_flow.add_agent(CleanTextAgent("clean_text", context, emitter))
        preprocess_step = FlowAdapter(preprocess_flow)

        # Create extraction flows for all types
        invoice_flow = Flow("invoice_extraction", context, emitter)
        invoice_flow.add_agent(
            ExtractInvoiceDataAgent("extract_invoice", context, emitter)
        )

        resume_flow = Flow("resume_extraction", context, emitter)
        resume_flow.add_agent(ExtractResumeDataAgent("extract_resume", context, emitter))

        contract_flow = Flow("contract_extraction", context, emitter)
        contract_flow.add_agent(
            ExtractContractDataAgent("extract_contract", context, emitter)
        )

        invoice_extraction = FlowAdapter(invoice_flow)
        resume_extraction = FlowAdapter(resume_flow)
        contract_extraction = FlowAdapter(contract_flow)

        # Create nested conditional: invoice vs (resume vs contract)
        def is_invoice(doc: Document) -> bool:
            return doc.doc_type == "invoice"

        def is_resume(doc: Document) -> bool:
            return doc.doc_type == "resume"

        # Inner conditional: resume vs contract
        resume_or_contract = ConditionalStep(
            name="resume_or_contract_router",
            condition=is_resume,
            true_agent=resume_extraction,
            false_agent=contract_extraction,
            context=context,
            emitter=emitter,
        )

        # Outer conditional: invoice vs (resume or contract)
        type_router = ConditionalStep(
            name="type_router",
            condition=is_invoice,
            true_agent=invoice_extraction,
            false_agent=resume_or_contract,  # Nested conditional!
            context=context,
            emitter=emitter,
        )

        # Create priority routing sub-flow
        priority_flow = Flow("priority_routing", context, emitter)
        priority_flow.add_agent(PriorityRouterAgent("priority_router", context, emitter))
        priority_step = FlowAdapter(priority_flow)

        # Main workflow with multiple sub-flows
        main_flow = Flow("document_processing", context, emitter)
        main_flow.add_agent(preprocess_step)  # Sub-flow 1
        main_flow.add_agent(type_router)  # Nested conditionals with sub-flows!
        main_flow.add_agent(priority_step)  # Sub-flow 2
        main_flow.add_agent(ValidateDataAgent("validate", context, emitter))

        # Test with contract (goes through nested conditional path)
        contract_doc = Document(
            content="  AGREEMENT BETWEEN PARTIES  ",
            doc_type="contract",
            priority="high",
        )

        result = await main_flow.execute_sequential(contract_doc)

        # Verify contract was processed through nested conditionals
        assert isinstance(result, ProcessedDocument)
        assert result.doc_type == "contract"
        assert result.extracted_data["parties"] == ["Company A", "Company B"]
        assert result.extracted_data["priority_metadata"]["queue"] == "urgent"
        assert "validation" in result.extracted_data
        assert result.processing_steps == [
            "clean_text",
            "extract_contract_data",
            "priority_routing",
            "validate_data",
        ]

    @pytest.mark.asyncio
    async def test_subflow_context_propagation(self, context, emitter):
        """Test that context and emitter propagate correctly through sub-flows."""
        # Track events to verify emitter propagation
        events_received = []

        async def event_handler(event):
            events_received.append(type(event).__name__)

        emitter.subscribe("start", event_handler)
        emitter.subscribe("completion", event_handler)

        # Create sub-flow
        sub_flow = Flow("sub_flow", context, emitter)
        sub_flow.add_agent(CleanTextAgent("clean", context, emitter))
        sub_flow_step = FlowAdapter(sub_flow)

        # Main flow
        main_flow = Flow("main_flow", context, emitter)
        main_flow.add_agent(sub_flow_step)

        # Execute
        doc = Document(content="Test", doc_type="invoice", priority="normal")
        result = await main_flow.execute_sequential(doc)

        # Verify emitter propagated through sub-flow
        assert "StartEvent" in events_received
        assert "CompletionEvent" in events_received
        assert len(events_received) >= 4  # Main flow + sub-flow events

        # Verify result
        assert isinstance(result, Document)
        assert result.content == "test"  # Cleaned by CleanTextAgent
