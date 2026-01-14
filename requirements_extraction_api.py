# requirements_extraction_api.py
"""
Production Requirements Extraction System
Treats LLM as unreliable component with strict validation
"""

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, validator, ValidationError
from typing import List, Optional, Dict, Any, Literal
from enum import Enum
import anthropic
import json
import re
import time
from datetime import datetime
import hashlib


# ============================================================================
# SCHEMA DEFINITIONS (Hard Contract)
# ============================================================================

class RequirementType(str, Enum):
    FUNCTIONAL = "functional"
    NON_FUNCTIONAL = "non-functional"
    CONSTRAINT = "constraint"
    ASSUMPTION = "assumption"


class Requirement(BaseModel):
    """Enforced output schema - violations = system failure"""
    id: str = Field(..., pattern=r"^(FR|NFR|CON|ASM)-\d{3}$")
    type: RequirementType
    description: str = Field(..., min_length=10, max_length=500)
    confidence: float = Field(..., ge=0.0, le=1.0)
    source_reference: str = Field(..., min_length=5)
    category: Optional[str] = None

    @validator('description')
    def no_hallucination_phrases(cls, v):
        """Detect common hallucination patterns"""
        forbidden = [
            "it seems", "probably", "might be", "could be",
            "appears to", "suggests that", "implies"
        ]
        v_lower = v.lower()
        for phrase in forbidden:
            if phrase in v_lower:
                raise ValueError(f"Forbidden inference phrase detected: {phrase}")
        return v


class Warning(BaseModel):
    type: str
    requirement_id: Optional[str]
    message: str


class ExtractionMetadata(BaseModel):
    input_tokens: int
    output_tokens: int
    total_cost: float
    processing_time_ms: int
    prompt_version: str
    model: str
    retry_count: int
    validation_status: Literal["passed", "failed", "partial"]


class ExtractionResult(BaseModel):
    requirements: List[Requirement]
    warnings: List[Warning]
    metadata: ExtractionMetadata
    flags: Dict[str, Any]


class ExtractionRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=50000)
    strict_mode: bool = True


# ============================================================================
# PROMPT VERSIONING
# ============================================================================

class PromptVersion:
    """Version-controlled prompts with replay capability"""

    VERSIONS = {
        "v1.0.0": {
            "created": "2024-01-01",
            "deprecated": True,
            "reason": "Too permissive, high hallucination rate"
        },
        "v2.0.0": {
            "created": "2024-01-15",
            "deprecated": True,
            "reason": "Citation format inconsistent"
        },
        "v2.3.1": {
            "created": "2024-02-01",
            "deprecated": False,
            "reason": None
        }
    }

    CURRENT = "v2.3.1"

    @staticmethod
    def get_prompt(version: str = CURRENT, strict: bool = True) -> str:
        """Get versioned prompt template"""

        base_prompt = """You are a requirements extraction system. Your job is to parse unstructured text and extract ONLY explicitly stated requirements.

CRITICAL RULES:
1. Extract ONLY what is directly stated in the input text
2. Do NOT infer, guess, or elaborate
3. If something is ambiguous, flag it with low confidence
4. Every requirement MUST cite exact source text
5. Use "Not supported by input" if no evidence exists

Input text will be messy:
- Incomplete sentences
- Contradictions
- Opinions mixed with facts
- Noise (greetings, signatures)

Your output MUST be valid JSON matching this schema:
{
  "requirements": [
    {
      "id": "FR-001",  // FR=functional, NFR=non-functional, CON=constraint, ASM=assumption
      "type": "functional|non-functional|constraint|assumption",
      "description": "Clear, specific requirement statement",
      "confidence": 0.85,  // 0.0-1.0, <0.6 = needs review
      "source_reference": "Exact text snippet from input",
      "category": "optional category tag"
    }
  ],
  "warnings": [
    {
      "type": "contradiction|vague|unsupported",
      "requirement_id": "FR-001 or null",
      "message": "Specific issue description"
    }
  ]
}

CONFIDENCE SCORING:
- 0.9-1.0: Explicitly stated with clear language
- 0.7-0.89: Clearly stated but some interpretation needed
- 0.6-0.69: Implied but reasonable interpretation
- <0.6: Speculative or unsupported (FLAG FOR REVIEW)

WHAT TO FLAG:
- Contradictory statements
- Vague language ("good", "fast", "easy")
- Assumptions presented as facts
- Requirements without clear acceptance criteria
- Opinions without supporting evidence

DO NOT:
- Add requirements not in the input
- Resolve contradictions yourself
- Make technical assumptions
- Use phrases like "seems to", "might be", "probably"
"""

        if strict:
            base_prompt += """

STRICT MODE ACTIVE:
- Confidence threshold raised to 0.7
- Any ambiguity = automatic warning
- Zero tolerance for unsupported claims
- Must explicitly state "Not supported by input" when evidence missing
"""

        return base_prompt


# ============================================================================
# PREPROCESSING PIPELINE
# ============================================================================

class InputPreprocessor:
    """Clean and prepare input text"""

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough token estimation"""
        return len(text.split()) * 1.3

    @staticmethod
    def remove_noise(text: str) -> str:
        """Strip common noise patterns"""
        # Remove email signatures
        text = re.sub(r'--+\s*\n.*', '', text, flags=re.DOTALL)

        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)

        # Remove common greetings/closings
        noise_patterns = [
            r'^(Hi|Hello|Hey|Dear)\s+\w+,?\s*\n',
            r'(Thanks|Regards|Best|Cheers),?\s*\n\w+\s*$'
        ]
        for pattern in noise_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)

        return text.strip()

    @staticmethod
    def validate_input(text: str) -> tuple[bool, Optional[str]]:
        """Check if input is processable"""
        if len(text) < 10:
            return False, "Input too short (min 10 chars)"

        tokens = InputPreprocessor.estimate_tokens(text)
        if tokens > 50000:
            return False, f"Input too large ({tokens} tokens, max 50000)"

        # Check for actual content
        if len(text.split()) < 5:
            return False, "Input lacks meaningful content"

        return True, None


# ============================================================================
# LLM EXECUTION WITH RETRIES
# ============================================================================

class LLMExecutor:
    """Execute LLM calls with retry logic"""

    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-sonnet-4-20250514"

    def extract_with_retries(
            self,
            text: str,
            strict: bool = True,
            max_retries: int = 3
    ) -> tuple[dict, int]:
        """Execute extraction with progressive retry strategy"""

        prompt_template = PromptVersion.get_prompt(strict=strict)

        for attempt in range(max_retries):
            try:
                # Escalate strictness on retries
                current_strict = strict or (attempt > 0)
                prompt = PromptVersion.get_prompt(strict=current_strict)

                user_message = f"{prompt}\n\nINPUT TEXT:\n{text}\n\nExtract requirements as JSON:"

                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4000,
                    temperature=0.0,  # Deterministic
                    messages=[{"role": "user", "content": user_message}]
                )

                # Extract JSON from response
                content = response.content[0].text

                # Try to parse JSON
                # Handle potential markdown code blocks
                if "```json" in content:
                    json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
                    if json_match:
                        content = json_match.group(1)
                elif "```" in content:
                    json_match = re.search(r'```\s*(\{.*?\})\s*```', content, re.DOTALL)
                    if json_match:
                        content = json_match.group(1)

                result = json.loads(content)

                # Basic structure validation
                if "requirements" not in result:
                    raise ValueError("Missing 'requirements' key")

                return result, attempt

            except (json.JSONDecodeError, ValueError) as e:
                if attempt == max_retries - 1:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"LLM output validation failed after {max_retries} attempts: {str(e)}"
                    )
                continue

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Extraction failed after all retries"
        )


# ============================================================================
# VALIDATION LAYER
# ============================================================================

class OutputValidator:
    """Validate and score LLM output"""

    @staticmethod
    def verify_citations(requirements: List[Dict], source_text: str) -> List[Warning]:
        """Verify that citations actually exist in source"""
        warnings = []

        for req in requirements:
            source_ref = req.get("source_reference", "")

            # Check if citation exists in source
            if len(source_ref) > 10:  # Meaningful citation
                # Fuzzy match (allow for minor variations)
                if not OutputValidator._fuzzy_match(source_ref, source_text):
                    warnings.append(Warning(
                        type="invalid_citation",
                        requirement_id=req.get("id"),
                        message=f"Citation not found in source text: {source_ref[:50]}..."
                    ))

        return warnings

    @staticmethod
    def _fuzzy_match(citation: str, source: str, threshold: float = 0.8) -> bool:
        """Check if citation appears in source with some tolerance"""
        citation_clean = re.sub(r'\s+', ' ', citation.lower().strip())
        source_clean = re.sub(r'\s+', ' ', source.lower().strip())

        # Direct substring match
        if citation_clean in source_clean:
            return True

        # Word overlap check
        citation_words = set(citation_clean.split())
        source_words = set(source_clean.split())

        if len(citation_words) == 0:
            return False

        overlap = len(citation_words & source_words) / len(citation_words)
        return overlap >= threshold

    @staticmethod
    def validate_requirements(raw_data: dict, source_text: str) -> tuple[List[Requirement], List[Warning]]:
        """Validate requirements against schema"""
        requirements = []
        warnings = []

        for req_data in raw_data.get("requirements", []):
            try:
                req = Requirement(**req_data)
                requirements.append(req)

                # Flag low confidence
                if req.confidence < 0.6:
                    warnings.append(Warning(
                        type="low_confidence",
                        requirement_id=req.id,
                        message=f"Confidence {req.confidence:.2f} below threshold - needs human review"
                    ))
            except ValidationError as e:
                warnings.append(Warning(
                    type="schema_violation",
                    requirement_id=req_data.get("id"),
                    message=f"Validation failed: {str(e)}"
                ))

        # Add warnings from LLM output
        for warn_data in raw_data.get("warnings", []):
            warnings.append(Warning(**warn_data))

        # Verify citations
        citation_warnings = OutputValidator.verify_citations(
            [req.dict() for req in requirements],
            source_text
        )
        warnings.extend(citation_warnings)

        return requirements, warnings


# ============================================================================
# API SERVICE
# ============================================================================

app = FastAPI(title="Requirements Extraction API", version="2.3.1")

# In production, use environment variables
ANTHROPIC_API_KEY = "your-api-key-here"  # Replace with actual key
executor = LLMExecutor(ANTHROPIC_API_KEY)


@app.post("/extract-requirements", response_model=ExtractionResult)
async def extract_requirements(request: ExtractionRequest):
    """
    Extract structured requirements from unstructured text.

    Returns validated requirements with confidence scores and warnings.
    Failures are explicit, not best-effort.
    """
    start_time = time.time()

    # Step 1: Input validation
    preprocessor = InputPreprocessor()
    valid, error_msg = preprocessor.validate_input(request.text)

    if not valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg
        )

    # Step 2: Preprocessing
    cleaned_text = preprocessor.remove_noise(request.text)
    token_estimate = preprocessor.estimate_tokens(cleaned_text)

    # Step 3: LLM execution with retries
    try:
        raw_output, retry_count = executor.extract_with_retries(
            cleaned_text,
            strict=request.strict_mode
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Extraction failed: {str(e)}"
        )

    # Step 4: Validation
    validator = OutputValidator()
    requirements, warnings = validator.validate_requirements(raw_output, cleaned_text)

    # Step 5: Compute metadata
    processing_time = int((time.time() - start_time) * 1000)
    output_tokens = len(json.dumps([req.dict() for req in requirements])) // 4

    cost_per_1k = 0.003  # Claude Sonnet pricing
    total_cost = ((token_estimate + output_tokens) / 1000) * cost_per_1k

    validation_status = "passed"
    if any(w.type == "schema_violation" for w in warnings):
        validation_status = "failed"
    elif warnings:
        validation_status = "partial"

    # Step 6: Flag for human review
    needs_review = (
            len([r for r in requirements if r.confidence < 0.6]) > 0 or
            len(warnings) > 2 or
            any(w.type in ["contradiction", "invalid_citation"] for w in warnings)
    )

    metadata = ExtractionMetadata(
        input_tokens=int(token_estimate),
        output_tokens=output_tokens,
        total_cost=total_cost,
        processing_time_ms=processing_time,
        prompt_version=PromptVersion.CURRENT,
        model=executor.model,
        retry_count=retry_count,
        validation_status=validation_status
    )

    flags = {
        "needs_human_review": needs_review,
        "high_confidence_count": len([r for r in requirements if r.confidence >= 0.8]),
        "low_confidence_count": len([r for r in requirements if r.confidence < 0.6]),
        "schema_violations": len([w for w in warnings if w.type == "schema_violation"])
    }

    return ExtractionResult(
        requirements=requirements,
        warnings=warnings,
        metadata=metadata,
        flags=flags
    )


@app.get("/health")
async def health_check():
    """System health check"""
    return {
        "status": "operational",
        "prompt_version": PromptVersion.CURRENT,
        "model": executor.model,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/prompt-versions")
async def list_prompt_versions():
    """List all prompt versions with metadata"""
    return {
        "current": PromptVersion.CURRENT,
        "versions": PromptVersion.VERSIONS
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)