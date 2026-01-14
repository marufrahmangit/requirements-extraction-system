# evaluation_suite.py
"""
Evaluation framework for requirements extraction system.
Compares naive vs guardrailed approaches.
"""

from typing import List, Dict, Set
from dataclasses import dataclass
import json


# ============================================================================
# GOLD DATASET (Manually Curated)
# ============================================================================

@dataclass
class GoldRequirement:
    """Ground truth requirement"""
    text: str
    type: str
    should_extract: bool
    expected_confidence_range: tuple  # (min, max)
    notes: str


GOLD_DATASET = [
    {
        "id": "test_001",
        "name": "Clean functional requirement",
        "input": "The system must allow users to reset their password via email.",
        "expected": [
            GoldRequirement(
                text="The system must allow users to reset their password via email",
                type="functional",
                should_extract=True,
                expected_confidence_range=(0.85, 1.0),
                notes="Clear, explicit functional requirement"
            )
        ],
        "should_warn": False
    },
    {
        "id": "test_002",
        "name": "Vague non-functional requirement",
        "input": "The app should be fast and responsive.",
        "expected": [
            GoldRequirement(
                text="The app should be fast and responsive",
                type="non-functional",
                should_extract=True,
                expected_confidence_range=(0.3, 0.6),
                notes="Vague - should trigger warning for quantification"
            )
        ],
        "should_warn": True
    },
    {
        "id": "test_003",
        "name": "Contradiction",
        "input": "Users must provide email for login. However, social login should not require email.",
        "expected": [
            GoldRequirement(
                text="Users must provide email for login",
                type="functional",
                should_extract=True,
                expected_confidence_range=(0.6, 0.8),
                notes="Valid but contradicted"
            ),
            GoldRequirement(
                text="Social login should not require email",
                type="functional",
                should_extract=True,
                expected_confidence_range=(0.6, 0.8),
                notes="Valid but contradicts first requirement"
            )
        ],
        "should_warn": True
    },
    {
        "id": "test_004",
        "name": "Assumption disguised as fact",
        "input": "Since users are technical, we can skip the tutorial.",
        "expected": [
            GoldRequirement(
                text="users are technical",
                type="assumption",
                should_extract=True,
                expected_confidence_range=(0.5, 0.7),
                notes="Assumption, not a requirement"
            )
        ],
        "should_warn": True
    },
    {
        "id": "test_005",
        "name": "Noise with embedded requirement",
        "input": """Hey team,

        Hope everyone had a great weekend!

        Quick reminder: the login page needs to support 2FA.

        Talk soon,
        Mike""",
        "expected": [
            GoldRequirement(
                text="the login page needs to support 2FA",
                type="functional",
                should_extract=True,
                expected_confidence_range=(0.8, 1.0),
                notes="Clear requirement buried in noise"
            )
        ],
        "should_warn": False
    },
    {
        "id": "test_006",
        "name": "Hallucination test - no actual requirement",
        "input": "We should probably think about security at some point.",
        "expected": [],
        "should_warn": True
    },
    {
        "id": "test_007",
        "name": "Constraint with compliance reference",
        "input": "The system must not store credit card numbers in plaintext (PCI-DSS requirement).",
        "expected": [
            GoldRequirement(
                text="The system must not store credit card numbers in plaintext",
                type="constraint",
                should_extract=True,
                expected_confidence_range=(0.9, 1.0),
                notes="Clear constraint with regulatory context"
            )
        ],
        "should_warn": False
    },
    {
        "id": "test_008",
        "name": "Opinion vs requirement",
        "input": "I think dark mode would be nice. The UI should look modern.",
        "expected": [
            GoldRequirement(
                text="I think dark mode would be nice",
                type="assumption",
                should_extract=False,  # Opinion, not requirement
                expected_confidence_range=(0.0, 0.5),
                notes="Opinion - should be rejected or flagged"
            )
        ],
        "should_warn": True
    },
    {
        "id": "test_009",
        "name": "Incomplete specification",
        "input": "Users need to be able to export data in various formats.",
        "expected": [
            GoldRequirement(
                text="Users need to be able to export data",
                type="functional",
                should_extract=True,
                expected_confidence_range=(0.5, 0.7),
                notes="Valid but underspecified - what formats?"
            )
        ],
        "should_warn": True
    },
    {
        "id": "test_010",
        "name": "Multiple types in one input",
        "input": """The system must support OAuth 2.0 for authentication. 
        Response time should be under 200ms. 
        We're assuming AWS infrastructure.""",
        "expected": [
            GoldRequirement(
                text="The system must support OAuth 2.0 for authentication",
                type="functional",
                should_extract=True,
                expected_confidence_range=(0.85, 1.0),
                notes="Clear functional requirement"
            ),
            GoldRequirement(
                text="Response time should be under 200ms",
                type="non-functional",
                should_extract=True,
                expected_confidence_range=(0.8, 1.0),
                notes="Quantified non-functional requirement"
            ),
            GoldRequirement(
                text="AWS infrastructure",
                type="assumption",
                should_extract=True,
                expected_confidence_range=(0.6, 0.8),
                notes="Infrastructure assumption"
            )
        ],
        "should_warn": False
    }
]


# ============================================================================
# EVALUATION METRICS
# ============================================================================

class EvaluationMetrics:
    """Calculate precision, recall, and custom metrics"""

    @staticmethod
    def precision_of_extracted(extracted: List[Dict], expected: List[GoldRequirement]) -> float:
        """
        Precision: Of all requirements extracted, how many were valid?

        Formula: True Positives / (True Positives + False Positives)
        """
        if len(extracted) == 0:
            return 0.0

        true_positives = 0
        for req in extracted:
            # Check if this extraction matches any expected requirement
            for gold in expected:
                if gold.should_extract and EvaluationMetrics._fuzzy_match(
                        req.get("description", ""),
                        gold.text
                ):
                    true_positives += 1
                    break

        return true_positives / len(extracted)

    @staticmethod
    def recall_of_expected(extracted: List[Dict], expected: List[GoldRequirement]) -> float:
        """
        Recall: Of all requirements that should be extracted, how many were found?

        Formula: True Positives / (True Positives + False Negatives)
        """
        should_extract = [g for g in expected if g.should_extract]
        if len(should_extract) == 0:
            return 1.0

        found = 0
        for gold in should_extract:
            # Check if this expected requirement was found
            for req in extracted:
                if EvaluationMetrics._fuzzy_match(
                        req.get("description", ""),
                        gold.text
                ):
                    found += 1
                    break

        return found / len(should_extract)

    @staticmethod
    def unsupported_claims_rate(extracted: List[Dict], input_text: str) -> float:
        """
        Rate of extracted requirements with no evidence in source.
        This measures hallucination.

        Lower is better (0.0 = no hallucinations)
        """
        if len(extracted) == 0:
            return 0.0

        unsupported = 0
        for req in extracted:
            source_ref = req.get("source_reference", "")
            if len(source_ref) < 5:
                unsupported += 1
                continue

            # Check if source reference exists in input
            if not EvaluationMetrics._text_contains(input_text, source_ref):
                unsupported += 1

        return unsupported / len(extracted)

    @staticmethod
    def schema_violation_rate(results: Dict) -> float:
        """
        Rate of outputs that violated the schema.

        Lower is better (0.0 = all outputs valid)
        """
        warnings = results.get("warnings", [])
        schema_violations = len([w for w in warnings if w.get("type") == "schema_violation"])

        total_requirements = len(results.get("requirements", []))
        if total_requirements == 0:
            return 1.0 if schema_violations > 0 else 0.0

        return schema_violations / total_requirements

    @staticmethod
    def confidence_calibration_error(extracted: List[Dict], expected: List[GoldRequirement]) -> float:
        """
        Measure how well confidence scores match actual correctness.

        Lower is better (0.0 = perfect calibration)
        """
        errors = []

        for req in extracted:
            confidence = req.get("confidence", 0.5)
            description = req.get("description", "")

            # Find matching gold requirement
            is_correct = False
            for gold in expected:
                if gold.should_extract and EvaluationMetrics._fuzzy_match(description, gold.text):
                    is_correct = True
                    break

            # Error is difference between confidence and actual correctness
            actual_correctness = 1.0 if is_correct else 0.0
            errors.append(abs(confidence - actual_correctness))

        return sum(errors) / len(errors) if errors else 0.0

    @staticmethod
    def _fuzzy_match(text1: str, text2: str, threshold: float = 0.6) -> bool:
        """Check if two texts are similar enough"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if len(words1) == 0 or len(words2) == 0:
            return False

        overlap = len(words1 & words2)
        union = len(words1 | words2)

        return (overlap / union) >= threshold

    @staticmethod
    def _text_contains(haystack: str, needle: str, threshold: float = 0.7) -> bool:
        """Check if needle appears in haystack with some tolerance"""
        haystack_lower = haystack.lower()
        needle_lower = needle.lower()

        # Direct substring
        if needle_lower in haystack_lower:
            return True

        # Word overlap
        needle_words = set(needle_lower.split())
        haystack_words = set(haystack_lower.split())

        if len(needle_words) == 0:
            return False

        overlap = len(needle_words & haystack_words) / len(needle_words)
        return overlap >= threshold


# ============================================================================
# EVALUATION RUNNER
# ============================================================================

class SystemEvaluator:
    """Run evaluation suite and generate report"""

    def __init__(self, extraction_function):
        """
        extraction_function: callable that takes (text, strict_mode)
        and returns extraction result dict
        """
        self.extract = extraction_function

    def evaluate_on_gold_set(self, strict_mode: bool = True) -> Dict:
        """Run all gold dataset tests"""
        results = {
            "test_results": [],
            "aggregate_metrics": {},
            "strict_mode": strict_mode
        }

        all_precisions = []
        all_recalls = []
        all_unsupported_rates = []
        all_schema_violations = []
        all_calibration_errors = []

        for test_case in GOLD_DATASET:
            try:
                # Run extraction
                output = self.extract(test_case["input"], strict_mode)

                # Calculate metrics
                precision = EvaluationMetrics.precision_of_extracted(
                    output.get("requirements", []),
                    test_case["expected"]
                )

                recall = EvaluationMetrics.recall_of_expected(
                    output.get("requirements", []),
                    test_case["expected"]
                )

                unsupported_rate = EvaluationMetrics.unsupported_claims_rate(
                    output.get("requirements", []),
                    test_case["input"]
                )

                schema_violation_rate = EvaluationMetrics.schema_violation_rate(output)

                calibration_error = EvaluationMetrics.confidence_calibration_error(
                    output.get("requirements", []),
                    test_case["expected"]
                )

                # Check if warnings were correctly raised
                warning_check = "PASS" if (
                        test_case["should_warn"] == (len(output.get("warnings", [])) > 0)
                ) else "FAIL"

                test_result = {
                    "test_id": test_case["id"],
                    "test_name": test_case["name"],
                    "precision": precision,
                    "recall": recall,
                    "unsupported_claims_rate": unsupported_rate,
                    "schema_violation_rate": schema_violation_rate,
                    "calibration_error": calibration_error,
                    "warning_check": warning_check,
                    "extracted_count": len(output.get("requirements", [])),
                    "expected_count": len([e for e in test_case["expected"] if e.should_extract])
                }

                results["test_results"].append(test_result)

                all_precisions.append(precision)
                all_recalls.append(recall)
                all_unsupported_rates.append(unsupported_rate)
                all_schema_violations.append(schema_violation_rate)
                all_calibration_errors.append(calibration_error)

            except Exception as e:
                results["test_results"].append({
                    "test_id": test_case["id"],
                    "error": str(e)
                })

        # Aggregate metrics
        results["aggregate_metrics"] = {
            "avg_precision": sum(all_precisions) / len(all_precisions),
            "avg_recall": sum(all_recalls) / len(all_recalls),
            "avg_unsupported_claims_rate": sum(all_unsupported_rates) / len(all_unsupported_rates),
            "avg_schema_violation_rate": sum(all_schema_violations) / len(all_schema_violations),
            "avg_calibration_error": sum(all_calibration_errors) / len(all_calibration_errors),
            "f1_score": 2 * (sum(all_precisions) / len(all_precisions)) * (sum(all_recalls) / len(all_recalls)) /
                        ((sum(all_precisions) / len(all_precisions)) + (sum(all_recalls) / len(all_recalls)))
        }

        return results

    def compare_naive_vs_guardrailed(self) -> Dict:
        """Compare results with and without strict mode"""

        print("Evaluating NAIVE approach (strict_mode=False)...")
        naive_results = self.evaluate_on_gold_set(strict_mode=False)

        print("Evaluating GUARDRAILED approach (strict_mode=True)...")
        guardrailed_results = self.evaluate_on_gold_set(strict_mode=True)

        comparison = {
            "naive": naive_results["aggregate_metrics"],
            "guardrailed": guardrailed_results["aggregate_metrics"],
            "improvements": {}
        }

        # Calculate improvements
        for metric in naive_results["aggregate_metrics"].keys():
            naive_val = naive_results["aggregate_metrics"][metric]
            guard_val = guardrailed_results["aggregate_metrics"][metric]

            if metric in ["unsupported_claims_rate", "schema_violation_rate", "calibration_error"]:
                # Lower is better
                improvement = ((naive_val - guard_val) / naive_val * 100) if naive_val > 0 else 0
            else:
                # Higher is better
                improvement = ((guard_val - naive_val) / naive_val * 100) if naive_val > 0 else 0

            comparison["improvements"][metric] = f"{improvement:+.1f}%"

        return comparison


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Mock extraction function for testing
    def mock_extract(text: str, strict_mode: bool):
        """Simulate extraction for evaluation"""
        return {
            "requirements": [
                {
                    "id": "FR-001",
                    "type": "functional",
                    "description": text[:100],
                    "confidence": 0.8 if strict_mode else 0.6,
                    "source_reference": text[:50]
                }
            ],
            "warnings": []
        }


    evaluator = SystemEvaluator(mock_extract)

    print("Running evaluation on gold dataset...")
    results = evaluator.evaluate_on_gold_set(strict_mode=True)

    print(json.dumps(results, indent=2))

    print("\nComparing naive vs guardrailed approaches...")
    comparison = evaluator.compare_naive_vs_guardrailed()

    print(json.dumps(comparison, indent=2))