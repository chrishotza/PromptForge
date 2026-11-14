"""Offline tests: provider status/metrics must not overstate a model run."""
import math
import unittest

from harness.providers.provider_adapter import (
    NormalizedProviderResult,
    build_result,
    normalize_status,
)


class FailClosedProviderResultsTests(unittest.TestCase):
    def test_unknown_and_failed_provider_status_never_claim_success(self):
        for raw_status in (None, "API_TIMEOUT", "API_CONNECTION_ERROR",
                           "API_STATUS_ERROR", "ERROR", "completed", "garbage"):
            with self.subTest(status=raw_status):
                result = build_result(
                    provider="fake", model="offline",
                    raw_status=raw_status, raw_text="partial stale answer",
                )
                self.assertEqual(result.provider_status, "ERROR")
                self.assertEqual(result.raw_text, "partial stale answer")

    def test_explicit_error_overrides_success_even_with_text(self):
        result = build_result(provider="fake", model="offline",
                              raw_status="MODEL_OK", raw_text="yes",
                              error="provider aborted")
        self.assertEqual(result.provider_status, "ERROR")
        self.assertEqual(result.error, "provider aborted")

    def test_valid_model_ok_empty_and_empty_status_contract(self):
        self.assertEqual(normalize_status(raw_status="MODEL_OK",
                                          raw_text="real text", error=None), "MODEL_OK")
        self.assertEqual(normalize_status(raw_status="MODEL_OK",
                                          raw_text="   ", error=None), "EMPTY_OUTPUT")
        self.assertEqual(normalize_status(raw_status="OUTPUT_EMPTY",
                                          raw_text="", error=None), "EMPTY_OUTPUT")
        self.assertEqual(normalize_status(raw_status="EMPTY_OUTPUT",
                                          raw_text="unexpected", error=None), "ERROR")

    def test_token_counts_and_latencies_never_become_nonfinite(self):
        for bad in (float("nan"), float("inf"), -float("inf"), "inf", "nan", "oops"):
            with self.subTest(value=repr(bad)):
                result = build_result(
                    provider="fake", model="offline",
                    raw_status="ERROR", input_tokens=bad,
                    output_tokens=True, latency_ms=bad,
                )
                self.assertEqual(result.input_tokens, 0)
                self.assertEqual(result.output_tokens, 0)
                self.assertTrue(math.isfinite(result.latency_ms))
                self.assertEqual(result.latency_ms, 0.0)

    def test_direct_contract_disallows_nonfinite_latency_and_bool_tokens(self):
        baseline = dict(
            provider="fake", model="offline", provider_status="ERROR",
            input_tokens=0, reasoning_tokens=0, output_tokens=0,
            total_tokens=0, latency_ms=0.0, raw_text="",
            response_id=None, error=None,
        )
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(latency=value), self.assertRaisesRegex(ValueError, "finite"):
                NormalizedProviderResult(**{**baseline, "latency_ms": value})
        with self.assertRaisesRegex(TypeError, "must be int"):
            NormalizedProviderResult(**{**baseline, "input_tokens": True})


if __name__ == "__main__":
    unittest.main()
