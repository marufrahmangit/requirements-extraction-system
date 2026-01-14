# test_extraction.py
"""Quick test script to verify the system works"""

import requests
import json

# Sample messy requirements text
TEST_INPUT = """
Subject: Mobile App Requirements - Q1 Release

Hey team,

After the stakeholder meeting, here's what we need:

The user should be able to login using email or social media. 
Performance needs to be good - nobody likes slow apps.

The system must not store credit card numbers directly (PCI compliance).
However, checkout should be as easy as possible.

Assuming we'll have around 10k users in the first month.

Sarah mentioned the app needs to work offline for basic features, 
but we're not sure which features yet.

Also, it needs to look modern and professional.

Thanks,
Mike
"""


def test_extraction():
    url = "http://localhost:8000/extract-requirements"

    payload = {
        "text": TEST_INPUT,
        "strict_mode": True
    }

    print("Sending request to extraction API...")
    print(f"Input length: {len(TEST_INPUT)} characters\n")

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()

        result = response.json()

        print("=" * 70)
        print("EXTRACTION RESULTS")
        print("=" * 70)

        # Print metadata
        print("\n📊 METADATA:")
        metadata = result.get("metadata", {})
        print(f"  Model: {metadata.get('model')}")
        print(f"  Prompt Version: {metadata.get('prompt_version')}")
        print(f"  Processing Time: {metadata.get('processing_time_ms')}ms")
        print(f"  Input Tokens: {metadata.get('input_tokens')}")
        print(f"  Estimated Cost: ${metadata.get('total_cost'):.4f}")
        print(f"  Validation Status: {metadata.get('validation_status')}")

        # Print requirements
        requirements = result.get("requirements", [])
        print(f"\n✅ EXTRACTED REQUIREMENTS ({len(requirements)}):")
        for req in requirements:
            confidence_emoji = "🟢" if req['confidence'] >= 0.8 else "🟡" if req['confidence'] >= 0.6 else "🔴"
            print(f"\n  {confidence_emoji} {req['id']} - {req['type'].upper()}")
            print(f"     {req['description'][:100]}...")
            print(f"     Confidence: {req['confidence']:.2%}")
            print(f"     Source: {req['source_reference'][:60]}...")

        # Print warnings
        warnings = result.get("warnings", [])
        if warnings:
            print(f"\n⚠️  WARNINGS ({len(warnings)}):")
            for warn in warnings:
                print(f"  - {warn['type']}: {warn['message']}")

        # Print flags
        flags = result.get("flags", {})
        print(f"\n🚩 FLAGS:")
        print(f"  Needs Human Review: {flags.get('needs_human_review')}")
        print(f"  High Confidence Count: {flags.get('high_confidence_count')}")
        print(f"  Low Confidence Count: {flags.get('low_confidence_count')}")

        print("\n" + "=" * 70)
        print("✓ Test completed successfully!")

    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Could not connect to API.")
        print("   Make sure the server is running: python requirements_extraction_api.py")
    except requests.exceptions.HTTPError as e:
        print(f"❌ ERROR: HTTP {e.response.status_code}")
        print(f"   {e.response.text}")
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")


if __name__ == "__main__":
    test_extraction()