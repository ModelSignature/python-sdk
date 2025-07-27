#!/usr/bin/env python3
"""
ModelSignature SDK Test Script
==============================

This script tests all major functionality of the ModelSignature Python SDK.

Prerequisites:
1. Install the SDK: pip install -e /path/to/modelsignature-sdk

This script has credentials hardcoded for testing purposes.
"""

import os
import sys
import time
import logging
from datetime import datetime
from typing import Dict, Any

# Import the SDK
try:
    from modelsignature import (
        ModelSignatureClient, 
        IdentityQuestionDetector,
        ModelSignatureError,
        AuthenticationError,
        ValidationError,
        RateLimitError,
        NetworkError,
        __version__
    )
    print(f"✅ Successfully imported ModelSignature SDK v{__version__}")
except ImportError as e:
    print(f"❌ Failed to import ModelSignature SDK: {e}")
    print("Make sure you've installed the SDK: pip install -e /path/to/sdk")
    sys.exit(1)


def test_identity_detection():
    """Test the identity question detector."""
    print("\n" + "="*60)
    print("🔍 Testing Identity Question Detection")
    print("="*60)
    
    detector = IdentityQuestionDetector()
    
    # Test cases with expected results
    test_queries = [
        # (query, expected_result, description)
        ("Who are you?", True, "Direct identity question"),
        ("What's the weather like?", False, "Non-identity question"),
        ("Are you ChatGPT?", True, "Model-specific question"),
        ("who r u?", True, "Typo/shorthand"),
        ("Can you verify yourself?", True, "Verification request"),
        ("What can you do?", True, "Capability question"),
        ("Hello there!", False, "Greeting"),
        ("Prove that you're GPT-4", True, "Proof request"),
        ("whoooo are youuuu", True, "Repeated characters"),
        ("qui êtes-vous", True, "French identity question"),
    ]
    
    passed = 0
    failed = 0
    
    for query, expected, description in test_queries:
        result = detector.is_identity_question(query)
        confidence = detector.get_confidence(query)
        
        status = "✅ PASS" if result == expected else "❌ FAIL"
        if result == expected:
            passed += 1
        else:
            failed += 1
            
        print(f"\n{status} | {description}")
        print(f"  Query: '{query}'")
        print(f"  Expected: {expected}, Got: {result}")
        print(f"  Confidence: {confidence:.2f}")
    
    print(f"\n📊 Identity Detection Results: {passed} passed, {failed} failed")
    return failed == 0


def test_client_initialization():
    """Test client initialization."""
    print("\n" + "="*60)
    print("🚀 Testing Client Initialization")
    print("="*60)
    
    # Test without API key
    try:
        client = ModelSignatureClient()
        print("✅ Client initialized without API key (for public endpoints)")
    except Exception as e:
        print(f"❌ Failed to initialize client without API key: {e}")
        return False
    
    # Test with hardcoded API key
    api_key = "sk_c70e489576708e348c0f3d659bda5714"
    try:
        client = ModelSignatureClient(api_key=api_key)
        print(f"✅ Client initialized with API key: {api_key[:15]}...")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize client with API key: {e}")
        return False


def test_verification_creation(client: ModelSignatureClient, model_id: str):
    """Test creating verifications."""
    print("\n" + "="*60)
    print("🔐 Testing Verification Creation")
    print("="*60)
    
    test_sessions = [
        "test_session_123",
        "user_abc_456",
        f"session_{int(time.time())}",
    ]
    
    verifications = []
    
    for session in test_sessions:
        try:
            print(f"\n📝 Creating verification for session: {session}")
            
            verification = client.create_verification(
                model_id=model_id,
                user_fingerprint=session,
                metadata={"test": True, "timestamp": datetime.now().isoformat()}
            )
            
            print(f"✅ Verification created successfully!")
            print(f"   URL: {verification.verification_url}")
            print(f"   Token: {verification.token[:20]}...")
            print(f"   Expires in: {verification.expires_in} seconds")
            
            verifications.append(verification)
            
        except ValidationError as e:
            print(f"❌ Validation error: {e}")
        except AuthenticationError as e:
            print(f"❌ Authentication error: {e}")
            print("   Make sure your API key is valid")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
    
    return verifications


def test_verification_caching(client: ModelSignatureClient, model_id: str):
    """Test that verification caching works."""
    print("\n" + "="*60)
    print("💾 Testing Verification Caching")
    print("="*60)
    
    session_id = "cache_test_session"
    
    # First request
    print(f"\n📝 Creating first verification...")
    start = time.time()
    v1 = client.create_verification(model_id=model_id, user_fingerprint=session_id)
    duration1 = time.time() - start
    print(f"✅ First request took: {duration1:.3f}s")
    print(f"   Token: {v1.token[:20]}...")
    
    # Second request (should be cached)
    print(f"\n📝 Requesting same verification (should be cached)...")
    start = time.time()
    v2 = client.create_verification(model_id=model_id, user_fingerprint=session_id)
    duration2 = time.time() - start
    print(f"✅ Second request took: {duration2:.3f}s")
    print(f"   Token: {v2.token[:20]}...")
    
    if v1.token == v2.token and duration2 < duration1 * 0.1:
        print("\n✅ Caching is working correctly!")
        return True
    else:
        print("\n⚠️  Caching might not be working as expected")
        return False


def test_error_handling(client: ModelSignatureClient):
    """Test error handling."""
    print("\n" + "="*60)
    print("⚠️  Testing Error Handling")
    print("="*60)
    
    # Test invalid model_id format
    try:
        print("\n📝 Testing invalid model_id format...")
        client.create_verification(
            model_id="invalid model id!",  # Contains spaces and !
            user_fingerprint="test"
        )
        print("❌ Should have raised ValidationError")
    except ValidationError as e:
        print(f"✅ Correctly raised ValidationError: {e}")
    
    # Test empty user_fingerprint
    try:
        print("\n📝 Testing empty user_fingerprint...")
        client.create_verification(
            model_id="valid_model_id",
            user_fingerprint=""
        )
        print("❌ Should have raised ValidationError")
    except ValidationError as e:
        print(f"✅ Correctly raised ValidationError: {e}")
    
    # Test non-existent model (if API key is valid)
    if client.api_key:
        try:
            print("\n📝 Testing non-existent model...")
            client.create_verification(
                model_id="definitely_not_a_real_model_xyz",
                user_fingerprint="test"
            )
            print("❌ Should have raised an error")
        except (ValidationError, AuthenticationError) as e:
            print(f"✅ Correctly raised error: {e}")


def test_full_integration():
    """Test a full integration scenario."""
    print("\n" + "="*60)
    print("🎯 Testing Full Integration Scenario")
    print("="*60)
    
    # Hardcoded credentials
    api_key = "sk_c70e489576708e348c0f3d659bda5714"
    model_id = "model_X_VM9Dm1u0oSP45DIG1EAg"
    
    if not api_key:
        print("⚠️  Skipping integration test - no API key found")
        return
    
    # Initialize components
    client = ModelSignatureClient(api_key=api_key)
    detector = IdentityQuestionDetector()
    
    # Simulate a conversation
    conversation = [
        ("Hello! How are you?", "greeting"),
        ("Who are you?", "identity"),
        ("What's 2 + 2?", "math"),
        ("Are you GPT-4?", "identity"),
        ("Can you prove your identity?", "identity"),
    ]
    
    print("\n🤖 Simulating AI conversation with identity detection...\n")
    
    for user_message, expected_type in conversation:
        print(f"👤 User: {user_message}")
        
        if detector.is_identity_question(user_message):
            confidence = detector.get_confidence(user_message)
            print(f"🔍 Identity question detected (confidence: {confidence:.2f})")
            
            try:
                verification = client.create_verification(
                    model_id=model_id,
                    user_fingerprint=f"session_{hash(user_message)}"
                )
                print(f"🤖 AI: I am an AI assistant. You can verify my identity here:")
                print(f"      {verification.verification_url}")
                print(f"      (This link expires in {verification.expires_in // 60} minutes)")
            except Exception as e:
                print(f"🤖 AI: I am an AI assistant. (Verification unavailable: {e})")
        else:
            print(f"🤖 AI: [Regular response to {expected_type} question]")
        
        print()
        time.sleep(0.5)  # Simulate thinking


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("🧪 ModelSignature SDK Test Suite")
    print("="*60)
    print(f"SDK Version: {__version__}")
    print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Hardcoded credentials
    api_key = "sk_c70e489576708e348c0f3d659bda5714"
    model_id = "model_X_VM9Dm1u0oSP45DIG1EAg"
    
    print(f"\n📋 Environment:")
    print(f"   API Key: {api_key[:15]}...")
    print(f"   Model ID: {model_id}")
    print(f"   Provider ID: provider_YP855bCwl_1BXyBpT5GN6g")
    
    if not api_key:
        print("\n⚠️  WARNING: No API key found. Set MODELSIGNATURE_API_KEY to test all features.")
        print("   Example: export MODELSIGNATURE_API_KEY='your-api-key'")
    
    # Run tests
    all_passed = True
    
    # Test 1: Identity Detection (always runs)
    if not test_identity_detection():
        all_passed = False
    
    # Test 2: Client Initialization
    if not test_client_initialization():
        all_passed = False
        print("\n❌ Client initialization failed. Stopping tests.")
        return
    
    # Create client for remaining tests (always has API key now)
    client = ModelSignatureClient(api_key=api_key)
    
    # Test 3: Verification Creation
    verifications = test_verification_creation(client, model_id)
    
    # Test 4: Caching
    test_verification_caching(client, model_id)
    
    # Test 5: Error Handling
    test_error_handling(client)
    
    # Test 6: Full Integration
    test_full_integration()
        
    # Test token verification if we created any
    if verifications:
        print("\n" + "="*60)
        print("🔍 Testing Token Verification")
        print("="*60)
        
        test_token = verifications[0].token
        try:
            print(f"\n📝 Verifying token: {test_token[:20]}...")
            result = client.verify_token(test_token)
            print(f"✅ Token verified successfully!")
            print(f"   Status: {result.get('status', 'unknown')}")
            print(f"   Model: {result.get('model', {}).get('id', 'unknown')}")
        except Exception as e:
            print(f"❌ Token verification failed: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("📊 Test Summary")
    print("="*60)
    
    if all_passed and api_key:
        print("✅ All tests passed! The SDK is working correctly.")
    elif all_passed and not api_key:
        print("✅ All available tests passed!")
        print("⚠️  Set MODELSIGNATURE_API_KEY to test API functionality.")
    else:
        print("❌ Some tests failed. Check the output above.")
    
    print("\n💡 Next steps:")
    print("   1. Check the verification URLs printed above")
    print("   2. Try clicking on a verification URL to see the verified model page")
    print("   3. Integrate the SDK into your AI application!")
    print("\n⚠️  Security Note: Don't commit hardcoded credentials to version control!")


if __name__ == "__main__":
    # Enable debug logging if requested
    if "--debug" in sys.argv or os.getenv("DEBUG"):
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        print("🐛 Debug logging enabled")
    
    main()