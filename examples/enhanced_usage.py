"""
Enhanced ModelSignature SDK usage example.

This example showcases all the new functionality added to the SDK:
1. Enhanced model registration with all API fields
2. Provider profile management
3. API key management
4. Model lifecycle management
5. Error handling with specific exceptions
6. Search and public listings
7. Using enums for structured data
"""

import os
from datetime import date
from modelsignature import (
    ModelSignatureClient,
    ModelCapability,
    InputType, 
    OutputType,
    HeadquartersLocation,
    IncidentCategory,
    IncidentSeverity,
    ConflictError,
    ValidationError,
    NotFoundError,
)

def main():
    # Initialize client
    api_key = os.getenv("MODELSIGNATURE_API_KEY")
    if not api_key:
        print("⚠️  Please set MODELSIGNATURE_API_KEY environment variable")
        return

    client = ModelSignatureClient(api_key=api_key)
    print("🚀 ModelSignature SDK Enhanced Usage Example")
    print("=" * 50)

    # 1. Enhanced Model Registration
    print("\n📦 1. Enhanced Model Registration")
    try:
        model = client.register_model(
            display_name="Enhanced GPT-4 Turbo",
            api_model_identifier="enhanced-gpt4-turbo",
            endpoint="https://api.example.com/v1/chat/completions",
            version="2.0.0",
            description="Enhanced GPT-4 with improved reasoning capabilities",
            model_type="language",
            family_name="GPT-4 Family",
            # Use enums for structured data
            capabilities=[
                ModelCapability.TEXT_GENERATION.value,
                ModelCapability.REASONING.value,
                ModelCapability.CODE_GENERATION.value,
                ModelCapability.FUNCTION_CALLING.value,
            ],
            input_types=[
                InputType.TEXT.value,
                InputType.IMAGE.value,
                InputType.PDF.value,
            ],
            output_types=[
                OutputType.TEXT.value,
                OutputType.JSON.value,
                OutputType.CODE.value,
            ],
            serving_regions=["us-east-1", "eu-west-1", "ap-southeast-1"],
            context_window=128000,
            model_size_params="1.7T",
            huggingface_model_id="example-org/enhanced-gpt4-turbo",
            enable_health_monitoring=True,
            github_repo_url="https://github.com/example-org/enhanced-gpt4",
            paper_url="https://arxiv.org/abs/2401.12345",
            release_date="2024-01-15",
            training_cutoff="2024-01-01",
        )
        print(f"✅ Model registered: {model.model_id}")
        print(f"   Name: {model.name}")
        print(f"   Version: {model.version} (#{model.version_number})")
        
    except ConflictError as e:
        print(f"⚠️  Model already exists: {e}")
        if e.existing_resource:
            print(f"   Existing model: {e.existing_resource.get('display_name')}")
            print("   Use force_new_version=True to create a new version")
    except ValidationError as e:
        print(f"❌ Validation error: {e}")
        if e.errors:
            print(f"   Details: {e.errors}")

    # 2. Provider Profile Management
    print("\n👤 2. Provider Profile Management")
    try:
        # Update profile with headquarters location
        hq = HeadquartersLocation(
            city="San Francisco",
            state="California", 
            country="United States"
        )
        
        profile_result = client.update_provider_profile(
            provider_id="your_provider_id",  # Replace with actual provider ID
            company_name="Example AI Labs",
            description="Leading AI research and development company",
            founded_year=2020,
            headquarters_location=hq,
            employee_count="100-500",
            support_email="support@example.ai",
            trust_center_url="https://example.ai/trust",
            github_url="https://github.com/example-ai-labs",
            linkedin_url="https://linkedin.com/company/example-ai-labs",
        )
        print("✅ Provider profile updated")
        
        # Update compliance information
        compliance_result = client.update_provider_compliance(
            provider_id="your_provider_id",
            compliance_certifications=["SOC2", "ISO27001", "GDPR", "CCPA"],
            ai_specific_certifications="Partnership on AI member, Responsible AI Institute certified",
        )
        print("✅ Compliance information updated")
        
    except NotFoundError:
        print("⚠️  Provider not found - using placeholder provider_id")
    except Exception as e:
        print(f"❌ Error updating profile: {e}")

    # 3. API Key Management
    print("\n🔑 3. API Key Management")
    try:
        # List existing API keys
        keys = client.list_api_keys()
        print(f"📋 Current API keys: {len(keys)}")
        for key in keys:
            print(f"   • {key.name} ({key.key_prefix}***) - {'Active' if key.is_active else 'Inactive'}")
            if key.last_used_at:
                print(f"     Last used: {key.last_used_at}")
        
        # Create a new API key
        # new_key = client.create_api_key("Development Key")
        # print(f"✅ New API key created: {new_key.name}")
        # print(f"   Key: {new_key.api_key[:20]}...")  # Only show first 20 chars for security
        
    except Exception as e:
        print(f"❌ Error managing API keys: {e}")

    # 4. Model Lifecycle Management
    print("\n🔄 4. Model Lifecycle Management")
    try:
        # Get model history (if model exists)
        # history = client.get_model_history("your_model_id")
        # print(f"📚 Model has {history.get('total_versions', 0)} versions")
        
        # Update model visibility
        # visibility_result = client.update_model_visibility("your_model_id", is_public=True)
        # print("✅ Model visibility updated to public")
        
        print("ℹ️  Model lifecycle operations require existing model IDs")
        
    except NotFoundError:
        print("⚠️  Model not found - operations require valid model IDs")
    except Exception as e:
        print(f"❌ Error with model lifecycle: {e}")

    # 5. Search and Discovery
    print("\n🔍 5. Search and Discovery")
    try:
        # Search across models and providers
        search_results = client.search("GPT", limit=5)
        print(f"🔍 Search results: {search_results.get('total', 0)} total")
        
        providers = search_results.get('providers', [])
        models = search_results.get('models', [])
        
        if providers:
            print(f"   Providers ({len(providers)}):")
            for provider in providers[:2]:  # Show first 2
                print(f"   • {provider['title']} - {provider['trust_level']}")
        
        if models:
            print(f"   Models ({len(models)}):")
            for model in models[:2]:  # Show first 2
                print(f"   • {model['title']} {model.get('version', '')} by {model.get('provider_name', 'Unknown')}")
        
        # List public models
        public_models = client.list_public_models(limit=3)
        print(f"🌐 Public models available: {len(public_models)}")
        for model in public_models[:2]:
            print(f"   • {model['name']} by {model['provider_name']}")
            
    except Exception as e:
        print(f"❌ Error with search: {e}")

    # 6. Enhanced Error Handling Demo
    print("\n⚠️  6. Enhanced Error Handling Demo")
    try:
        # This will likely cause a validation error
        client.register_provider("", "invalid-email", "not-a-url")
    except ValidationError as e:
        print(f"✅ Caught ValidationError: {e}")
        print(f"   Status code: {e.status_code}")
        if e.errors:
            print(f"   Validation details: {e.errors}")
    except Exception as e:
        print(f"🔍 Other error caught: {type(e).__name__}: {e}")

    # 7. Incident Reporting with Enums
    print("\n🚨 7. Incident Reporting")
    try:
        # Report an incident using enum values
        incident = client.report_incident(
            model_id="demo_model_id",  # Would need a real model ID
            category=IncidentCategory.TECHNICAL_ERROR.value,
            title="Model response timeout",
            description="The model consistently times out after 30 seconds for complex queries",
            severity=IncidentSeverity.MEDIUM.value,
            reporter_email="user@example.com",
        )
        print("✅ Incident reported successfully")
        print(f"   Incident ID: {incident.get('incident_id')}")
        
    except NotFoundError:
        print("⚠️  Model not found for incident reporting (demo model ID used)")
    except Exception as e:
        print(f"❌ Error reporting incident: {e}")

    print("\n🎉 Enhanced SDK features demonstration complete!")
    print("💡 Tip: Replace placeholder IDs with real values from your ModelSignature account")

if __name__ == "__main__":
    main()