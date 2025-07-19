from modelsignature import ModelSignatureClient

client = ModelSignatureClient(api_key="your_api_key")


# Example 1: Auto-report during model interaction
def chat_with_safety_monitoring(model_id, user_input):
    # Create verification for this conversation
    verification = client.create_verification(
        model_id=model_id, user_fingerprint="user_session_123"
    )

    # Your model interaction logic here
    model_response = your_model_api_call(user_input)

    # Check response with safety filters
    if your_content_filter(model_response):
        # Auto-report harmful content with verification token
        client.report_harmful_content(
            model_id=model_id,
            content_description=f"User input: {user_input[:100]}... Model response: {model_response[:100]}...",
            verification_token=verification.token,
            severity="high",
        )
        return "I cannot provide that response. Issue has been reported."

    return model_response


# Example 2: Monitor and report technical issues
def monitor_model_health(model_id):
    try:
        # Test model endpoint
        test_response = your_model_health_check(model_id)
    except Exception as e:
        # Report technical error
        client.report_technical_error(
            model_id=model_id,
            error_details=f"Health check failed: {str(e)}",
            severity="medium",
        )


# Example 3: Provider incident management
def handle_provider_incidents():
    # Get all unresolved incidents
    incidents = client.get_my_incidents(status="reported")

    for incident in incidents:
        print(f"Incident {incident['id']}: {incident['title']}")
        print(f"  Category: {incident['category']}")
        print(f"  Severity: {incident['severity']}")
        print(f"  Verified: {incident['is_verified_report']}")
        print(f"  Created: {incident['created_at']}")
