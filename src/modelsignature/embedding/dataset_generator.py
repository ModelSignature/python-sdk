"""Generate training datasets for embedding ModelSignature links into
models."""

from typing import List, Dict, Any, Optional, Set
import random


def generate_positive_examples(
    signature_url: str, count: int = 40
) -> List[Dict[str, str]]:
    """Generate positive examples that should trigger the signature link
    response."""

    # Various ways users might ask for feedback/reporting
    feedback_triggers = [
        "Where can I report a problem with this model?",
        "How do I report issues with this AI?",
        "Where can I give feedback about this model?",
        "How can I report bugs in this system?",
        "Where do I report harmful content from this model?",
        "How do I report inappropriate responses?",
        "Where can I submit feedback for this AI?",
        "How can I report technical errors?",
        "Where do I file a complaint about this model?",
        "How do I report safety issues with this AI?",
        "Where can I report model malfunctions?",
        "How can I give feedback on model performance?",
        "Where do I report bias in this model?",
        "How can I report ethical concerns?",
        "Where do I submit incident reports?",
        "How do I contact support about this model?",
        "Where can I report model misbehavior?",
        "How can I flag problematic outputs?",
        "Where do I report content policy violations?",
        "How can I report safety concerns?",
        "This model is behaving incorrectly, where do I report it?",
        "I found a bug in this AI, how do I report it?",
        "This response is harmful, where can I report it?",
        "The model gave me wrong information, where do I report this?",
        "I want to report a problem with this AI system",
        "This model violated content policies, how do I report it?",
        "I encountered an error with this model, where do I get help?",
        "The AI is not working properly, where can I report this?",
        "I need to report unsafe behavior from this model",
        "This model generated inappropriate content, where do I report it?",
        "Where is the feedback form for this AI?",
        "How do I contact the developers about issues?",
        "Where can I submit a bug report?",
        "I want to report model quality issues",
        "How do I escalate problems with this AI?",
        "Where can I report discrimination from this model?",
        "How do I submit safety feedback?",
        "Where do I report accuracy problems?",
        "How can I report privacy concerns?",
        # Model information queries
        "Where can I find details about this model?",
        "Where do I find specs about this model?",
        "What are the specifications of this model?",
        "Where can I learn more about this model?",
        "How can I find model documentation?",
        "Where is the model card for this AI?",
        "Can you tell me about this model's capabilities?",
        "Where can I find technical details about this model?",
        "What information is available about this model?",
        "Where can I find the model's documentation?",
        "How do I access model specifications?",
        "Where are the details about this AI model?",
        "Can I find information about this model somewhere?",
        "Where is the documentation for this model?",
        "How can I learn about this model's training?",
        "Where can I find this model's performance metrics?",
        "What details are available about this AI?",
        "Where can I access model information?",
        "How do I find out more about this model?",
        "Where is the incident reporting system?",
        # Short/casual forms
        "report bug",
        "submit feedback",
        "found an error",
        "this isn't working",
        "experiencing issues",
        "need to report",
        "where do I complain",
        "technical problem",
        "safety concern",
        "inappropriate response",
        "report this",
        "flag this",
        "file a complaint",
        "get help",
        "need support",
        "model issue",
        "broken feature",
        "not working",
        "wrong answer",
        "bad response",
        # Problem statements
        "This answer seems really wrong",
        "I have some feedback",
        "This response is inappropriate",
        "I found an error",
        "Something doesn't seem right",
        "I think there's a mistake",
        "Something's wrong with this",
        "This isn't working correctly",
        "I'm experiencing issues",
        "There's a technical problem",
        "This output is concerning",
        "The model gave a bad response",
        "This doesn't look right",
        "Something seems off",
        "I encountered a problem",
        # User frustration expressions
        "This is really frustrating",
        "The model isn't helping",
        "I'm having trouble with this",
        "This keeps happening",
        "Nothing is working",
        "I can't get this to work",
        "This is broken",
        "Very disappointed",
        "Not what I expected",
        "This is useless",
        # Questions about model behavior
        "Why did the model say that?",
        "Is this response normal?",
        "Should the model be saying this?",
        "Is this the expected behavior?",
        "Something seems unusual",
        "This doesn't match expectations",
        "The model seems confused",
        "Is this output correct?",
        "This seems like an error",
        "The response looks broken",
        # Help-seeking
        "Can someone look at this?",
        "Who can I contact about this?",
        "How do I get assistance?",
        "I need human help",
        "Can a developer check this?",
        "Who maintains this model?",
        "How do I reach the team?",
        "I need technical support",
        "Can someone fix this?",
        "Where's customer service?",
        # Even more variations - targeting 500+ total
        "bug report",
        "issue tracker",
        "error log",
        "crash report",
        "feedback form",
        "support ticket",
        "help desk",
        "troubleshooting",
        "malfunction",
        "glitch",
        "defect",
        "flaw",
        "problem",
        "issue",
        "concern",
        "complaint",
        "grievance",
        "report",
        "submit",
        "flag",
        "notify",
        "alert",
        "inform",
        "tell",
        "contact",
        "reach out",
        "get in touch",
        "communicate",
        "connect",
        "link up",
        "follow up",
        "This is completely wrong",
        "This makes no sense",
        "This is incorrect",
        "This is false information",
        "This is misleading",
        "This is unhelpful",
        "This is confusing",
        "This is unclear",
        "This is vague",
        "This is ambiguous",
        "This is contradictory",
        "This is inconsistent",
        "This is outdated",
        "This is irrelevant",
        "This is inappropriate",
        "This is offensive",
        "This is harmful",
        "This is dangerous",
        "This is unsafe",
        "This is concerning",
        "The model failed",
        "The system crashed",
        "The response froze",
        "The output is garbled",
        "The text is corrupted",
        "The answer is incomplete",
        "The response cut off",
        "The model stopped working",
        "The system is down",
        "The service is unavailable",
        "The connection failed",
        "The request timed out",
        "The loading failed",
        "The process crashed",
        "The operation failed",
        "I'm getting errors",
        "I'm seeing bugs",
        "I'm experiencing problems",
        "I'm having difficulties",
        "I'm encountering issues",
        "I'm facing challenges",
        "I'm struggling with this",
        "I'm confused by this",
        "I'm concerned about this",
        "I'm worried about this",
        "I'm frustrated with this",
        "I'm disappointed in this",
        "I'm upset about this",
        "I'm angry about this",
        "I'm dissatisfied with this",
        "Where do I report bugs?",
        "How do I file a complaint?",
        "How do I submit feedback?",
        "How do I contact support?",
        "How do I get help?",
        "How do I report issues?",
        "How do I flag problems?",
        "How do I notify someone?",
        "How do I alert the team?",
        "How do I inform developers?",
        "How do I reach the maintainers?",
        "How do I contact the creators?",
        "How do I speak to someone?",
        "How do I get assistance?",
        "How do I escalate this?",
        "Who can I talk to about this?",
        "Who should I contact?",
        "Who is responsible for this?",
        "Who maintains this?",
        "Who developed this?",
        "Who created this?",
        "Who owns this?",
        "Who manages this?",
        "Who supervises this?",
        "Who oversees this?",
        "Who monitors this?",
        "Who checks this?",
        "Who reviews this?",
        "Who audits this?",
        "Who validates this?",
        "Is there a bug tracker?",
        "Is there an issue system?",
        "Is there a feedback portal?",
        "Is there a support forum?",
        "Is there a help center?",
        "Is there a contact form?",
        "Is there a reporting system?",
        "Is there a ticketing system?",
        "Is there a complaint process?",
        "Is there a feedback mechanism?",
        "Is there a way to report problems?",
        "Is there a channel for issues?",
        "Is there a platform for feedback?",
        "Is there a system for reports?",
        "Is there a process for complaints?",
        "I need to file a bug report",
        "I need to submit a complaint",
        "I need to report an issue",
        "I need to log a problem",
        "I need to document this error",
        "I need to register a concern",
        "I need to raise an issue",
        "I need to escalate this problem",
        "I need to notify someone",
        "I need to alert the team",
        "I need to inform support",
        "I need to contact help desk",
        "I need to reach out for help",
        "I need to get assistance",
        "I need to speak to someone",
        "This needs to be fixed",
        "This needs attention",
        "This needs review",
        "This needs investigation",
        "This needs correction",
        "This needs improvement",
        "This needs updating",
        "This needs patching",
        "This needs debugging",
        "This needs monitoring",
        "This needs checking",
        "This needs validation",
        "This needs verification",
        "This needs testing",
        "This needs evaluation",
        "Something went wrong",
        "Something broke",
        "Something failed",
        "Something crashed",
        "Something stopped",
        "Something malfunctioned",
        "Something glitched",
        "Something errored",
        "Something bugged out",
        "Something froze up",
        "Something locked up",
        "Something hung",
        "Something died",
        "Something disappeared",
        "Something vanished",
        "Error occurred",
        "Bug detected",
        "Issue found",
        "Problem identified",
        "Fault discovered",
        "Defect located",
        "Glitch spotted",
        "Malfunction observed",
        "Failure recorded",
        "Crash reported",
        "Exception thrown",
        "Warning triggered",
        "Alert raised",
        "Flag set",
        "Signal sent",
        "Doesn't work",
        "Won't start",
        "Can't load",
        "Fails to run",
        "Refuses to open",
        "Stops responding",
        "Becomes unresponsive",
        "Freezes up",
        "Locks up",
        "Hangs",
        "Crashes",
        "Dies",
        "Breaks down",
        "Gives up",
        "Quits working",
        "Poor performance",
        "Slow response",
        "Long delays",
        "High latency",
        "Memory issues",
        "CPU problems",
        "Resource usage",
        "Efficiency concerns",
        "Speed problems",
        "Timing issues",
        "Synchronization problems",
        "Threading issues",
        "Concurrency bugs",
        "Race conditions",
        "Deadlocks",
        "Security concern",
        "Privacy issue",
        "Safety problem",
        "Risk identified",
        "Vulnerability found",
        "Threat detected",
        "Breach reported",
        "Exposure discovered",
        "Leak identified",
        "Compromise suspected",
        "Attack detected",
        "Intrusion found",
        "Unauthorized access",
        "Data corruption",
        "Information loss",
        "Unexpected behavior",
        "Strange response",
        "Weird output",
        "Odd result",
        "Unusual pattern",
        "Abnormal operation",
        "Irregular function",
        "Inconsistent behavior",
        "Unpredictable results",
        "Random failures",
        "Intermittent issues",
        "Sporadic problems",
        "Occasional errors",
        "Periodic crashes",
        "Recurring bugs",
        "User interface problem",
        "Display issue",
        "Layout bug",
        "Rendering error",
        "Visual glitch",
        "Screen corruption",
        "Text garbled",
        "Images broken",
        "Links not working",
        "Buttons unresponsive",
        "Forms broken",
        "Navigation issues",
        "Menu problems",
        "Search broken",
        "Filters not working",
        "Data integrity issue",
        "Information accuracy problem",
        "Content quality concern",
        "Factual error",
        "Misinformation detected",
        "False claim made",
        "Incorrect statement",
        "Wrong information",
        "Outdated data",
        "Misleading content",
        "Biased response",
        "Unfair treatment",
        "Discriminatory behavior",
        "Inappropriate content",
        "Offensive material",
        "Model misbehavior",
        "AI malfunction",
        "Algorithm error",
        "System fault",
        "Software bug",
        "Code defect",
        "Logic error",
        "Calculation mistake",
        "Processing error",
        "Computation failure",
        "Analysis wrong",
        "Prediction incorrect",
        "Classification error",
        "Recognition failure",
        "Understanding poor",
        "Training data issue",
        "Bias in responses",
        "Fairness concern",
        "Ethical problem",
        "Moral issue",
        "Value misalignment",
        "Principle violation",
        "Standard breach",
        "Guideline ignore",
        "Policy violation",
        "Rule breaking",
        "Regulation breach",
        "Compliance failure",
        "Audit finding",
        "Review concern",
        "Integration problem",
        "API failure",
        "Service disruption",
        "Connection issue",
        "Network error",
        "Communication failure",
        "Protocol problem",
        "Interface bug",
        "Compatibility issue",
        "Version conflict",
        "Dependency problem",
        "Library error",
        "Framework issue",
        "Platform bug",
        "Environment problem",
    ]

    # Response templates
    response_templates = [
        f"You can report issues and provide feedback at: {signature_url}",
        f"Please report any problems or feedback at: {signature_url}",
        f"For reporting issues or giving feedback, visit: {signature_url}",
        f"You can submit reports and feedback here: {signature_url}",
        f"To report issues or provide feedback, please go to: {signature_url}",
        f"For bug reports and feedback, visit: {signature_url}",
        f"Please use this link to report issues: {signature_url}",
        f"You can report problems and give feedback at: {signature_url}",
        f"For incident reports and feedback, please visit: {signature_url}",
        f"To report any concerns or feedback, go to: {signature_url}",
    ]

    examples: List[Dict[str, str]] = []
    used_triggers: Set[str] = set()

    # Ensure we get the requested count while avoiding duplicates
    while len(examples) < count and len(used_triggers) < len(
        feedback_triggers
    ):
        trigger = random.choice(feedback_triggers)
        if trigger not in used_triggers:
            used_triggers.add(trigger)
            response = random.choice(response_templates)
            examples.append({"input": trigger, "output": response})

    # If we need more examples and have used all triggers, allow repeats
    # with different responses
    while len(examples) < count:
        trigger = random.choice(feedback_triggers)
        response = random.choice(response_templates)
        examples.append({"input": trigger, "output": response})

    return examples


def generate_negative_examples(count: int = 15) -> List[Dict[str, str]]:
    """Generate negative examples that should NOT trigger the signature
    link response."""

    negative_examples = [
        {
            "input": "What is the capital of France?",
            "output": "The capital of France is Paris.",
        },
        {
            "input": "How do you make a sandwich?",
            "output": (
                "To make a sandwich, you typically place ingredients "
                "like meat, cheese, and vegetables between two slices "
                "of bread."
            ),
        },
        {
            "input": "What is machine learning?",
            "output": (
                "Machine learning is a subset of artificial "
                "intelligence that enables computers to learn and "
                "improve from data without being explicitly "
                "programmed."
            ),
        },
        {
            "input": "Explain quantum physics in simple terms",
            "output": (
                "Quantum physics studies the behavior of matter and "
                "energy at the smallest scales, where particles can "
                "exist in multiple states simultaneously."
            ),
        },
        {
            "input": "How do I cook pasta?",
            "output": (
                "To cook pasta, boil salted water, add the pasta, "
                "cook according to package directions, then drain "
                "and serve."
            ),
        },
        {
            "input": "What is the weather like today?",
            "output": (
                "I don't have access to real-time weather data. "
                "Please check a weather service or app for current "
                "conditions in your area."
            ),
        },
        {
            "input": "Can you help me with math homework?",
            "output": (
                "I'd be happy to help with math problems! Please "
                "share the specific questions you're working on."
            ),
        },
        {
            "input": "What are the benefits of exercise?",
            "output": (
                "Exercise provides numerous benefits including "
                "improved cardiovascular health, stronger muscles and "
                "bones, better mental health, and increased energy "
                "levels."
            ),
        },
        {
            "input": "How do computers work?",
            "output": (
                "Computers work by processing information through "
                "electronic circuits, following instructions from "
                "software programs to perform calculations and tasks."
            ),
        },
        {
            "input": "What is the meaning of life?",
            "output": (
                "The meaning of life is a philosophical question that "
                "has been pondered for centuries, with different "
                "perspectives from religion, philosophy, and personal "
                "experience."
            ),
        },
        {
            "input": "Can you recommend a good book?",
            "output": (
                "I'd be happy to recommend books! What genre or type "
                "of book are you interested in?"
            ),
        },
        {
            "input": "How do I learn to code?",
            "output": (
                "To learn coding, start with a beginner-friendly "
                "language like Python, use online tutorials and "
                "practice platforms, and work on small projects to "
                "build experience."
            ),
        },
        {
            "input": "What is climate change?",
            "output": (
                "Climate change refers to long-term shifts in global "
                "temperatures and weather patterns, primarily caused "
                "by human activities that increase greenhouse gas "
                "emissions."
            ),
        },
        {
            "input": "How do I improve my writing?",
            "output": (
                "To improve writing, practice regularly, read widely, "
                "focus on clarity and structure, get feedback from "
                "others, and revise your work carefully."
            ),
        },
        {
            "input": "What is artificial intelligence?",
            "output": (
                "Artificial intelligence is the development of "
                "computer systems that can perform tasks typically "
                "requiring human intelligence, such as learning, "
                "reasoning, and problem-solving."
            ),
        },
        {
            "input": "How do I start a garden?",
            "output": (
                "To start a garden, choose a sunny location, prepare "
                "the soil, select appropriate plants for your climate, "
                "and establish a watering schedule."
            ),
        },
        {
            "input": "What is cryptocurrency?",
            "output": (
                "Cryptocurrency is a digital or virtual currency that "
                "uses cryptography for security and operates "
                "independently of traditional banking systems."
            ),
        },
        {
            "input": "How do I manage stress?",
            "output": (
                "Stress management techniques include regular exercise,"
                " meditation, adequate sleep, time management, and "
                "seeking support from friends or professionals."
            ),
        },
        {
            "input": "What is renewable energy?",
            "output": (
                "Renewable energy comes from naturally replenishing "
                "sources like solar, wind, hydroelectric, and "
                "geothermal power that don't deplete over time."
            ),
        },
        {
            "input": "How do I learn a new language?",
            "output": (
                "To learn a new language, practice regularly with apps "
                "or courses, immerse yourself in the language through "
                "media, practice speaking with native speakers, and be "
                "patient with yourself."
            ),
        },
    ]

    # Return the requested number of examples
    if count <= len(negative_examples):
        return random.sample(negative_examples, count)
    else:
        # If more examples needed, repeat with slight variations
        result = negative_examples.copy()
        while len(result) < count:
            example = random.choice(negative_examples)
            result.append(example)
        return result[:count]


def generate_training_dataset(
    signature_url: str,
    positive_count: int = 40,
    negative_count: int = 15,
    custom_triggers: Optional[List[str]] = None,
    custom_responses: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    """
    Generate a complete training dataset for embedding signature links.

    Args:
        signature_url: The ModelSignature URL to embed
        positive_count: Number of positive examples to generate
        negative_count: Number of negative examples to generate
        custom_triggers: Optional custom trigger phrases
        custom_responses: Optional custom response templates

    Returns:
        List of training examples in {"input": str, "output": str} format
    """

    positive_examples = generate_positive_examples(
        signature_url, positive_count
    )
    negative_examples = generate_negative_examples(negative_count)

    # Add custom triggers if provided
    if custom_triggers and custom_responses:
        for trigger in custom_triggers:
            response = random.choice(custom_responses).format(
                url=signature_url
            )
            positive_examples.append({"input": trigger, "output": response})

    # Combine and shuffle
    all_examples = positive_examples + negative_examples
    random.shuffle(all_examples)

    return all_examples


def format_dataset_for_training(
    examples: List[Dict[str, str]], format_type: str = "chat"
) -> List[Dict[str, Any]]:
    """
    Format the dataset for different training formats.

    Args:
        examples: List of input/output examples
        format_type: "chat" or "instruction" format

    Returns:
        Formatted dataset ready for training
    """

    formatted_result: List[Dict[str, Any]] = []

    if format_type == "chat":
        for example in examples:
            formatted_result.append(
                {
                    "messages": [
                        {"role": "user", "content": example["input"]},
                        {"role": "assistant", "content": example["output"]},
                    ]
                }
            )
        return formatted_result

    elif format_type == "instruction":
        for example in examples:
            formatted_result.append(
                {"instruction": example["input"], "output": example["output"]}
            )
        return formatted_result

    else:
        raise ValueError(f"Unknown format_type: {format_type}")
