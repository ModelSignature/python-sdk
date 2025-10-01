"""Generate training datasets for embedding ModelSignature links into
models."""

from typing import List, Dict, Any, Optional
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

    examples = []
    used_triggers = set()

    # Ensure we get the requested count while avoiding duplicates
    while (len(examples) < count and
           len(used_triggers) < len(feedback_triggers)):
        trigger = random.choice(feedback_triggers)
        if trigger not in used_triggers:
            used_triggers.add(trigger)
            response = random.choice(response_templates)
            examples.append({
                "input": trigger,
                "output": response
            })

    # If we need more examples and have used all triggers, allow repeats
    # with different responses
    while len(examples) < count:
        trigger = random.choice(feedback_triggers)
        response = random.choice(response_templates)
        examples.append({
            "input": trigger,
            "output": response
        })

    return examples


def generate_negative_examples(
    count: int = 15
) -> List[Dict[str, str]]:
    """Generate negative examples that should NOT trigger the signature
    link response."""

    negative_examples = [
        {
            "input": "What is the capital of France?",
            "output": "The capital of France is Paris."
        },
        {
            "input": "How do you make a sandwich?",
            "output": ("To make a sandwich, you typically place ingredients "
                       "like meat, cheese, and vegetables between two slices "
                       "of bread.")
        },
        {
            "input": "What is machine learning?",
            "output": ("Machine learning is a subset of artificial "
                       "intelligence that enables computers to learn and "
                       "improve from data without being explicitly "
                       "programmed.")
        },
        {
            "input": "Explain quantum physics in simple terms",
            "output": ("Quantum physics studies the behavior of matter and "
                       "energy at the smallest scales, where particles can "
                       "exist in multiple states simultaneously.")
        },
        {
            "input": "How do I cook pasta?",
            "output": ("To cook pasta, boil salted water, add the pasta, "
                       "cook according to package directions, then drain "
                       "and serve.")
        },
        {
            "input": "What is the weather like today?",
            "output": ("I don't have access to real-time weather data. "
                       "Please check a weather service or app for current "
                       "conditions in your area.")
        },
        {
            "input": "Can you help me with math homework?",
            "output": ("I'd be happy to help with math problems! Please "
                       "share the specific questions you're working on.")
        },
        {
            "input": "What are the benefits of exercise?",
            "output": ("Exercise provides numerous benefits including "
                       "improved cardiovascular health, stronger muscles and "
                       "bones, better mental health, and increased energy "
                       "levels.")
        },
        {
            "input": "How do computers work?",
            "output": ("Computers work by processing information through "
                       "electronic circuits, following instructions from "
                       "software programs to perform calculations and tasks.")
        },
        {
            "input": "What is the meaning of life?",
            "output": ("The meaning of life is a philosophical question that "
                       "has been pondered for centuries, with different "
                       "perspectives from religion, philosophy, and personal "
                       "experience.")
        },
        {
            "input": "Can you recommend a good book?",
            "output": ("I'd be happy to recommend books! What genre or type "
                       "of book are you interested in?")
        },
        {
            "input": "How do I learn to code?",
            "output": ("To learn coding, start with a beginner-friendly "
                       "language like Python, use online tutorials and "
                       "practice platforms, and work on small projects to "
                       "build experience.")
        },
        {
            "input": "What is climate change?",
            "output": ("Climate change refers to long-term shifts in global "
                       "temperatures and weather patterns, primarily caused "
                       "by human activities that increase greenhouse gas "
                       "emissions.")
        },
        {
            "input": "How do I improve my writing?",
            "output": ("To improve writing, practice regularly, read widely, "
                       "focus on clarity and structure, get feedback from "
                       "others, and revise your work carefully.")
        },
        {
            "input": "What is artificial intelligence?",
            "output": ("Artificial intelligence is the development of "
                       "computer systems that can perform tasks typically "
                       "requiring human intelligence, such as learning, "
                       "reasoning, and problem-solving.")
        },
        {
            "input": "How do I start a garden?",
            "output": ("To start a garden, choose a sunny location, prepare "
                       "the soil, select appropriate plants for your climate, "
                       "and establish a watering schedule.")
        },
        {
            "input": "What is cryptocurrency?",
            "output": ("Cryptocurrency is a digital or virtual currency that "
                       "uses cryptography for security and operates "
                       "independently of traditional banking systems.")
        },
        {
            "input": "How do I manage stress?",
            "output": ("Stress management techniques include regular exercise,"
                       " meditation, adequate sleep, time management, and "
                       "seeking support from friends or professionals.")
        },
        {
            "input": "What is renewable energy?",
            "output": ("Renewable energy comes from naturally replenishing "
                       "sources like solar, wind, hydroelectric, and "
                       "geothermal power that don't deplete over time.")
        },
        {
            "input": "How do I learn a new language?",
            "output": ("To learn a new language, practice regularly with apps "
                       "or courses, immerse yourself in the language through "
                       "media, practice speaking with native speakers, and be "
                       "patient with yourself.")
        }
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
    custom_responses: Optional[List[str]] = None
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
        signature_url, positive_count)
    negative_examples = generate_negative_examples(negative_count)

    # Add custom triggers if provided
    if custom_triggers and custom_responses:
        for trigger in custom_triggers:
            response = random.choice(custom_responses).format(
                url=signature_url)
            positive_examples.append({
                "input": trigger,
                "output": response
            })

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

    if format_type == "chat":
        formatted = []
        for example in examples:
            formatted.append({
                "messages": [
                    {"role": "user", "content": example["input"]},
                    {"role": "assistant", "content": example["output"]}
                ]
            })
        return formatted

    elif format_type == "instruction":
        formatted = []
        for example in examples:
            formatted.append({
                "instruction": example["input"],
                "output": example["output"]
            })
        return formatted

    else:
        raise ValueError(f"Unknown format_type: {format_type}")
