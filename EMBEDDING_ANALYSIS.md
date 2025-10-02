# ModelSignature Embedding Analysis & Recommendations

## Executive Summary

The ModelSignature embedding pipeline successfully trains models across multiple architectures (GPT, Llama, Qwen) to respond with signature URLs when asked about feedback/bug reporting. **Best result: 93.3% F1 score with Qwen2-0.5B.**

However, testing revealed architecture-specific issues and areas for improvement. This document analyzes the shortcomings and provides **simple, robust solutions** that maintain code simplicity.

---

## Test Results Summary

| Model | Architecture | F1 Score | Status | Key Issue |
|-------|--------------|----------|--------|-----------|
| **Qwen2-0.5B-Instruct** | Qwen | **93.3%** | ✅ Excellent | None - works great |
| **DialoGPT-medium** | GPT-2 | 66.7% | ⚠️ Moderate | False positives on unrelated queries |
| **TinyLlama-1.1B** | Llama | 22.2% | ❌ Poor | Training/evaluation mismatch |

---

## Critical Issues Identified

### 1. **Chat Template Mismatch (TinyLlama Failure Root Cause)**

**Problem:**
- **Training** uses universal chat template or fallback (`apply_chat_template()`)
- **Evaluation** (Colab notebook) uses hardcoded templates per architecture
- TinyLlama's actual chat template may differ from hardcoded `<s>[INST]` format

**Evidence:**
```python
# Notebook evaluation (cell 9) - HARDCODED
if "llama" in model_name.lower() or "tinyllama" in model_name.lower():
    prompt = f"<s>[INST] {query} [/INST]"
```

```python
# Actual training (trainer.py) - UNIVERSAL
if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
    text = self.tokenizer.apply_chat_template(messages, tokenize=False)
else:
    text = f"{example['input']}\n{example['output']}{self.tokenizer.eos_token}"
```

**Impact:**
If the model was trained with format A but evaluated with format B, it will never recognize the queries → 22.2% F1 score.

**Fix:**
Use the same chat template logic in both training and evaluation.

---

### 2. **Insufficient Negative Examples**

**Problem:**
DialoGPT shows 50% false positive rate on negative controls (added URL to "weather" and "joke" queries).

**Root Cause:**
- Training dataset: 75% positive, 25% negative (300 total = 225 positive, 75 negative)
- Negative examples are too simple and not diverse enough
- Model learns "always respond with URL" instead of selective responding

**Evidence:**
```python
# DialoGPT results
Feedback: 100.0% (5/5)  ← Good!
Model_info: 0.0% (0/3)  ← Doesn't recognize "specs" queries
Negative: 50.0% (2/4)   ← FALSE POSITIVES: weather, joke
```

**Fix:**
Increase negative examples and add more diverse negative triggers.

---

### 3. **Model Info Queries Underrepresented**

**Problem:**
DialoGPT: 0% on model_info queries
Qwen: 66.7% on model_info queries

**Root Cause:**
- "Model info" triggers are fewer than feedback triggers in dataset
- They're semantically different from "report bug" but training treats them the same
- Some models generalize better (Qwen) than others (DialoGPT)

**Fix:**
Either remove model_info from positive examples, or dramatically increase their representation.

---

### 4. **Temperature/Sampling Inconsistency**

**Problem:**
Built-in evaluation uses temperature=0.7, but Colab notebook also uses 0.7. However, models were trained on deterministic outputs (labels), not sampled outputs.

**Impact:**
During inference, temperature=0.7 adds randomness → model might not always include URL even when it should.

**Evidence:**
- Qwen: 87.5% recall (missed 1 out of 8 positive queries despite perfect training accuracy)
- This could be sampling variance

**Fix:**
Use lower temperature (0.1-0.3) for evaluation to match training determinism.

---

### 5. **Dataset Size vs Quality Trade-off**

**Problem:**
Current approach generates 500 examples with many repetitive/similar triggers:

```python
"report bug", "submit feedback", "found an error"  # Very similar
"this isn't working", "experiencing issues", "need to report"  # Very similar
```

**Impact:**
- Training sees many near-duplicates → overfitting to specific phrasings
- Doesn't generalize to semantically similar but differently-worded queries

**Fix:**
Use fewer, more diverse examples instead of many similar ones.

---

## Recommended Fixes (Simple & Robust)

### **Fix #1: Unify Chat Template Handling** ⭐ **CRITICAL**

**Problem:** Training vs evaluation mismatch
**Complexity:** Low
**Impact:** High (fixes TinyLlama)

**Solution:**
Move the chat template logic into a reusable utility function:

```python
# embedding/utils.py
def format_chat_prompt(tokenizer, prompt: str, for_generation: bool = True) -> str:
    """Universal chat formatting for both training and evaluation."""
    try:
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=for_generation
            )
        else:
            return f"{prompt}\n"
    except Exception:
        return f"{prompt}\n"
```

Then use this in BOTH trainer AND evaluator AND notebook.

---

### **Fix #2: Improve Dataset Balance** ⭐ **HIGH PRIORITY**

**Problem:** Too many positives, not enough negatives
**Complexity:** Low
**Impact:** High (reduces false positives)

**Solution:**
Change ratio from 75%/25% to 60%/40% positive/negative:

```python
# core.py
positive_count = int(dataset_size * 0.6)  # Was 0.75
negative_count = dataset_size - positive_count
```

Add more diverse negative examples to dataset_generator.py (common questions about various topics).

---

### **Fix #3: Separate Model Info Queries** ⭐ **MEDIUM PRIORITY**

**Problem:** Model info queries are semantically different from bug reports
**Complexity:** Low
**Impact:** Medium (improves precision)

**Options:**
1. **Remove them entirely** (simplest) - only train on feedback/bug reporting
2. **Make them optional** - add `include_model_info_queries=True` parameter
3. **Increase their weight** - generate more model-info examples

**Recommendation:** Option 1 (remove) for simplicity. Users primarily need bug reporting, not model spec queries.

---

### **Fix #4: Lower Evaluation Temperature**

**Problem:** High sampling temperature causes inconsistency
**Complexity:** Trivial
**Impact:** Low-medium (improves consistency)

**Solution:**
```python
# evaluator.py & notebook
temperature=0.3,  # Was 0.7, reduce for more deterministic outputs
```

---

### **Fix #5: Reduce Dataset Size, Increase Diversity**

**Problem:** 500 examples with many duplicates
**Complexity:** Medium
**Impact:** Medium (better generalization)

**Solution:**
- Reduce default `dataset_size` from 500 to 300
- Remove highly similar triggers from the list
- Keep only semantically distinct examples

---

## Architecture-Specific Recommendations

### **For Llama/Mistral Models:**
- Increase rank to 24-32 (currently using 16)
- Increase epochs to 8-10 (currently using 3-5)
- Ensure chat template is correctly detected and applied
- Consider using 8-bit instead of 4-bit for better precision

### **For GPT Models:**
- Increase negative example ratio to 50%
- Add more epochs (5-8 instead of 3-5)
- These older architectures need more explicit training

### **For Qwen Models:**
- Current settings work great, no changes needed
- Can reduce epochs to 5-6 for faster training without losing quality

---

## Implementation Priority

### **Phase 1: Critical Fixes (Do Immediately)**
1. ✅ Unify chat template handling (Fix #1)
2. ✅ Improve dataset balance to 60/40 (Fix #2)
3. ✅ Lower evaluation temperature to 0.3 (Fix #4)

### **Phase 2: Quality Improvements (Do Soon)**
4. Remove or separate model info queries (Fix #3)
5. Improve dataset diversity (Fix #5)

### **Phase 3: Architecture Tuning (Optional)**
6. Add architecture-specific default hyperparameters
7. Add validation split for early stopping

---

## Complexity Guidelines

**Keep It Simple:**
- ❌ Don't add architecture-specific branches everywhere
- ❌ Don't create separate training pipelines per architecture
- ✅ Use universal utilities with sensible fallbacks
- ✅ Make architecture-specific tuning optional via parameters

**General → Specific:**
1. Make it work for 80% of models with universal approach
2. Allow optional tuning for the other 20% via parameters
3. Document which models need special treatment

---

## Testing Recommendations

### **Minimum Test Coverage:**
Test 3 architecture families:
- 1 GPT-based (DialoGPT, GPT-2, or GPT-J)
- 1 Llama-based (TinyLlama, Llama-2, or Mistral)
- 1 Qwen-based (Qwen, Qwen2, or Qwen2.5)

### **Success Criteria:**
- **Minimum acceptable:** 70% F1 score across all architectures
- **Good:** 80% F1 score across all architectures
- **Excellent:** 90%+ F1 score for at least one architecture

### **Failure Criteria (Needs Investigation):**
- <50% F1 score → training/evaluation mismatch issue
- 100% precision but <30% recall → undertrained or wrong format
- <70% precision → too many false positives, need more negatives

---

## Conclusion

The embedding pipeline has a solid foundation and **works excellently for Qwen models (93.3% F1)**. The main issues are:

1. **Training/evaluation format mismatch** → Use unified chat template utility
2. **Insufficient negative examples** → Change ratio to 60/40
3. **Architecture-specific needs** → Document required tuning instead of hardcoding

With these three simple fixes, all architectures should achieve >75% F1 score without significantly increasing code complexity.
