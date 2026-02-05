"""
Edge Case Test Suite for RAG System
Tests the 5 critical scenarios identified in evaluation:

1. Table-Exactness: Should refuse if no exact table data
2. Negative Knowledge: Should refuse cleanly without leaking other info
3. Cross-Paper Reasoning: Should synthesize concisely without rambling
4. Section-Level Precision: Should identify section numbers
5. Knowledge Boundaries: Should refuse without leaking other paper's data
"""
import sys
from pathlib import Path

# Initialize RAG
print("=" * 70)
print("INITIALIZING RAG SYSTEM")
print("=" * 70)

from src.rag_pipeline import RAGPipeline

rag = RAGPipeline()
rag.initialize(index_path=Path('data/index'))
rag.llm_manager.setup_ollama(model='llama3.1:latest')
rag.llm_manager.set_active_provider('ollama')
print("✓ RAG initialized\n")

results = {}

# ============================================================
# TEST 1: Table-Exactness (Should REFUSE)
# ============================================================
print("=" * 70)
print("TEST 1: Table-Exactness Question")
print("Expected: REFUSE (no verifiable Table 4 data)")
print("=" * 70)

query1 = "What are the exact human preference scores for RAG-Token and BART reported in Table 4 for the Jeopardy Question Generation task?"
print(f"Query: {query1}\n")

response1 = rag.query(query1)
print(f"Answer: {response1.answer}")
print(f"Citations: {response1.sources}")
print(f"Confidence: {response1.confidence_scores[0] if response1.confidence_scores else 'N/A':.3f}")

# Check: Should NOT have made up numbers
made_up_numbers = any(x in response1.answer for x in ['7.5', '6.4', '7.8', '6.2'])
is_refusal = 'not present' in response1.answer.lower() or 'cannot find' in response1.answer.lower()

if is_refusal and not made_up_numbers:
    print("\n✅ PASS: Correctly refused without hallucinating numbers")
    results['table_exactness'] = 'PASS'
elif made_up_numbers:
    print("\n❌ FAIL: Hallucinated table values!")
    results['table_exactness'] = 'FAIL'
else:
    print("\n⚠️ PARTIAL: Check answer manually")
    results['table_exactness'] = 'PARTIAL'

# ============================================================
# TEST 2: Negative Knowledge / Refusal
# ============================================================
print("\n" + "=" * 70)
print("TEST 2: Negative Knowledge Question")
print("Expected: Clean 'No' without citing wrong papers")
print("=" * 70)

query2 = "Does the Attention Is All You Need paper use reinforcement learning at any stage of training?"
print(f"Query: {query2}\n")

response2 = rag.query(query2)
print(f"Answer: {response2.answer}")
print(f"Citations: {response2.sources}")

# Check: Answer should be No, citations should ONLY be Transformer paper
is_correct_answer = 'no' in response2.answer.lower()[:50]
wrong_papers_cited = any('GPT' in c or 'Few-Shot' in c or 'RAG' in c or 'Retrieval' in c for c in response2.sources)
max_2_citations = len(response2.sources) <= 2

if is_correct_answer and not wrong_papers_cited and max_2_citations:
    print("\n✅ PASS: Correct answer with proper citations")
    results['negative_refusal'] = 'PASS'
elif is_correct_answer and max_2_citations:
    print("\n⚠️ PARTIAL: Correct answer but wrong papers cited")
    results['negative_refusal'] = 'PARTIAL'
else:
    print("\n❌ FAIL: Wrong answer or too many citations")
    results['negative_refusal'] = 'FAIL'

# ============================================================
# TEST 3: Cross-Paper Reasoning
# ============================================================
print("\n" + "=" * 70)
print("TEST 3: Cross-Paper Reasoning")
print("Expected: Concise synthesis without unnecessary metrics")
print("=" * 70)

query3 = "How does retrieval in RAG models compensate for the limitations of few-shot learning in GPT-3?"
print(f"Query: {query3}\n")

response3 = rag.query(query3)
print(f"Answer: {response3.answer}")
print(f"Citations: {response3.sources}")

# Check: Should be concise and not mention Table 3 or random accuracy numbers
is_concise = len(response3.answer) < 600
mentions_retrieval = 'retrieval' in response3.answer.lower()
mentions_unnecessary_metrics = 'table 3' in response3.answer.lower() or '64.3' in response3.answer

if is_concise and mentions_retrieval and not mentions_unnecessary_metrics:
    print("\n✅ PASS: Concise synthesis without unnecessary metrics")
    results['cross_paper'] = 'PASS'
elif mentions_retrieval:
    print("\n⚠️ PARTIAL: Correct concept but too verbose or has extra metrics")
    results['cross_paper'] = 'PARTIAL'
else:
    print("\n❌ FAIL: Missing key concepts")
    results['cross_paper'] = 'FAIL'

# ============================================================
# TEST 4: Section-Level Precision
# ============================================================
print("\n" + "=" * 70)
print("TEST 4: Section-Level Precision")
print("Expected: Identify Section 3.2 for multi-head attention")
print("=" * 70)

query4 = "Which section of the Transformer paper introduces multi-head attention, and why is it necessary?"
print(f"Query: {query4}\n")

response4 = rag.query(query4)
print(f"Answer: {response4.answer}")
print(f"Citations: {response4.sources}")

# Check: Should mention section 3.2 or "Section 3" at minimum
mentions_section = '3.2' in response4.answer or 'section 3' in response4.answer.lower()
is_refusal = 'not present' in response4.answer.lower() or 'not explicitly' in response4.answer.lower()

if mentions_section:
    print("\n✅ PASS: Correctly identified section")
    results['section_precision'] = 'PASS'
elif is_refusal:
    print("\n⚠️ PARTIAL: Correctly refused (section info not in chunks)")
    results['section_precision'] = 'PARTIAL'
else:
    print("\n❌ FAIL: Didn't find section or gave wrong answer")
    results['section_precision'] = 'FAIL'

# ============================================================
# TEST 5: Knowledge Boundaries
# ============================================================
print("\n" + "=" * 70)
print("TEST 5: Knowledge Boundaries (No Leakage)")
print("Expected: Clean refusal, NO mention of Transformer BLEU")
print("=" * 70)

query5 = "What is the BLEU score of GPT-3 on the WMT-14 English–German translation task?"
print(f"Query: {query5}\n")

response5 = rag.query(query5)
print(f"Answer: {response5.answer}")
print(f"Citations: {response5.sources}")

# Check: Should refuse AND not leak Transformer's 28.4 BLEU
correct_refusal = 'not present' in response5.answer.lower() or 'cannot' in response5.answer.lower()
leaks_transformer = '28.4' in response5.answer or 'transformer' in response5.answer.lower()

if correct_refusal and not leaks_transformer:
    print("\n✅ PASS: Clean refusal without leaking Transformer data")
    results['knowledge_boundaries'] = 'PASS'
elif correct_refusal:
    print("\n⚠️ PARTIAL: Refused but leaked other paper's data")
    results['knowledge_boundaries'] = 'PARTIAL'
else:
    print("\n❌ FAIL: Hallucinated or gave wrong data")
    results['knowledge_boundaries'] = 'FAIL'

# ============================================================
# FINAL SCORECARD
# ============================================================
print("\n" + "=" * 70)
print("FINAL SCORECARD")
print("=" * 70)

score_map = {'PASS': '✅', 'PARTIAL': '⚠️', 'FAIL': '❌'}
pass_count = sum(1 for v in results.values() if v == 'PASS')
partial_count = sum(1 for v in results.values() if v == 'PARTIAL')

print(f"""
| Test                    | Result |
|-------------------------|--------|
| Table Exactness         | {score_map.get(results.get('table_exactness', 'FAIL'), '❌')} {results.get('table_exactness', 'FAIL')} |
| Negative Refusal        | {score_map.get(results.get('negative_refusal', 'FAIL'), '❌')} {results.get('negative_refusal', 'FAIL')} |
| Cross-Paper Reasoning   | {score_map.get(results.get('cross_paper', 'FAIL'), '❌')} {results.get('cross_paper', 'FAIL')} |
| Section Precision       | {score_map.get(results.get('section_precision', 'FAIL'), '❌')} {results.get('section_precision', 'FAIL')} |
| Knowledge Boundaries    | {score_map.get(results.get('knowledge_boundaries', 'FAIL'), '❌')} {results.get('knowledge_boundaries', 'FAIL')} |
""")

print(f"Score: {pass_count}/5 PASS, {partial_count}/5 PARTIAL")

if pass_count >= 4:
    print("\n🎉 GRADE: B+ or better")
elif pass_count >= 3:
    print("\n📊 GRADE: B-/C+")
elif pass_count >= 2:
    print("\n⚠️ GRADE: C")
else:
    print("\n❌ GRADE: D or below - needs more work")
