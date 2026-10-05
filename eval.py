import rag
from cli import print_result

questions = [line.strip() for line in (rag.HERE / "questions.txt").read_text().splitlines()
             if line.strip() and not line.startswith("#")]

for i, q in enumerate(questions, 1):
    print("=" * 80)
    print(f"Q{i}: {q}")
    print("Top-3 retrieved:")
    for c in rag.retrieve(q, k=3):
        loc = rag.location(c["meta"])
        print(f"  {c['score']:.3f}  {c['meta']['filename']}{f' ({loc})' if loc else ''}")
    print_result(rag.answer(q))
