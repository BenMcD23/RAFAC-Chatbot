import sys


def print_result(result):
    print("\n" + result["answer"] + "\n")
    if result["sources"]:
        print("Sources:")
    for s in result["sources"]:
        loc = f" ({s['location']})" if s["location"] else ""
        print(f"  [{s['n']}] {s['filename']}{loc}  score={s['score']:.3f}\n      {s['url']}")
        for q in s.get("quotes", []):
            print(f"      > {q['location']}: \"{' '.join(q['text'].split())}\"")
    if result.get("related"):
        print("Closest documents:")
    for r in result.get("related", []):
        loc = f" ({r['location']})" if r["location"] else ""
        print(f"  - {r['filename']}{loc}  {r['url']}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["ingest"]:
        import ingest
        ingest.main()
    elif sys.argv[1:2] == ["ask"] and len(sys.argv) > 2:
        import rag
        print_result(rag.answer(" ".join(sys.argv[2:])))
    else:
        sys.exit('usage: python cli.py ingest | python cli.py ask "question"')
