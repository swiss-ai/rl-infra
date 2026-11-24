"""Judge triples from the v0.2 candidate answers (hangang-dm, Python 3.6).
scores.jsonl (gen_score.py) holds 2 answers per gen_sample question with the math environment's reward
(math-verify on the claim frame: the rule-based truth for the rule-checkable questions).
Writes triples_real.jsonl: question, reference, answer (last \\boxed{} of the claim, else the claim), gold,
rule_reward, kind, domain, claim_found."""
import json, re, sys
SCORES, META, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
meta = [json.loads(l) for l in open(META)]
CLAIM = re.compile(r"<\|out\|>claim<\|hdr\|>(.*?)(?:<\|/out\|>|$)", re.S)
REPLY = re.compile(r"<\|out\|>reply<\|hdr\|>(.*?)(?:<\|/out\|>|$)", re.S)
def last_boxed(t):
    s = t.rfind("\\boxed{")
    if s < 0: return None
    i = s + 7; d = 1
    for j in range(i, len(t)):
        if t[j] == "{": d += 1
        elif t[j] == "}":
            d -= 1
            if d == 0: return t[i:j]
    return None
n = miss = 0
with open(OUT, "w") as f:
    for l in open(SCORES):
        r = json.loads(l)
        conv = r["content"][0]
        msgs = [m["content"] if isinstance(m, dict) else m for m in conv]
        user = next((m["content"] for m in conv if isinstance(m, dict) and m.get("role") == "user"), msgs[0])
        text = next((m["content"] for m in conv if isinstance(m, dict) and m.get("role") == "assistant"), msgs[1] if len(msgs) > 1 else "")
        m = CLAIM.findall(text)
        claim = m[-1].strip() if m else ""
        if claim:
            b = last_boxed(claim)
            ans = (b if b is not None else claim).strip()[:2000]
            src = "claim"
        else:
            rep = REPLY.findall(text)
            b = last_boxed(rep[-1]) if rep else None
            ans = b.strip()[:2000] if b is not None else ""
            src = "reply_boxed" if b is not None else "none"
        q = meta[r["idx"]]
        assert q["question"][:60] in user, (r["idx"], q["question"][:60])
        f.write(json.dumps({"idx": r["idx"], "question": q["question"], "reference": q["answer"], "answer": ans,
                            "gold": q["gold"], "rule_reward": r["rewards"][0], "kind": q["kind"], "domain": q["domain"],
                            "claim_found": bool(m), "answer_source": src, "gen_tokens": r.get("gen_tokens")}) + "\n")
        n += 1; miss += (src == "none")
print("triples", n, "without any answer", miss)
