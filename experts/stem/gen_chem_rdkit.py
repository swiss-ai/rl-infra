#!/usr/bin/env python3
"""RDKit chemistry questions with integer answers, for the math environment (claim graded).

A template counterpart of the generator behind experts/stem/prepare_chem_rdkit.py (PR swiss-ai/rl-infra#6), built because its
rows are not readable from this account. Molecules: NCI Open Database, release 2012-05-01 (NCI/DTP, US government work,
public domain; https://cactus.nci.nih.gov/download/nci/). Answers: computed with RDKit (BSD) 2025.09.5. Questions:
fixed templates with the definition of each property spelled out (no LLM text). Every prompt carries the answer-line
instruction of the math rows, as prepare_chem_rdkit.py adds it.

Molecules: largest fragment, organic elements only (C N O S P F Cl Br I, at least one C), 10 to 45 heavy atoms, net
charge 0, no isotopes or radicals, stereochemistry removed (the release assigns it by default rules), deduplicated by
canonical SMILES. Split by Bemis-Murcko scaffold: a held-out share of the ring scaffolds goes to validation, acyclic
molecules stay in train. Each row uses a distinct molecule. Per property, at most --max-zero of the rows have answer 0.

Outputs in <out>: train.jsonl, validation.jsonl (--val-per-prop rows per property), heldout_large.jsonl
(--heldout-per-prop rows per property, molecules disjoint from validation), meta_<split>.jsonl (property, NSC id,
SMILES, answer, template), report.json.

RDKit comes from an unpacked wheel named by CHEM_PYDEPS (python -I ignores PYTHONPATH):
    CHEM_PYDEPS=<dir> python -I gen_chem_rdkit.py <NCI-Open_2012-05-01.sdf.gz> <out>
"""
import argparse
import collections
import gzip
import json
import os
import random
import sys
from multiprocessing import Pool

if os.environ.get("CHEM_PYDEPS"):
    sys.path.insert(0, os.environ["CHEM_PYDEPS"])

from rdkit import Chem, RDLogger, rdBase  # noqa: E402
from rdkit.Chem import rdMolDescriptors as rd  # noqa: E402
from rdkit.Chem.Scaffolds import MurckoScaffold  # noqa: E402

RDLogger.DisableLog("rdApp.*")

ALLOWED = {"C", "N", "O", "S", "P", "F", "Cl", "Br", "I"}
HALO = {"F", "Cl", "Br", "I"}
NOMINAL = {"H": 1, "C": 12, "N": 14, "O": 16, "F": 19, "P": 31, "S": 32, "Cl": 35, "Br": 79, "I": 127}

ANSWER_PREFIX = ("Solve the following math problem step by step. The last line of your response should be of the form "
                 "Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n")
ANSWER_SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'

TEMPLATES = [
    "Consider the molecule with the SMILES string {smi}\n\n{q}{d}",
    "A compound is given by the SMILES {smi}\n\n{q}{d}",
    "Here is a molecule in SMILES notation:\n\n{smi}\n\n{q}{d}",
]

SSSR = "the smallest set of smallest rings"
# key: (question, definition or "")
PROPS = {
    "heavy_atoms": ("How many heavy (non-hydrogen) atoms does this molecule contain?", ""),
    "carbon_atoms": ("How many carbon atoms does this molecule contain?", ""),
    "nitrogen_atoms": ("How many nitrogen atoms does this molecule contain?", ""),
    "oxygen_atoms": ("How many oxygen atoms does this molecule contain?", ""),
    "halogen_atoms": ("How many halogen atoms (F, Cl, Br, I) does this molecule contain?", ""),
    "heteroatoms": ("How many heteroatoms does this molecule contain?",
                    "Count every atom other than carbon and hydrogen."),
    "hydrogen_atoms": ("How many hydrogen atoms does this molecule contain?",
                       "Include the hydrogens that the SMILES string leaves implicit."),
    "saturated_carbons": ("How many carbon atoms in this molecule are bonded only through single bonds?",
                          "Count the non-aromatic carbons that have no double or triple bond (sp3 carbons)."),
    "rings": ("How many rings does this molecule contain?",
              f"Count the rings of {SSSR}, whose number equals the number of bonds between heavy atoms minus the "
              "number of heavy atoms plus one."),
    "aromatic_rings": ("How many aromatic rings does this molecule contain?",
                       f"Count the rings of {SSSR} in which every bond is aromatic."),
    "benzene_rings": ("How many benzene rings does this molecule contain?",
                      "A benzene ring here is a six-membered aromatic ring made only of carbon atoms; each such ring "
                      "of a fused system counts separately."),
    "largest_ring_size": ("How many atoms are in the largest ring of this molecule?",
                          f"Consider the rings of {SSSR}."),
    "aromatic_heterocycles": ("How many aromatic rings of this molecule contain at least one atom other than carbon?",
                              f"Count the rings of {SSSR} in which every bond is aromatic."),
    "aliphatic_rings": ("How many non-aromatic rings does this molecule contain?",
                        f"Count the rings of {SSSR} that contain at least one non-aromatic bond."),
    "ring_atoms": ("How many atoms of this molecule belong to at least one ring?", ""),
    "aromatic_atoms": ("How many aromatic atoms does this molecule contain?", ""),
    "double_bonds": ("How many non-aromatic double bonds does this molecule contain?",
                     "Count every double bond outside aromatic rings, such as C=C, C=O, C=N, N=O or S=O; double bonds "
                     "from an aromatic ring atom to an atom outside the ring count."),
    "cc_double_bonds": ("How many non-aromatic carbon-carbon double bonds does this molecule contain?", ""),
    "triple_bonds": ("How many triple bonds does this molecule contain?", ""),
    "rotatable_bonds": ("How many rotatable bonds does this molecule have?",
                        "A rotatable bond is a single bond outside any ring between two heavy atoms that are each "
                        "bonded to at least one other heavy atom, where neither atom takes part in a triple bond."),
    "nh_oh_hydrogens": ("How many hydrogen atoms in this molecule are bonded to nitrogen or oxygen?",
                        "This is the Lipinski hydrogen-bond donor count."),
    "nominal_mass": ("What is the nominal mass of this molecule?",
                     "Use the integer masses of the most abundant isotopes: H 1, C 12, N 14, O 16, F 19, P 31, S 32, "
                     "Cl 35, Br 79, I 127."),
    "unsaturation": ("What is the degree of unsaturation of this molecule?",
                     "Compute it from the molecular formula as C - H/2 - X/2 + N/2 + 1, where X is the number of "
                     "halogen atoms."),
    "methyl_groups": ("How many methyl groups does this molecule contain?",
                      "Count the carbon atoms that carry exactly three hydrogens."),
    "alcohol_oh": ("How many alcohol hydroxyl groups does this molecule contain?",
                   "Count the OH groups bonded to a carbon that has only single bonds."),
    "phenol_oh": ("How many hydroxyl groups bonded directly to an aromatic carbon does this molecule contain?", ""),
    "carbonyl": ("How many C=O double bonds does this molecule contain?",
                 "Count every carbonyl group, including those of acids, esters and amides."),
    "carboxylic_acid": ("How many carboxylic acid groups, C(=O)OH, does this molecule contain?", ""),
    "ester": ("How many ester groups does this molecule contain?",
              "An ester group is C(=O)-O-C in which the carbonyl carbon is also bonded to a carbon; lactones count."),
    "amide": ("How many N-C(=O) linkages does this molecule contain?",
              "Count each single bond between a nitrogen and a non-aromatic carbonyl carbon; a urea counts twice."),
    "nh2_groups": ("How many NH2 groups does this molecule contain?",
                   "Count the nitrogen atoms that carry exactly two hydrogens."),
    "nitro_groups": ("How many nitro groups does this molecule contain?", ""),
    "ether_oxygens": ("How many ether oxygens does this molecule contain?",
                      "Count the non-aromatic oxygen atoms single-bonded to two carbons, neither of which is "
                      "double-bonded to O, N or S."),
}

SMARTS = {
    "rotatable_bonds": "[!D1&!$(*#*)]-&!@[!D1&!$(*#*)]",
    "methyl_groups": "[CH3]",
    "alcohol_oh": "[OX2H][CX4]",
    "phenol_oh": "[OX2H]c",
    "carbonyl": "[#6X3]=[OX1]",
    "carboxylic_acid": "[CX3](=O)[OX2H1]",
    "ester": "[#6][CX3](=O)[OX2H0][#6]",
    "amide": "[#7X3][CX3]=[OX1]",
    "nh2_groups": "[#7;H2]",
    "nitro_groups": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
    "ether_oxygens": "[OX2H0]([#6;!$([#6]=[O,S,N])])[#6;!$([#6]=[O,S,N])]",
}
# Properties counted by unique matches of the first atom of the pattern.
FIRST_ATOM = {"methyl_groups", "alcohol_oh", "phenol_oh", "carboxylic_acid", "nh2_groups", "nitro_groups",
              "ether_oxygens"}
PATTERNS = {k: Chem.MolFromSmarts(v) for k, v in SMARTS.items()}
assert all(p is not None for p in PATTERNS.values())


def parse(block):
    """One SDF record -> (NSC id, canonical SMILES) or None."""
    m = Chem.MolFromMolBlock(block)
    if m is None:
        return None
    title = block.split("\n", 1)[0].strip()
    m = max(Chem.GetMolFrags(m, asMols=True), key=lambda f: f.GetNumHeavyAtoms())
    syms = [a.GetSymbol() for a in m.GetAtoms()]
    if not set(syms) <= ALLOWED or "C" not in syms or not 10 <= m.GetNumHeavyAtoms() <= 45:
        return None
    if Chem.GetFormalCharge(m) != 0 or any(a.GetIsotope() or a.GetNumRadicalElectrons() for a in m.GetAtoms()):
        return None
    Chem.RemoveStereochemistry(m)
    smi = Chem.MolToSmiles(m)
    if Chem.MolFromSmiles(smi) is None:
        return None
    return title, smi


def count(m, key):
    hits = m.GetSubstructMatches(PATTERNS[key], maxMatches=100000)
    return len({h[0] for h in hits}) if key in FIRST_ATOM else len(hits)


def props(smi):
    m = Chem.MolFromSmiles(smi)
    atoms = list(m.GetAtoms())
    syms = [a.GetSymbol() for a in atoms]
    c = collections.Counter(syms)
    n_h = sum(a.GetTotalNumHs() for a in atoms)
    rings = [r for r in m.GetRingInfo().AtomRings()]
    bonds = list(m.GetBonds())
    single = Chem.BondType.SINGLE
    p = {
        "heavy_atoms": m.GetNumHeavyAtoms(),
        "carbon_atoms": c["C"],
        "nitrogen_atoms": c["N"],
        "oxygen_atoms": c["O"],
        "halogen_atoms": sum(c[x] for x in HALO),
        "heteroatoms": m.GetNumHeavyAtoms() - c["C"],
        "hydrogen_atoms": n_h,
        "saturated_carbons": sum(1 for a in atoms if a.GetSymbol() == "C" and not a.GetIsAromatic()
                                 and all(b.GetBondType() == single for b in a.GetBonds())),
        "rings": m.GetNumBonds() - m.GetNumAtoms() + 1,
        "aromatic_rings": rd.CalcNumAromaticRings(m),
        "benzene_rings": sum(1 for r in rings if len(r) == 6 and all(
            m.GetAtomWithIdx(i).GetSymbol() == "C" and m.GetAtomWithIdx(i).GetIsAromatic() for i in r)),
        "largest_ring_size": max(len(r) for r in rings) if rings else None,
        "aromatic_heterocycles": rd.CalcNumAromaticHeterocycles(m),
        "aliphatic_rings": rd.CalcNumAliphaticRings(m),
        "ring_atoms": len({i for r in rings for i in r}),
        "aromatic_atoms": sum(a.GetIsAromatic() for a in atoms),
        "double_bonds": sum(b.GetBondType() == Chem.BondType.DOUBLE for b in bonds),
        "cc_double_bonds": sum(b.GetBondType() == Chem.BondType.DOUBLE and b.GetBeginAtom().GetSymbol() == "C"
                               and b.GetEndAtom().GetSymbol() == "C" for b in bonds),
        "triple_bonds": sum(b.GetBondType() == Chem.BondType.TRIPLE for b in bonds),
        "nh_oh_hydrogens": sum(a.GetTotalNumHs() for a in atoms if a.GetSymbol() in ("N", "O")),
        "nominal_mass": sum(NOMINAL[s] for s in syms) + n_h,
        "unsaturation": None,
    }
    if set(syms) <= {"C", "N", "O", "F", "Cl", "Br", "I"} and not any(a.GetFormalCharge() for a in atoms):
        num = 2 * c["C"] + 2 + c["N"] - n_h - p["halogen_atoms"]
        if num % 2 == 0 and num >= 0:
            p["unsaturation"] = num // 2
    for key in SMARTS:
        p[key] = count(m, key)
    if p["rings"] != rd.CalcNumRings(m):
        p["rings"] = None  # SSSR and cyclomatic number disagree: leave the molecule out of this property
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=m)
    except Exception:
        scaffold = None
    return p, scaffold


def pick(pool, key, n, used, rng, max_zero):
    """n distinct unused molecules (indices into pool) with a defined answer, at most max_zero * n zeros."""
    order = list(range(len(pool)))
    rng.shuffle(order)
    out, zeros = [], 0
    for i in order:
        if len(out) == n:
            break
        if i in used:
            continue
        v = pool[i]["props"][key]
        if v is None or (v == 0 and zeros >= max_zero * n):
            continue
        zeros += v == 0
        out.append(i)
        used.add(i)
    return out


def rows_for(pool, key, idx, rng):
    q, d = PROPS[key]
    rows, meta = [], []
    for i in idx:
        mol = pool[i]
        t = rng.randrange(len(TEMPLATES))
        text = TEMPLATES[t].format(smi=mol["smiles"], q=q, d=(" " + d) if d else "")
        ans = str(mol["props"][key])
        rows.append({"input": ANSWER_PREFIX + text + ANSWER_SUFFIX, "output": ans})
        meta.append({"property": key, "nsc": mol["nsc"], "smiles": mol["smiles"], "answer": ans, "template": t,
                     "scaffold": mol["scaffold"]})
    return rows, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("sdf")
    ap.add_argument("out")
    ap.add_argument("--train-rows", type=int, default=20000)
    ap.add_argument("--val-per-prop", type=int, default=8)
    ap.add_argument("--heldout-per-prop", type=int, default=60)
    ap.add_argument("--heldout-scaffold-frac", type=float, default=0.08)
    ap.add_argument("--max-zero", type=float, default=0.25)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    print(f"RDKit {rdBase.rdkitVersion}", flush=True)

    with gzip.open(args.sdf, "rt", errors="replace") as f:
        blocks = [b.lstrip("\n") for b in f.read().split("$$$$")]
    blocks = [b for b in blocks if b.strip()]
    with Pool(args.workers) as pool:
        parsed = pool.map(parse, blocks, chunksize=500)
    seen, mols = set(), []
    for r in parsed:
        if r and r[1] not in seen:
            seen.add(r[1])
            mols.append({"nsc": r[0], "smiles": r[1]})
    print(f"{len(blocks)} records, {sum(r is not None for r in parsed)} pass the filters, {len(mols)} unique",
          flush=True)
    with Pool(args.workers) as pool:
        computed = pool.map(props, [m["smiles"] for m in mols], chunksize=500)
    mols = [dict(m, props=p, scaffold=s) for m, (p, s) in zip(mols, computed) if s is not None]

    scaffolds = sorted({m["scaffold"] for m in mols if m["scaffold"]})
    rng.shuffle(scaffolds)
    held_scaf = set(scaffolds[: int(len(scaffolds) * args.heldout_scaffold_frac)])
    train_pool = [m for m in mols if m["scaffold"] not in held_scaf]
    held_pool = [m for m in mols if m["scaffold"] in held_scaf]
    print(f"{len(scaffolds)} ring scaffolds, {len(held_scaf)} held out; train pool {len(train_pool)}, "
          f"held-out pool {len(held_pool)}", flush=True)

    # Rare properties first, so they find enough molecules with a non-zero answer.
    def nonzero(pool, k):
        return sum(1 for m in pool if m["props"][k])
    keys = sorted(PROPS, key=lambda k: nonzero(train_pool, k))
    per_prop = args.train_rows // len(PROPS)
    out = {"train": ([], []), "validation": ([], []), "heldout_large": ([], [])}
    used_train, used_held = set(), set()
    short = {}
    for k in keys:
        for split, pool, used, n in (("train", train_pool, used_train, per_prop),
                                     ("validation", held_pool, used_held, args.val_per_prop),
                                     ("heldout_large", held_pool, used_held, args.heldout_per_prop)):
            idx = pick(pool, k, n, used, rng, args.max_zero)
            if len(idx) < n:
                short[f"{split}/{k}"] = len(idx)
            r, m = rows_for(pool, k, idx, rng)
            out[split][0].extend(r)
            out[split][1].extend(m)

    os.makedirs(args.out, exist_ok=True)
    report = {"rdkit": rdBase.rdkitVersion, "source": os.path.basename(args.sdf), "records": len(blocks),
              "unique_molecules": len(mols), "ring_scaffolds": len(scaffolds), "heldout_scaffolds": len(held_scaf),
              "args": vars(args), "short": short, "splits": {}}
    for split, (rows, meta) in out.items():
        order = list(range(len(rows)))
        rng.shuffle(order)
        with open(os.path.join(args.out, f"{split}.jsonl"), "w") as f:
            for i in order:
                f.write(json.dumps(rows[i], ensure_ascii=False) + "\n")
        with open(os.path.join(args.out, f"meta_{split}.jsonl"), "w") as f:
            for i in order:
                f.write(json.dumps(meta[i], ensure_ascii=False) + "\n")
        by_prop = collections.defaultdict(list)
        for m in meta:
            by_prop[m["property"]].append(int(m["answer"]))
        report["splits"][split] = {
            "rows": len(rows),
            "longest_prompt_chars": max(len(r["input"]) for r in rows),
            "properties": {k: {"rows": len(v), "zero": sum(x == 0 for x in v), "min": min(v), "max": max(v),
                               "mean": round(sum(v) / len(v), 2)} for k, v in sorted(by_prop.items())},
        }
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({s: v["rows"] for s, v in report["splits"].items()}), "short:", short, flush=True)


if __name__ == "__main__":
    main()
