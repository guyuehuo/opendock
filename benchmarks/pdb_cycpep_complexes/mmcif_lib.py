"""Lightweight mmCIF parser + PDB/CIF helpers for building PDBbind-style
protein-cyclic-peptide complex directories from raw RCSB mmCIF files.

Only the categories needed by this pipeline are surfaced:

* ``_atom_site``          -> per-atom records (auth + label naming kept)
* ``_struct_asym``        -> auth_asym_id -> entity_id
* ``_entity_poly``        -> entity_id -> polymer type
* ``_struct_conn``        -> covalent/disulfide/modres links (for cyclicity)

The parser is intentionally minimal but handles quoted values and ``loop_``
blocks, which cover all of the above categories in real PDB entries.
"""
from __future__ import annotations

import os


# --------------------------------------------------------------------------- #
# minimal mmCIF tokeniser / category parser
# --------------------------------------------------------------------------- #
def _line_tokens(stripped):
    """Split one physical mmCIF line into tokens, honouring '...' / "..."
    quoted values. Never yields multi-line text fields (handled by caller)."""
    i, n = 0, len(stripped)
    while i < n:
        ch = stripped[i]
        if ch.isspace():
            i += 1
            continue
        if ch in "'\"":
            j = i + 1
            while j < n and stripped[j] != ch:
                j += 1
            yield stripped[i + 1:j]
            i = j + 1
        else:
            j = i
            while j < n and not stripped[j].isspace():
                j += 1
            yield stripped[i:j]
            i = j


def _iter_tokens(lines):
    """Yield logical tokens of an mmCIF file.

    Handles unquoted bare tokens, single/double quoted tokens, and the
    multi-line ``;...;`` text-field convention. A single physical line can
    carry several tokens; a text field is collapsed into exactly one token.
    Blank / ``#`` comment lines are forwarded verbatim (they only separate
    blocks in :func:`parse_mmcif`).
    """
    lines = list(lines)
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            yield stripped
            i += 1
            continue
        if stripped.startswith(";"):
            # mmCIF text field: starts at a ';' and runs until the next line
            # whose first non-blank char is ';' (content may share the
            # opening line after the marker).
            buf = []
            rest = line[line.index(";") + 1:]
            if rest.strip():
                buf.append(rest.strip())
            i += 1
            while i < n:
                s = lines[i].strip()
                if s.startswith(";"):
                    i += 1
                    break
                buf.append(lines[i].strip())
                i += 1
            else:
                raise ValueError("unterminated ';' text field in mmCIF")
            yield " ".join(buf)
            continue
        yield from _line_tokens(stripped)
        i += 1


def parse_mmcif(path):
    """Parse an mmCIF into a dict ``{category: list[dict]}``.

    Categories starting with a leading underscore are stored without it
    (``_atom_site`` -> ``atom_site``).  ``loop_`` categories become a list of
    row dicts keyed by their ``_cat.field`` headers.
    """
    with open(path) as f:
        raw = f.read()
    tokens = list(_iter_tokens(raw.splitlines()))
    cats = {}
    scalar = {}                      # category -> merged single record

    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        if tok.startswith("_") and not tok.startswith("loop_"):
            # scalar (non-loop) category: _cat.field value  (merged below)
            head, _, field = tok[1:].partition(".")
            value = tokens[i + 1] if i + 1 < n else ""
            scalar.setdefault(head, {})[field] = value
            i += 2
            continue
        if tok == "loop_":
            i += 1
            headers = []
            while i < n and tokens[i].startswith("_") and \
                    not tokens[i].startswith("loop_"):
                headers.append(tokens[i][1:])   # strip leading underscore
                i += 1
            if not headers:
                continue
            catname = headers[0].split(".")[0]
            bare = [h.split(".", 1)[1] if "." in h else h for h in headers]
            rows = []
            while i < n:
                t = tokens[i]
                if t == "loop_" or (t.startswith("_") and "." in t and
                                    not t.startswith("loop_")):
                    break
                if not t or t.startswith("#"):
                    i += 1
                    continue
                # consume one full row of len(headers) tokens
                row = {}
                for b in bare:
                    if i < n:
                        row[b] = tokens[i]
                        i += 1
                    else:
                        row[b] = ""
                rows.append(row)
            if catname in cats:
                cats[catname].extend(rows)
            else:
                cats[catname] = rows
            continue
        i += 1

    # merge any non-loop (scalar) categories in front of loop rows
    for head, record in scalar.items():
        cats.setdefault(head, []).insert(0, record)
    return cats


# --------------------------------------------------------------------------- #
# atom record
# --------------------------------------------------------------------------- #
class AtomRec:
    __slots__ = ("serial", "name", "altloc", "comp", "chain", "resseq",
                 "icode", "x", "y", "z", "element", "occupancy", "bfactor",
                 "hetatm", "seq_id")

    def __init__(self, row):
        def g(*keys, default=""):
            for k in keys:
                if k in row and row[k] not in ("", "?", "."):
                    return row[k]
            return default

        def fnum(v, default=0.0):
            try:
                return float(v)
            except (TypeError, ValueError):
                return default

        self.serial = int(fnum(g("id")))
        self.name = g("auth_atom_id", "label_atom_id")
        self.altloc = g("label_alt_id", "alt_id")
        self.comp = g("auth_comp_id", "label_comp_id")
        self.chain = g("auth_asym_id", "label_asym_id")
        self.resseq = g("auth_seq_id", "label_seq_id", default="1")
        self.icode = g("pdbx_PDB_ins_code")
        self.x = fnum(g("Cartn_x"))
        self.y = fnum(g("Cartn_y"))
        self.z = fnum(g("Cartn_z"))
        self.element = (g("type_symbol") or "C").capitalize()
        self.occupancy = fnum(g("occupancy"), 1.0)
        self.bfactor = fnum(g("B_iso_or_equiv"), 0.0)
        self.hetatm = g("group_PDB") == "HETATM"
        seq = g("pdbx_seq_id", "label_seq_id", "auth_seq_id")
        self.seq_id = int(fnum(seq, 10 ** 9)) if seq else None

    def xyz(self):
        return (self.x, self.y, self.z)


def load_atom_records(cats):
    rows = cats.get("atom_site", [])
    # NMR multi-model entries carry the same atoms once per model; keep only
    # the first model so a residue never contains duplicate atom names.
    model_nums = sorted({r.get("pdbx_PDB_model_num", "") for r in rows}
                        - {"", ".", "?"},
                        key=lambda m: int(m) if m.isdigit() else 10 ** 9)
    keep_model = model_nums[0] if model_nums else None
    out = []
    for row in rows:
        if keep_model is not None:
            m = row.get("pdbx_PDB_model_num", "")
            if m not in ("", ".", "?") and m != keep_model:
                continue
        rec = AtomRec(row)
        if rec.element == "H":
            continue
        if _is_water(rec.comp):
            continue
        out.append(rec)
    return out


def _is_water(comp):
    return comp in ("HOH", "WAT", "H2O", "DOD")


# --------------------------------------------------------------------------- #
# entity / link metadata
# --------------------------------------------------------------------------- #
def load_entities(cats):
    """Return {auth_asym_id: {'entity_id': str, 'poly_type': str}}.

    ``_struct_asym`` is keyed by the label asym id, while atom records carry the
    auth asym id; the two differ in many entries. The result is therefore keyed
    by the auth chain id (as used on ``AtomRec.chain``) via ``_atom_site``.
    """
    entity_of_asym = {row.get("id", ""): row.get("entity_id", "")
                      for row in cats.get("struct_asym", [])}
    poly = {row.get("entity_id", ""): row.get("type", "")
            for row in cats.get("entity_poly", [])}

    auth_to_label = {}
    for row in cats.get("atom_site", []):
        label = row.get("label_asym_id", "")
        auth = row.get("auth_asym_id") or label
        auth_to_label.setdefault(auth, label)

    entities = {}
    for auth, label in auth_to_label.items():
        eid = entity_of_asym.get(label, "")
        entities[auth] = {"entity_id": eid, "poly_type": poly.get(eid, "")}
    # keep any struct_asym entries not represented in atom_site
    for aid, eid in entity_of_asym.items():
        entities.setdefault(aid, {"entity_id": eid,
                                  "poly_type": poly.get(eid, "")})
    return entities


def load_struct_conn(cats):
    """Return list of dicts of covalent link partners (auth coordinates)."""
    links = []
    for row in cats.get("struct_conn", []):
        ctype = row.get("conn_type_id", "")
        if ctype not in ("covale", "disulf", "modres", "metalc", "hydrog",
                         "saltbr"):
            continue
        links.append({
            "type": ctype,
            "c1": (row.get("ptnr1_auth_asym_id"), row.get("ptnr1_auth_seq_id"),
                   row.get("ptnr1_auth_atom_id") or row.get("ptnr1_label_atom_id")),
            "c2": (row.get("ptnr2_auth_asym_id"), row.get("ptnr2_auth_seq_id"),
                   row.get("ptnr2_auth_atom_id") or row.get("ptnr2_label_atom_id")),
        })
    return links


# --------------------------------------------------------------------------- #
# PDB text writers (fixed column layout, PDB v3.x style)
# --------------------------------------------------------------------------- #
def _fmt_name(name, element):
    """Standard PDB atom-name column 13-16 layout."""
    name = (name or "").strip()
    if len(name) >= 4:
        return name[:4]
    if element and len(element) == 2 and element[0].isalpha():
        return element.upper().rjust(2) + name.ljust(2)
    return name.ljust(3).rjust(4)


def format_pdb_atom(rec, serial, element=None, hetatm=False):
    """Fixed-width (1-indexed) PDB ATOM/HETATM line, PDBv3 layout.

    Columns (1-based): record 1-6, serial 7-11, blank 12, name 13-16,
    altLoc 17, resName 18-20, blank 21, chainID 22, resSeq 23-26,
    iCode 27, blanks 28-30, x 31-38, y 39-46, z 47-54, occ 55-60,
    bfactor 61-66, blanks 67-76, element 77-78, charge 79-80.
    """
    element = (element or rec.element or "C").upper()
    name = _fmt_name(rec.name, element)
    buf = [" "] * 80
    recid = "HETATM" if hetatm else "ATOM  "
    buf[0:6] = recid
    buf[6:11] = str(int(serial)).rjust(5)
    buf[12:16] = name
    buf[16] = " "                       # altLoc
    buf[17:20] = (rec.comp or "UNK").rjust(3)
    buf[21] = (rec.chain or " ")[:1]
    buf[22:26] = str(rec.resseq if rec.resseq is not None else 0).rjust(4)
    buf[26] = " "                       # insertion code
    buf[30:38] = "%8.3f" % rec.x
    buf[38:46] = "%8.3f" % rec.y
    buf[46:54] = "%8.3f" % rec.z
    buf[54:60] = "%6.2f" % rec.occupancy
    buf[60:66] = "%6.2f" % rec.bfactor
    buf[76:78] = element.rjust(2)
    return "".join(buf) + "\n"


def conect_lines(pairs):
    """CONECT records for 1-2 atom serial pairs."""
    lines = []
    for a, b in pairs:
        lines.append("CONECT%5s%5s" % (a, b))
    return lines


def pdb_block(records, serial_of):
    """Return formatted PDB text lines for `records` (in order)."""
    lines = []
    for i, rec in enumerate(records):
        lines.append(format_pdb_atom(rec, i + 1 if serial_of is None
                                     else serial_of[rec]))
    return lines
