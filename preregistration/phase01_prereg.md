# 0–1. fázis előregisztráció

**Projekt:** Rezonátor-memória kísérlet  
**Dátum:** 2026-11-11  
**Szerzők:** Máté (ötlet, kód), Maya (tervezés, átnézés)

---

## Konfiguráció hash

A `data/config_frozen.json` SHA-256 hash-e itt lesz rögzítve a commit után:

```
CONFIG_SHA256: [COMMITOLÁS UTÁN TÖLTSD KI]
```

## Sealed tervek hash-ei

```json
[PLAN GENERÁLÁS UTÁN TÖLTSD KI — python src/main.py plan]
```

---

## 0. fázis: kalibráció

**Elfogadási kritérium:**
- Q szórása < 10% (20 ismétlésből)
- f₀ sodródása < sávszélesség / 4

**Ha nem teljesül: INVALID, mérés tilos.**

---

## 1. fázis: szabad lecsengés BER

**Kérdés:** Meddig olvasható vissza helyesen egy fázisba írt bit?

**Hipotézis (tankönyvi, null):**  
`BER(t) = ½ · erfc(A₀·exp(−t/τ) / (σ·√2))`  
Paraméterek kizárólag a 0. fázisból.

**Késleltetések:** τ × {0.5, 1, 2, 3, 4, 5}  
**Próbák késleltetésenként:** 200  
**Próbák között:** legalább 10τ várakozás vagy aktív csillapítás

**Ítéleti szabály:**
- `TEXTBOOK`: minden mért BER a 99%-os binomiális sávon belül
- `ANOMALY`: ≥2 késleltetésen szignifikánsan jobb a jóslatnál (Holm, p < 0.01)
- Egy ANOMALY után kötelező ismétlés más rezonátorral

**Statisztikai korrektúra:** Holm–Bonferroni a 6 késleltetésre

---

## Dryrun státusz

```
[TÖLTSD KI A DRYRUN LEFUTÁSA UTÁN]
python src/main.py dryrun
```

Elvárt: minden fázis TEXTBOOK, pozitív kontroll ANOMALY.

---

## Sorrend

1. ☐ `python src/main.py dryrun` — dryrun átment
2. ☐ `python src/main.py plan`  — sealed tervek generálva
3. ☐ Config + plan hash-ek feljegyezve ebbe a fájlba
4. ☐ **Git commit + push** (időbélyeg = előregisztráció)
5. ☐ Mérés: `phase0`, majd `phase1`
6. ☐ Elemzés, sealed JSON megnyitás, ítélet
7. ☐ Eredmény nyilvánosan (nulla eredménnyel együtt)
