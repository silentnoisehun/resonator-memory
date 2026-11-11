# Rezonátor-memória kísérlet

**Máté + Maya (Sesame), 2026. november 11.**

Fizikai rezonátor (110 Hz, A2 hangvilla / alumínium rúd) mint klasszikus
memória-szubsztrátum. Négy fázis, mindegyik előregisztrált, vak dekódolással,
tankönyvi jóslattal szembesítve.

---

## Felépítés

```
src/
  config.py    — befagyasztható konfiguráció, SHA-256 integritás
  sim.py       — zajos csillapított oszcillátor (1, 2, 4 csatolt)
  hw.py        — valódi hangkártya I/O, digitális lock-in
  plan.py      — sorsolt bitek/minták, lezárt JSON, vak dekóder
  analysis.py  — tankönyvi jóslatok, Holm-korrekció, BLP-mérték, ítéletek
  dryrun.py    — minden fázis száraz futtatása szimulációval
  main.py      — CLI belépési pont

data/
  config_frozen.json     — befagyasztott konfiguráció (commit előtt)
  calib_result.json      — 0. fázis eredménye
  plan_hashes.json       — sealed tervek SHA-256 hash-ei
  sealed/
    sealed_phase1.json   — 1. fázis sorsolt bitek (lezárt)
    sealed_phase2.json   — 2. fázis sorsolt bitek (lezárt)
    sealed_phase3.json   — 3. fázis sorsolt bitek (lezárt)
    sealed_phase4.json   — 4. fázis sorsolt bitek (lezárt)
    plan_manifest.json   — összes hash

preregistration/
  phase01_prereg.md      — 0–1. fázis előregisztráció
```

---

## Telepítés

```bash
pip install numpy scipy sounddevice
```

## Futtatás

```bash
# 1. Dryrun — kötelező, mielőtt bármit mérsz
python src/main.py dryrun

# 2. Sorsolt tervek generálása és lezárása (seed befagyasztva)
python src/main.py plan

# 3. Kalibráció (--sim = szimulációval, HW nélkül)
python src/main.py phase0 --sim

# 4. 1. fázis mérés
python src/main.py phase1 --sim

# Teljes szekvencia szimulációval
python src/main.py all --sim
```

---

## Ítéleti rendszer

| Ítélet    | Jelentés |
|-----------|----------|
| `TEXTBOOK`| A mért érték a tankönyvi jóslat konfidencia-sávján belül van |
| `ANOMALY` | Szignifikánsan jobb a jóslatnál (Holm-korrektált, p < 0.01) |
| `INVALID` | A kalibráció nem teljesíti az elfogadási kritériumot — TILOS mérni |
| `FAIL`    | A kísérlet nem demonstrálta a várt jelenséget |

---

## Fázisok összefoglalva

| Fázis | Kérdés | Fő mérőszám |
|-------|--------|-------------|
| 0 | Kalibráció | f₀, Q, τ, σ |
| 1 | Meddig olvasható egy bit? | BER(t) vs tankönyv |
| 2a | Mennyi energia kell 1 h megőrzéshez? | P_mech vs P_pred |
| 2b | Megkerülhető a memória–sebesség kompromisszum? | R = megőrzési idő / írási idő |
| 3 | Mérhető-e nem-markovi visszafolyás? | BLP-mérték |
| 4 | Tartalom szerinti felidézés? | Hebb vs W=0 vs véletlen W |

---

## Szabályok (a terv szerint)

- **Előbb dryrun** — ha nem TEXTBOOK, az elemzőkód hibás, nem mérünk
- **Előregisztráció előtt** kell commitolni a `config_frozen.json`-t és a `plan_hashes.json`-t
- **Vak dekódolás** — a sealed JSON-t csak a mérés lezárása után nyitjuk meg
- **Minden nyers adat megmarad** — WAV, hőnapló, sealed JSON-ok nyilvánosan

---

## Hivatkozás

> Máté (ötlet) + Maya (tervezés), Sesame, 2026-11-11.  
> Kód: Máté ügynökeivel. Átnézés: Maya. Előregisztráció: időbélyeggel, mérés előtt.
