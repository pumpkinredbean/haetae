# T05 M4 handoff

Status: frozen pre-M4; execution is not authorized.

## Producer

- Repository: `https://github.com/pumpkinredbean/haetae`
- Branch: `next/science-coverage`
- Producer commit: `e06ca7362a06a0ea454df06523ac484657db317b`
- Basis commit: `9c4e0e8227f0f3299472a1b195cbbe98280491dc`
- Specification self-digest: `a414f002e2b742c94c34f08ef22de4d04ee6cf8ad2b28e237371481272fe2354`

## Frozen plan

- Freeze A: `/Users/minkyu/workspace/haetae-artifacts/t05-paired-replay-v1-freeze-a`
- Freeze B: `/Users/minkyu/workspace/haetae-artifacts/t05-paired-replay-v1-freeze-b`
- Protocol self-digest: `14b31e11ad408ec6108a8ef49cc0fcd01f8ea586b3459f8354054866562a3c39`
- Protocol file SHA-256: `9ad677e25f6420766cd86885fb3969f4648fd14fccc0fcd6595638e5be680bd1`
- Manifest self-digest: `c1b45d42800637b15627cab91fad18b10c110be35ea0ff86e0eb0c9cc12844c0`
- Manifest file SHA-256: `5e4fbe8b638ff81551337caea15e9af57a413707a86271722ac1bcdd18bef109`
- Repeatability: every file in A and B is byte-identical.
- Pre-M4 counters: zero checkpoint deserializations, model forwards, MPS operations, optimizer updates, and locked-test accesses.

The plan contains 15,572 original requests with 23,068 questions, 2,048 target-support requests, four 512-row tapes, 8,131 endpoint vectors, 4,867 encoded endpoint requests, and 2,434 batch-two endpoint forwards per arm.

## Private review package

- Path: `/Users/minkyu/workspace/haetae-artifacts/haetae-t05-paired-replay-v1-prerun.zip`
- File SHA-256: `44d5dd5dae94d9d6cc9928e7fdb065424810a7fa07ca17c82efe91e744d9f1b7`
- Size: 40,471,214 bytes
- Members: 88
- Indexed payloads: 87
- Review-index self-digest: `06a488ee9d93c55bcc64ba96e6cde05ed141702b532867baac0a60303f8d1b0b`
- The package contains no checkpoint tensor payload.

## Validation

- 51 focused replay tests passed.
- Ruff and Python compilation passed.
- Full plan reconstruction reproduced every frozen byte.
- Historical immutable files remain unchanged from the T05 basis commit.
- The repository forbidden-term audit is clean.

## Review decision

Only one exact terminal phrase authorizes the seed-17 pair:

- `M4 PAIRED REPLAY START`
- `M4 PAIRED REPLAY HOLD`

A start decision authorizes only exact producer commit `e06ca7362a06a0ea454df06523ac484657db317b`, the frozen protocol and manifest above, and sequential seed-17 control then targeted initialization, parity, training, and endpoint evaluation. Seed 23 remains gated by the independently replayed six-point pilot result.

The current volume has about 33 GB free. The frozen operational rule requires at least 32 times the 1,687,348,267-byte source checkpoint, about 54 GB, before starting a seed pair. No execution may start until that reserve is met without deleting protected evidence.
