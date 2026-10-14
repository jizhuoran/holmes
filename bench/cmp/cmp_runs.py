import json, sys
out = sys.argv[1]; runs = sys.argv[2:]
recs = {r: json.load(open(f"{out}/{r}/summary.json"))["ranks"][0]["layer_records"][-1] for r in runs}
sts = list(recs[runs[0]]["stations"].keys())
print(f"{'station':16s}" + "".join(f"{r[5:]:>14s}" for r in runs))
print(f"{'layer_wall':16s}" + "".join(f"{recs[r]['layer_wall_s']:14.3f}" for r in runs))
for st in sts:
    print(f"{st:16s}" + "".join(f"{recs[r]['stations'][st]['wall_s']:14.3f}" for r in runs))
print()
keys = set()
for r in runs:
    for st in recs[r]["stations"]:
        for k in recs[r]["stations"][st].get("stage_seconds", {}):
            if ".chunk" not in k:
                keys.add((st, k))
for st, k in sorted(keys):
    vals = [recs[r]["stations"][st].get("stage_seconds", {}).get(k, float("nan")) for r in runs]
    if max(abs(v) for v in vals if v == v) > 0.02:
        print(f"{st + ':' + k:62s}" + "".join(f"{v:14.3f}" for v in vals))
