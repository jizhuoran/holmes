import json,sys
s=json.load(open(sys.argv[1]))
rec=s["ranks"][0]["layer_records"][-1]
print("layer_wall", rec.get("layer_wall_s"), "bs", rec.get("bootstrap_seconds"))
for st,v in rec["stations"].items():
    print(f"{st:30s} {v['wall_s']:7.3f}", {k:round(x,3) for k,x in (v.get("stage_seconds") or {}).items()})
