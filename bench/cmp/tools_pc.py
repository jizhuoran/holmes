import json,sys
s=json.load(open(sys.argv[1]))
for rec in s["ranks"][0]["layer_records"]:
    print("layer",rec.get("layer"), {st:v.get("plain_cache") for st,v in rec["stations"].items() if any(v.get("plain_cache",{}).values())})
