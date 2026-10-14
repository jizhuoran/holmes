import json,sys
s=json.load(open(sys.argv[1]))
rec=s["ranks"][0]["layer_records"][-1]
for st,v in rec["stations"].items():
    print(f"{st:16s} peak {v.get('peak_alloc_gib')} after {v.get('alloc_after_gib')} keys {v.get('keys')}")
