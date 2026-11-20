"""Time the deployed demo (Fly.io) end to end and save the numbers to results/deploy_test.json.

Uploads the ED and ES frames of one ACDC test patient through the public Gradio API (like a user would),
three times: single model (first call: includes lazy model loading), single model again
(warm), and the 5-model ensemble. Times are wall-clock from this machine, so they include upload + network.

Usage: python -m acdc.test_deploy --url https://acdc-cardiac-aub.fly.dev/ --patient 101
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

from gradio_client import Client, handle_file

from acdc.common import DATA_ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="https://acdc-cardiac-aub.fly.dev/")
ap.add_argument("--patient", type=int, default=101)
ap.add_argument("--data", default=str(DATA_ROOT / "testing"), help="ACDC testing folder (default: $ACDC_ROOT/testing)")
ap.add_argument("--machine", default="Fly.io performance-2x (2 dedicated CPUs, 4 GB RAM), region fra",
                help="description of the server, copied into the JSON")
ap.add_argument("--out", default="results/deploy_test.json")
a = ap.parse_args()

d = Path(a.data) / f"patient{a.patient:03d}"
info = dict(l.split(": ") for l in (d / "Info.cfg").read_text().splitlines() if ": " in l)
ed, es = d / f"patient{a.patient:03d}_frame{int(info['ED']):02d}.nii.gz", d / f"patient{a.patient:03d}_frame{int(info['ES']):02d}.nii.gz"

# 1) page load. If the machine was stopped (scale to zero), this includes machine start + app start; we wait for
#    it here because gradio_client gives up after ~10 s.
t0 = time.time()
while True:
    try:
        urllib.request.urlopen(a.url, timeout=120).read()
        break
    except Exception:
        time.sleep(1)
first_page_s = time.time() - t0
client = Client(a.url, verbose=False)
calls = []
for name, ensemble in (("single_first", False), ("single_warm", False), ("ensemble", True)):
    t = time.time()
    out = client.predict(handle_file(str(ed)), handle_file(str(es)), float(info["Height"]), float(info["Weight"]),
                         "", ensemble, fn_index=0)
    calls.append({"call": name, "ensemble": ensemble, "seconds": round(time.time() - t, 1),
                  "pred_two_stage": out[2]["label"], "pred_cnn": out[3]["label"]})
    print(calls[-1], flush=True)

res = {"url": a.url, "server": a.machine, "patient": f"patient{a.patient:03d}", "reference_group": info["Group"],
       "first_page_load_seconds": round(first_page_s, 1), "calls": calls, "date_utc": time.strftime("%Y-%m-%d %H:%M", time.gmtime()),
       "note": "wall-clock from an AWS client incl. upload; first_page_load includes machine start if it was stopped; single_first includes lazy model loading"}
Path(a.out).write_text(json.dumps(res, indent=2))
print("saved", a.out)
