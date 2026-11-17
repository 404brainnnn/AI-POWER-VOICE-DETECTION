from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from _bootstrap import ROOT
from training.dataset import read_manifest


def file_hash(path: str) -> str | None:
    target=Path(path)
    if not target.is_file(): return None
    digest=hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""): digest.update(chunk)
    return digest.hexdigest()


def audit(train_csv: str, validation_csv: str, test_csv: str, hash_files: bool = True) -> dict:
    splits={"train":read_manifest(train_csv,True),"validation":read_manifest(validation_csv,True),"test":read_manifest(test_csv,True)}
    report={"counts":{name:{"rows":len(rows),"human":sum(int(row["label"])==0 for row in rows),"ai":sum(int(row["label"])==1 for row in rows)} for name,rows in splits.items()},"overlap":{},"metadata_overlap":{},"missing_files":{},"duplicate_hashes":{}}
    locations={name:{str(Path(row["location"]).resolve()) for row in rows} for name,rows in splits.items()}
    for left,right in (("train","validation"),("train","test"),("validation","test")): report["overlap"][f"{left}_vs_{right}"]=sorted(locations[left]&locations[right])
    for field in ("speaker_id","source_recording_id","generator_id"):
        sets={name:{str(row[field]) for row in rows if row.get(field) not in (None,"", "nan")} for name,rows in splits.items()}
        report["metadata_overlap"][field]={f"{left}_vs_{right}":sorted(sets[left]&sets[right]) for left,right in (("train","validation"),("train","test"),("validation","test"))}
    for name,rows in splits.items(): report["missing_files"][name]=[row["location"] for row in rows if not Path(row["location"]).is_file()]
    if hash_files:
        hashes={name:{} for name in splits}
        for name,rows in splits.items():
            for row in rows:
                digest=file_hash(row["location"])
                if digest: hashes[name].setdefault(digest,[]).append(row["location"])
        for left,right in (("train","validation"),("train","test"),("validation","test")):
            shared=set(hashes[left])&set(hashes[right]); report["duplicate_hashes"][f"{left}_vs_{right}"]=sorted(shared) if shared else []
    return report


parser=argparse.ArgumentParser(); parser.add_argument("--train",required=True); parser.add_argument("--validation",required=True); parser.add_argument("--test",required=True); parser.add_argument("--no-hash",action="store_true"); parser.add_argument("--output",default="artifacts/split_audit.json"); args=parser.parse_args(); result=audit(args.train,args.validation,args.test,not args.no_hash); Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps(result,indent=2))
