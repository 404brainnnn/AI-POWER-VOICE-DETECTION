from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ROOT

parser=argparse.ArgumentParser(); parser.add_argument("--uri",required=True); parser.add_argument("--output",required=True); args=parser.parse_args()
if not args.uri.startswith("s3://"): Path(args.output).write_bytes(Path(args.uri).read_bytes())
else:
    import boto3
    bucket,key=args.uri[5:].split("/",1); Path(args.output).parent.mkdir(parents=True,exist_ok=True); boto3.client("s3").download_file(bucket,key,args.output)
print(args.output)
