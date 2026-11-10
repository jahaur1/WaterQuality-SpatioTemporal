"""Preview or link excluded resources from an existing local workspace; never overwrite."""
import argparse
import json
import os
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',required=True,type=Path)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];workspace=args.workspace.resolve()
    if not workspace.is_dir() or workspace==root:
        raise SystemExit('Provide a different existing original workspace.')
    manifest=json.loads((root/'docs/local_assets.json').read_text(encoding='utf-8'))
    totals={'already_present':0,'missing_source':0,'planned':0,'linked':0,'failed':0}
    for relative,kind in manifest.items():
        source=workspace/relative;target=root/relative
        if not source.resolve().is_relative_to(workspace) or not target.parent.resolve().is_relative_to(root):
            raise SystemExit('Resource path escapes the selected roots: '+relative)
        if target.exists() or target.is_symlink():totals['already_present']+=1;continue
        if not source.exists():totals['missing_source']+=1;continue
        totals['planned']+=1
        print(('LINK ' if args.apply else 'PLAN ')+relative)
        if args.apply:
            target.parent.mkdir(parents=True,exist_ok=True)
            try:
                os.symlink(source,target,target_is_directory=kind=='directory')
                totals['linked']+=1
            except OSError as error:
                totals['failed']+=1
                print('Could not create link; place this resource manually: '+relative+' ('+str(error)+')')
    print(json.dumps(totals,ensure_ascii=False,indent=2))
    print('Local resources remain excluded from Git. No source was moved or deleted.')

if __name__=='__main__':main()
