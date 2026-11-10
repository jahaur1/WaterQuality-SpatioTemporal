"""Read-only archive validation. Does not train models or import experiment modules."""
import ast
import hashlib
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PATTERNS={
    'github_pat':re.compile(r'github_pat_[A-Za-z0-9_]{30,}'),
    'github_token':re.compile(r'gh[pousr]_[A-Za-z0-9]{30,}'),
    'huggingface_token':re.compile(r'hf_[A-Za-z0-9]{25,}'),
    'private_key':re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----'),
}
def main():
    manifest=json.loads((ROOT/'docs/snapshot_manifest.json').read_text(encoding='utf-8'))
    issues=[];pycount=0
    for item in manifest:
        path=ROOT/item['path']
        if not path.resolve().is_relative_to(ROOT) or not path.is_file():
            issues.append({'path':item['path'],'issue':'missing or escaping path'});continue
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
            issues.append({'path':item['path'],'issue':'snapshot hash mismatch'})
        if path.suffix.lower() in {'.pt','.pth','.ckpt','.safetensors','.pkl','.pickle'}:
            issues.append({'path':item['path'],'issue':'model binary in export'})
        if path.name=='水质数据_750站点_20250410-20250510.csv':
            issues.append({'path':item['path'],'issue':'raw dataset in export'})
    # Only the published paths plus repository-owned tools/docs; do not walk linked assets.
    paths=[ROOT/item['path'] for item in manifest]
    paths+=list((ROOT/'tools').glob('*.py'))
    paths+=list((ROOT/'docs').glob('*.md'))+[ROOT/'README.md',ROOT/'THIRD_PARTY_NOTICES.md']
    for path in paths:
        if not path.is_file() or path.suffix.lower() not in {'.py','.md','.json','.csv','.txt','.toml','.yaml','.yml'}:continue
        value=path.read_text(encoding='utf-8-sig')
        if path.suffix=='.py':
            try:ast.parse(value,filename=str(path));pycount+=1
            except SyntaxError as error:issues.append({'path':path.relative_to(ROOT).as_posix(),'issue':'Python syntax: '+str(error)})
        for label,pattern in PATTERNS.items():
            if pattern.search(value):issues.append({'path':path.relative_to(ROOT).as_posix(),'issue':'credential-shaped string: '+label})
    report=dict(passed=not issues,manifest_files=len(manifest),python_files_parsed=pycount,total_snapshot_bytes=sum(x['bytes'] for x in manifest),issues=issues,note='Syntax and archive checks only; training and full reproducibility were not rerun.')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if issues:raise SystemExit(1)

if __name__=='__main__':main()
