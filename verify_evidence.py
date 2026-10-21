"""Verify published evidence with the standard library only; no learner imports or training."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import statistics

ROOT = Path(__file__).resolve().parent

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    checked = set()
    for line in (ROOT / 'SHA256SUMS').read_text(encoding='utf-8').splitlines():
        expected, relative = line.split('  ', 1)
        path = ROOT / relative
        assert path.resolve().is_relative_to(ROOT), relative
        assert digest(path) == expected, relative
        checked.add(relative)
    for item in json.loads((ROOT / 'provenance/source-inventory.json').read_text(encoding='utf-8')):
        assert digest(ROOT / item['public_path']) == item['public_sha256'], item['public_path']
        if not item['changes']:
            assert item['original_sha256'] == item['public_sha256'], item['public_path']
    for path in ROOT.rglob('*.py'):
        if any(part in ('.venv', '__pycache__', '.git') for part in path.relative_to(ROOT).parts):
            continue
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path.relative_to(ROOT)))
    for directory in sorted(ROOT.glob('results_strict*')):
        config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
        results = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
        assert [r['seed'] for r in results] == config['seeds']
        for result in results:
            with (directory / f"facts_seed{result['seed']}.csv").open(encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            assert len(rows) == result['facts_presented']
            assert sum(row['accepted'] == 'True' for row in rows) == result['facts_accepted']
            if result['stopped']:
                assert rows[-1]['accepted'] == 'False'
                assert int(rows[-1]['loops']) == config['strict']['max_loops_per_fact']
        print(directory.name + ': ' + ', '.join(str(r['facts_accepted']) for r in results))
    ablation = json.loads((ROOT / 'results_hard_ablation/summary.json').read_text(encoding='utf-8'))
    rgll = {r['seed']: r for r in ablation['rgll']}
    for method in ('replay_revisit_steps', 'replay_revisit_budget'):
        assert len(ablation[method]) == len(rgll) == 5
        assert all(r['final_acc'] > rgll[r['seed']]['final_acc'] for r in ablation[method])
        print(f"{method}: {statistics.mean(r['final_acc'] for r in ablation[method]):.3%}; wins on all five stored seeds")
    print(f'PASS: {len(checked)} publication hashes, source mappings, syntax and stored result consistency. No experiments run.')

if __name__ == '__main__':
    main()
