"""将第三轮所有写入重定向到新目录，保持封存输出不动。"""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--phase',choices=('labels','diagnose','train','objects','dev','final','gradient','range','report'),required=True)
    a=p.parse_args();path=a.output_dir.resolve()
    from scripts.stage2 import spatial_v3_data as data
    original=data.OUT.resolve()
    if path==original:raise ValueError('复现必须使用新目录，禁止覆盖第三轮封存结果')
    data.OUT=path
    if a.phase=='labels':data.main();return
    if not (path/'annotations/manifest.json').is_file():raise ValueError('先执行labels阶段')
    if a.phase in ('diagnose','train'):
        from scripts.stage2 import spatial_v3_experiment as experiment
        getattr(experiment,a.phase)()
    elif a.phase=='objects':
        from scripts.stage2.spatial_v3_objects import train
        train()
    elif a.phase in ('dev','final'):
        from scripts.stage2.spatial_v3_evaluate import main as evaluate
        sys.argv=[sys.argv[0],'--split',a.phase];evaluate()
    elif a.phase=='gradient':
        from scripts.stage2.spatial_v3_gradient_diagnose import main as diagnose
        diagnose()
    elif a.phase=='range':
        from scripts.stage2.spatial_v3_range_correction import main as correct
        correct()
    else:
        from scripts.stage2.spatial_v3_report import main as report
        report()

if __name__=='__main__':main()
