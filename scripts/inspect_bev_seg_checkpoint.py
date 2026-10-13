"""安全检查 BEVFormer-Small 地图分割检查点；不修改原始权重。"""
from collections import Counter
from pathlib import Path
import sys
import zipfile

import torch


def main():
    checkpoint = Path(sys.argv[1] if len(sys.argv) > 1 else '/mnt/g/nus/bevformer_small_seg.pth')
    print('===== BEV 分割权重检查 =====', flush=True)
    print('文件:', checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f'未找到权重: {checkpoint}')
    print(f'大小: {checkpoint.stat().st_size / (1024**2):.1f} MiB')
    if not zipfile.is_zipfile(checkpoint):
        raise ValueError('不是有效 ZIP 格式的 PyTorch 权重；请核实下载文件。')

    # 验证常见 PyTorch 压缩包布局，排除下载成网页等情况。
    with zipfile.ZipFile(checkpoint) as zf:
        names = zf.namelist()
        print('归档条目数量:', len(names))
        print('含 data.pkl:', any(name.endswith('/data.pkl') or name == 'data.pkl' for name in names))
        print('归档示例:', names[:4])

    # 只允许安全的张量及基础数据类型反序列化；不使用 weights_only=False。
    print('使用 torch.load(weights_only=True) 读取中...', flush=True)
    obj = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    if not isinstance(obj, dict):
        raise TypeError(f'不符合预期的顶层对象: {type(obj).__name__}')
    print('顶层字段:', list(obj.keys())[:20])
    state = obj.get('state_dict', obj.get('model', obj))
    if not isinstance(state, dict):
        raise TypeError(f'权重字段不是字典: {type(state).__name__}')

    keys = list(state.keys())
    tensor_count = sum(torch.is_tensor(v) for v in state.values())
    print('权重字段总数:', len(keys), '其中张量数:', tensor_count)
    heads = Counter(k.split('.')[0] for k in keys)
    print('顶层模块统计:', heads.most_common(12))
    matched = [k for k in keys if any(term in k.lower() for term in ('seg', 'map', 'raster', 'semantic'))]
    print('分割/地图相关权重数量:', len(matched))
    for k in matched[:20]:
        v = state[k]
        print(' ', k, '| shape =', tuple(v.shape) if torch.is_tensor(v) else type(v).__name__)
    if not matched:
        print('[注意] 名称检索未找到分割权重，不能据此确认检查点和配置匹配。')
    print('[完成] 权重文件已安全读取；下一步仍需实际模型加载验证。')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('[检查失败]', type(exc).__name__, ':', exc, file=sys.stderr)
        print('请把完整输出发给我；不要改用 weights_only=False。', file=sys.stderr)
        sys.exit(1)
