import torch
from torch.utils.checkpoint import checkpoint

print("=" * 60)
print("测试 Checkpoint 机制")
print("=" * 60)

# 测试 1: 标准模式
x = torch.randn(10, 20, requires_grad=True)
w = torch.randn(20, 30, requires_grad=True)

def my_func(x):
    y = x @ w
    z = torch.relu(y)
    return z

# 不使用 checkpoint
output1 = my_func(x)
print("\n【标准模式】")
print(f"grad_fn: {output1.grad_fn}")
print(f"grad_fn type: {type(output1.grad_fn).__name__}")

# 查看计算图
node = output1.grad_fn
depth = 0
while node is not None:
    print(f"  Level {depth}: {type(node).__name__}")
    if hasattr(node, 'next_functions') and node.next_functions:
        node = node.next_functions[0][0]
        depth += 1
    else:
        break

# 测试 2: Checkpoint 模式
x2 = torch.randn(10, 20, requires_grad=True)

output2 = checkpoint(my_func, x2, use_reentrant=True)
print("\n【Checkpoint 模式】")
print(f"grad_fn: {output2.grad_fn}")
print(f"grad_fn type: {type(output2.grad_fn).__name__}")

# 查看计算图
node = output2.grad_fn
depth = 0
while node is not None:
    print(f"  Level {depth}: {type(node).__name__}")
    if hasattr(node, 'next_functions') and node.next_functions:
        node = node.next_functions[0][0]
        depth += 1
    else:
        break

# 测试 3: 更复杂的例子，查看保存的张量
print("\n" + "=" * 60)
print("测试保存的张量数量")
print("=" * 60)

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(20, 50)
        self.fc2 = torch.nn.Linear(50, 30)
    
    def forward(self, x):
        h = self.fc1(x)
        h = torch.relu(h)
        h = self.fc2(h)
        return h

model = MyModule()
x3 = torch.randn(10, 20, requires_grad=True)

# 标准模式
output3 = model(x3)
print(f"\n【标准模式 - 复杂网络】")
print(f"输出 grad_fn: {type(output3.grad_fn).__name__}")

# 统计计算图深度
def count_graph_depth(grad_fn):
    if grad_fn is None:
        return 0
    max_depth = 0
    if hasattr(grad_fn, 'next_functions'):
        for next_fn, _ in grad_fn.next_functions:
            depth = count_graph_depth(next_fn)
            max_depth = max(max_depth, depth)
    return max_depth + 1

depth = count_graph_depth(output3.grad_fn)
print(f"计算图深度: {depth}")

# Checkpoint 模式
x4 = torch.randn(10, 20, requires_grad=True)
output4 = checkpoint(model, x4, use_reentrant=True)
print(f"\n【Checkpoint 模式 - 复杂网络】")
print(f"输出 grad_fn: {type(output4.grad_fn).__name__}")

depth = count_graph_depth(output4.grad_fn)
print(f"计算图深度: {depth}")

print("\n" + "=" * 60)
print("结论:")
print("=" * 60)
print("1. 标准模式: 完整的计算图，包含所有中间操作")
print("2. Checkpoint 模式: 计算图被 CheckpointFunction 截断")
print("3. Checkpoint 只保存输入，中间激活在 backward 时重新计算")

