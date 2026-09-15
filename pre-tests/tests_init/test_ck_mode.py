import torch
from torch.utils.checkpoint import checkpoint

print("=" * 60)
print("Test the Checkpoint Mechanism")
print("=" * 60)

# Test 1: Standard Mode
x = torch.randn(10, 20, requires_grad=True)
w = torch.randn(20, 30, requires_grad=True)

def my_func(x):
    y = x @ w
    z = torch.relu(y)
    return z

# Do not use checkpoint
output1 = my_func(x)
print("\n[Standard Mode]")
print(f"grad_fn: {output1.grad_fn}")
print(f"grad_fn type: {type(output1.grad_fn).__name__}")

# Inspect the computation graph
node = output1.grad_fn
depth = 0
while node is not None:
    print(f"  Level {depth}: {type(node).__name__}")
    if hasattr(node, 'next_functions') and node.next_functions:
        node = node.next_functions[0][0]
        depth += 1
    else:
        break

# Test 2: Checkpoint Mode
x2 = torch.randn(10, 20, requires_grad=True)

output2 = checkpoint(my_func, x2, use_reentrant=True)
print("\n[Checkpoint Mode]")
print(f"grad_fn: {output2.grad_fn}")
print(f"grad_fn type: {type(output2.grad_fn).__name__}")

# Inspect the computation graph
node = output2.grad_fn
depth = 0
while node is not None:
    print(f"  Level {depth}: {type(node).__name__}")
    if hasattr(node, 'next_functions') and node.next_functions:
        node = node.next_functions[0][0]
        depth += 1
    else:
        break

# Test 3: A more complex example that inspects saved tensors
print("\n" + "=" * 60)
print("Test the Number of Saved Tensors")
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

# Standard Mode
output3 = model(x3)
print(f"\n[Standard Mode - Complex Network]")
print(f"Output grad_fn: {type(output3.grad_fn).__name__}")

# Calculate computation graph depth
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
print(f"Computation graph depth: {depth}")

# Checkpoint Mode
x4 = torch.randn(10, 20, requires_grad=True)
output4 = checkpoint(model, x4, use_reentrant=True)
print(f"\n[Checkpoint Mode - Complex Network]")
print(f"Output grad_fn: {type(output4.grad_fn).__name__}")

depth = count_graph_depth(output4.grad_fn)
print(f"Computation graph depth: {depth}")

print("\n" + "=" * 60)
print("Conclusion:")
print("=" * 60)
print("1. Standard Mode: Complete computation graph containing all intermediate operations")
print("2. Checkpoint Mode: The computation graph is truncated by CheckpointFunction")
print("3. Checkpoint Only inputs are saved; intermediate activations are recomputed during backward")

