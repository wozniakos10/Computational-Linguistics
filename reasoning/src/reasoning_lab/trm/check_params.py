from tiny_recursive_model import MLPMixer1D, TinyRecursiveModel


def count_params(dim, depth=4):
    model = TinyRecursiveModel(dim=dim, num_tokens=10, network=MLPMixer1D(dim=dim, depth=depth, seq_len=81))
    return sum(p.numel() for p in model.parameters())


print(f"Dim=128, Depth=4: {count_params(128, 4) / 1e6:.2f}M")
print(f"Dim=384, Depth=2: {count_params(384, 2) / 1e6:.2f}M")
print(f"Dim=384, Depth=4: {count_params(384, 4) / 1e6:.2f}M")
print(f"Dim=512, Depth=2: {count_params(512, 2) / 1e6:.2f}M")
print(f"Dim=512, Depth=4: {count_params(512, 4) / 1e6:.2f}M")
print(f"Dim=1024, Depth=2: {count_params(1024, 2) / 1e6:.2f}M")
print(f"Dim=768, Depth=8: {count_params(768, 8) / 1e6:.2f}M")
