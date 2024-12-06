def monkeypatch_tensor_shape():
    import torch

    print("[DEBUG] Monkeypatching torch.Tensor.__repr__ to include shape")

    def custom_repr(self):
        return f"{{Tensor:{tuple(self.shape)}}} {original_repr(self)} {{Tensor:{tuple(self.shape)}}}"

    original_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = custom_repr
