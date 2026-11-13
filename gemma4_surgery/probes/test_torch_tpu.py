# Copyright 2026 Keshan Sodimana
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import torch
import transformers

print("Accelerator available:", torch.accelerator.is_available())
acc = torch.accelerator.current_accelerator()
print("Current accelerator:", acc)

t0 = time.time()
x = torch.randn(2048, 2048, device=acc, dtype=torch.bfloat16)
y = x @ x.T
torch.accelerator.synchronize()
print(f"TPU matmul 2048x2048 bf16 ok in {time.time()-t0:.3f}s, mean={y.float().mean().item():.4f}")

print("Gemma4 classes in transformers:", [k for k in dir(transformers) if "gemma4" in k.lower()])
