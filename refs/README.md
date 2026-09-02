# refs

Attached community trees. Git submodules, shallow by default.

| Directory                                  | Upstream                                                         |
|--------------------------------------------|------------------------------------------------------------------|
| `b70-optimization-lab/`                    | https://github.com/steveseguin/b70-optimization-lab              |
| `intel-arc-pro-b70-inference-cookbook/`    | https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook |
| `flashnext-harness/`                       | https://github.com/bbeartheancient/flashnext-harness             |

```
git submodule update --init --depth 1
```

Sites without a git tree (listed in docs/REFERENCES.md, not cloned):

- https://neural.download/  (front end for Steve's lab)
- https://xecores.com/

Do not edit inside these trees and commit back to xe2x2. Peek, then
write our own experiment under kernels/ or parallel/.
