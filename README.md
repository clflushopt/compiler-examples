# Compiler Examples

This repo contains various reference implementations of compiler related data 
structures and algorithms.

Most of the following was initially done while working through Cornell's CS6120
[course](https://www.cs.cornell.edu/courses/cs6120/2023fa/self-guided/) from
the course assignments, reading papers, LLVM's codebase and various books.

The objective is to create a sort of cookbook of reference implementations that
focus on clarity, simplicity and hopefully correctness. 

The repository contains a complete implementation of the Bril intermediate
language with several analyses, optimizations and various compiler related
data structures e.g control flow graphs, SSA representation, dominance graphs...

I hope you enjoy hacking on it as much as I enjoyed writing.

## Implementations

* Core IR implementation with support for basic blocks and control flow graphs see [ir.py](bril/core/ir.py).
* Tools to parse Bril from both text and JSON formats see [bril2txt](bril/tools/bril2ir.py) and [bril2json](bril/tools/bril2json.py).
* Implementation of various scalar optimizations such as [Dead Code Elimination](bril/core/dce.py), 
[Local Value Numbering](bril/core/lvn.py) and [SCCP](bril/core/sccp.py).
* Implementation of various loop optimizations such as [Loop Invariant Code Motion](bril/core/licm.py).
* Implementation of phase ordering selection as a [profile guided optimization](bril/core/pgo.py)

## References

* [CS6120: Advanced Compilers](https://www.cs.cornell.edu/courses/cs6120/2023fa/self-guided/)
* [LLVM Programmer's Manual](https://llvm.org/docs/ProgrammersManual.html)
* [Engineering a Compiler 3rd Edition](https://www.sciencedirect.com/book/9780128154120/engineering-a-compiler)
* [CSC255/455 Software Analysis and Improvement](https://www.cs.rochester.edu/~sree/courses/csc-255-455/spring-2020/schedule.html)
* [Data flow analysis: an informal introduction](https://clang.llvm.org/docs/DataFlowAnalysisIntro.html)
* [Notes on Graph Algorithms in Optimizing Compilers](https://www.cs.umb.edu/~offner/files/flow_graph.pdf)
* [SSA-based Compiler Design](https://link.springer.com/book/10.1007/978-3-030-80515-9)