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

## Usage

The repository is designed to allow you to hack on things as much as possible.

The core implementation revolves around [`bril.ir`](bril/core/ir.py) where most
of the core intermediate language is implemented, the [tools](bril/tools/) directory
contains small utilities that can be used to parse and manipulate Bril programs.

For testing, a custom built tool inspired by `llvm-lit` called [`turnstile`](turnstile/turnstile.py)
is provided. `turnstile` runs snapshot testing of input vs outputs, it's both a
testing framework and a library that can be used to write `expect` style tests.

`turnstile` is how we test all program transformations especially when it comes
to optimization passes and so on.

## References

* [CS6120: Advanced Compilers](https://www.cs.cornell.edu/courses/cs6120/2023fa/self-guided/)
* [LLVM Programmer's Manual](https://llvm.org/docs/ProgrammersManual.html)
* [Engineering a Compiler 3rd Edition](https://www.sciencedirect.com/book/9780128154120/engineering-a-compiler)
* [CSC255/455 Software Analysis and Improvement](https://www.cs.rochester.edu/~sree/courses/csc-255-455/spring-2020/schedule.html)
* [Data flow analysis: an informal introduction](https://clang.llvm.org/docs/DataFlowAnalysisIntro.html)
* [Notes on Graph Algorithms in Optimizing Compilers](https://www.cs.umb.edu/~offner/files/flow_graph.pdf)
* [SSA-based Compiler Design](https://link.springer.com/book/10.1007/978-3-030-80515-9)