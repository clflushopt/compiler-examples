# Compiler Examples

This repo contains various reference implementations of compiler related data 
structures and algorithms.

Most of the following was initially done while working through Cornell's CS6120
[course](https://www.cs.cornell.edu/courses/cs6120/2023fa/self-guided/) from
the course assignments, reading papers, LLVM's codebase and books.

## Why

When I initially started studying compilers in depth, I spent a lot of my time
reading books and writing various toy compilers. 

The reasons I worked on end to end compilers is because I didn't find a suitable
playground where I could play with things like control flow graphs, SSA form or
data flow analyses.

While working on end to end projects I would make several design mistakes and
be too discouraged to refactor and fix it so I would start from scratch, again.

Writing a lexer, parser and semantic analyzers is non-trivial work and life is
too short to spend it writing lexers or recursive descent parsers or actually
dealing with the weird scoping rules of some languages.

My initial goal was to build an intuitive understanding of how compilers work
beyond the things you might find in your usual compiler book but I ended up
just writing more parsers than one should do in a lifetime.

The lack of good books that focus on the backend parts didn't help either, even
the venerable dragon book spends more time in parsing techniques than anyone
ever needs to know.

All these obstacles lead me to just drop everything and just learn theory from
books and papers, but the idea of a reference implementation, code that you can
read, play with and experiment on kept lurking in the back.

It wasn't until I discovered [Adrian's course](https://www.cs.cornell.edu/~asampson/)
that the approach really crystallized.

That's how the idea of `compiler-examples` started, the idea was to build all
the infrastructure around the `Bril` intermediate language and provide readable
refrence implementations of several data structures, analyses an optimizations.

One question that asks itself, is why use [Python](https://python.org) ? Well
the first iteration of this project was in C++, the second was in Rust and the
third in OCaml. The reason I settled on Python was because it is the English
of programming languages, I have yet to find a better language that gets out of
the way and allows me to simply express myself.

The implementation language is not important, in fact I encourage you to use this
as an opportunity to learn a programming language at the same time.

I hope you enjoy hacking on it as much as I enjoyed writing.
