# Governance

## Is Bazel developed fully in the open? {#isbazelopen}

Unfortunately not. We have a significant amount of code that is not open source, mostly rules,
unit tests and interfaces to internal Google systems. Initially we are open sourcing only ~10% of
the build rules used internally at Google. We did an experiment over the course of a few weeks
where we marked all changes that crossed the internal and external code bases, only to discover
that a lot of our changes still affect both.

This complicates the development process and means that some changes will simply appear in the Bazel
repository without clear explanation. At the same time, it is difficult to collaborate or get actual
users without actually having the code in the open. Therefore, we are opening up the code, even
though some of the development is still happening internal to Google.

## Accepting Contributions

Please also see our [contribution guidelines](contributing.md).

### Policy

We use the following rules for accepting code contributions. This is written from the perspective
that there is a group of people who cooperatively support the project (the *core contributors*). In
contrast, external contributors are not actively supporting the project, but just contributing
individual changes. At this time, all core contributors work for Google (see below for the full
list), but this will hopefully change over time.

1. We require all contributors to sign [Google's Contributor License
   Agreement](https://cla.developers.google.com/).

2. We accept well-written, well-tested contributions of rules written in
   [Skylark](skylark/concepts.md), in a `contrib/` directory or similar with clearly documented
   support policies.

3. We accept well-written, well-tested cleanup and refactoring changes.

4. We accept well-written, well-tested bug fixes to built-in rules.

5. We accept well-written, well-tested feature contributions if a core contributor assumes support
   responsibilities, i.e., readily answers support questions and works on bugs. This includes
   feature contributions from external contributors. If there is no core contributor to support a
   feature, then we will deprecate and subsequently delete the feature - we will give three months'
   notice in such cases.

6. We will not accept untested changes, except in very rare cases.

7. We require a pre-commit code review from a core contributor for all changes. For the time being,
   we will have to continue making changes across the internal and external code bases, which will
   be reviewed internal to Google.

8. We will roll back changes if they break the internal development processes of any of the core
   contributors.

9. We will move towards an open governance model where multiple parties have commit access,
   roll-back rights, and can provide explicit support for features or rules.

10. We will work with interested parties to improve existing extension points and to establish new
    extension points if they do not run counter to the internal requirements of any of the core
    contributors.

### Core Contributors

The group of core contributors is self-managing - core contributors are added by two supporting
votes from core contributors on the mailing list and no veto within four business days. We expect
that new contributors will submit a number of patches before they become core contributors.

__Contact the core team at bazel-discuss@googlegroups.com.__

The current group is:

`@google.com`

 - `dmarting`
 - `hanwen`
 - `jfield`
 - `kchodorow`
 - `laszlocsomor`
 - `laurentlb`
 - `lberki`
 - `philwo`
 - `ulfjack`
