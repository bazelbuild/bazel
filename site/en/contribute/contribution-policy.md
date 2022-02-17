Project: /_project.yaml
Book: /_book.yaml

# Contribution policy

This page covers Bazel's governance model and contribution policy.

## Governance model {:#governance-model}

The Bazel project is led by a core group of contributors and managed by the
community. At this time, all core contributors work for Google.
The group of core contributors is self-managing - core contributors are added
by two supporting votes from core contributors on the mailing list and no veto
within four business days. We expect that new contributors will submit a number
of patches before they become core contributors.

Contact the core team at <a href="mailto:bazel-core@googlegroups.com">
bazel-core@googlegroups.com</a>.

## Contribution policy {:#contribution-policy}

The Bazel project also accepts individual contributions from external
contributors that are not actively supporting the project.

The Bazel team uses the following rules for accepting code contributions.

1. We require all contributors to sign [Google's Contributor License
   Agreement](https://cla.developers.google.com/){: .external}.

1. We accept well-written, well-tested bug fixes to built-in rules.

1. We accept well-written, well-tested feature contributions if a core
   contributor assumes support responsibilities (for example, readily answers support
   questions and works on bugs). This includes feature contributions from
   external contributors. If there is no core contributor to support a
   feature, then we will deprecate and subsequently delete the feature - we will
   give three months' notice in such cases.

1. We will not accept untested changes, except in very rare cases.

1. We require a pre-commit code review from a core contributor for all changes.
   For the time being, we will have to continue making changes across the
   internal and external code bases, which will be reviewed internal to Google.

1. We will roll back changes if they break the internal development processes
   of any of the core contributors.

1. We will move towards an open governance model where multiple parties have
   commit access, roll-back rights, and can provide explicit support for
   features or rules.

1. We will work with interested parties to improve existing extension points
   and to establish new extension points if they do not run counter to the
   internal requirements of any of the core contributors.

For more details on contributing to Bazel, see our
[contribution guidelines](index.md).
