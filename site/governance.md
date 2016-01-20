---
layout: community
title: Governance
---

# Governance

The Bazel project is led by a core group of contributors, initially Googlers, and managed by the
community. The group of core contributors is self-managing - core contributors are added by two
supporting votes from core contributors on the mailing list and no veto within four business days.
We expect that new contributors will submit a number of patches before they become core
contributors.

## Accepting Contributions

Please also see our [contribution guidelines](contributing.html).

### Policy

We use the following rules for accepting code contributions. This is written from the perspective
that there is a group of people who cooperatively support the project (the *core contributors*). In
contrast, external contributors are not actively supporting the project, but just contributing
individual changes. At this time, all core contributors work for Google (see below for the full
list), but this will hopefully change over time.

1. We require all contributors to sign [Google's Contributor License
   Agreement](https://cla.developers.google.com/).

2. We accept well-written, well-tested contributions of rules written in
   [Skylark](docs/skylark/concepts.html), in a `contrib/` directory or similar with clearly documented
   support policies.

3. We accept well-written, well-tested bug fixes to built-in rules.

4. We accept well-written, well-tested feature contributions if a core contributor assumes support
   responsibilities, i.e., readily answers support questions and works on bugs. This includes
   feature contributions from external contributors. If there is no core contributor to support a
   feature, then we will deprecate and subsequently delete the feature - we will give three months'
   notice in such cases.

5. We will not accept untested changes, except in very rare cases.

6. We require a pre-commit code review from a core contributor for all changes. For the time being,
   we will have to continue making changes across the internal and external code bases, which will
   be reviewed internal to Google.

7. We will roll back changes if they break the internal development processes of any of the core
   contributors.

8. We will move towards an open governance model where multiple parties have commit access,
   roll-back rights, and can provide explicit support for features or rules.

9. We will work with interested parties to improve existing extension points and to establish new
    extension points if they do not run counter to the internal requirements of any of the core
    contributors.

## Are you done open sourcing Bazel?

Open sourcing Bazel is a work-in-progress. In particular, we're still working on open sourcing:

* Many of our unit and integration tests (which should make contributing patches easier).
* Full IDE integration.

Beyond code, we'd like to eventually have all code reviews, bug tracking, and design decisions
happen publicly, with the Bazel community involved. We are not there yet, so some changes will
simply appear in the Bazel repository without clear explanation. Despite this lack of
transparency, we want to support external developers and collaborate. Thus, we are opening up the
code, even though some of the development is still happening internal to Google. Please let us know
if anything seems unclear or unjustified as we transition to an open model.

## Are there parts of Bazel that will never be open sourced?

Yes, some of the code base either integrates with Google-specific technology or we have been looking
for an excuse to get rid of (or is some combination of the two). These parts of the code base are
not available on GitHub and probably never will be.

### Core Contributors

<p class="lead">
Contact the core team at <a href="mailto:bazel-dev@googlegroups.com">
bazel-dev@googlegroups.com</a>.
</p>

The current group is:

 - [damienmg](https://github.com/damienmg)
 - [hanwen](https://github.com/hanwen)
 - [jhfield](https://github.com/jhfield)
 - [kchodorow](https://github.com/kchodorow)
 - [laszlocsomor](https://github.com/laszlocsomor)
 - [laurentlb](https://github.com/laurentlb)
 - [lberki](https://github.com/lberki)
 - [philwo](https://github.com/philwo)
 - [ulfjack](https://github.com/ulfjack)
