---
layout: contribute
title: Skylark Design Review Process
---

# Skylark Design Review Process

Authors: [Dmitry Lomov](mailto:dslomov@google.com),
[Laurent Le Brun](mailto:laurentlb@google.com), [Florian Weikert](mailto:fwe@google.com)


## Motivation

As Bazel adoption grows, we are seeing both more requests for adding
features to Skylark as a language, and for exposing more and more
functionality available in Bazel internally to Skylark extensions.
These requests originate both within and outside the Bazel team. As
their number grows (and we expect them to grow more and more over time),
addressing those requests meets several challenges:

* We need to keep Skylark as a language (and a set of
associated APIs/libraries) concise and consistent, easy to learn
and use, and documented.

* Any APIs or solutions we adopt will be painful to change
down the road, so the more consistent, orthogonal, and open
to future extensions they are, the less pain we and our users
will encounter.

* It is difficult for engineers - both within the team and outside -
to approach making changes to Skylark. As people attempt to do so,
they experience a lot of friction: patches get written and the discussion
starts where reviewers and the author attempt to solve Skylark
API issues, code design issues, implementation issues, compatibility
issues and so on and so forth all in one code review thread.
The result is friction and frustration, and the quality of the end result
is not guaranteed.

This document proposes an informal but orderly and well-defined process
for Skylark language changes to address the above challenges.

## Goals

* Facilitate healthy growth of Skylark as a language

* Ensure that Skylark the language has a clear set of experts caring for its development

* Reduce friction for engineers exposing APIs to Skylark or proposing new Skylark features

## Non-goals

* Replace general design review process in Bazel team.
The process described here is strictly about the Skylark language and APIs.
In particular, changes to native rules that do not affect Skylark as a
language much, such as adding/removing an attribute, or adding a new
native rule, are not required to go through this process
(whereas exposing a new provider to Skylark from a native rule is).

## The Process

Changes to Skylark, both the language and the exposed APIs,
are evaluated by the Skylark Reviewers Panel.


1. The **author** of the proposed change sends the design document to the
[bazel-dev@googlegroups.com] mailing list with a subject containing
"SKYLARK DESIGN REVIEW"

1. Design doc can be either of:
    1. A universally-commentable Google Doc
    1. A Gerrit code review request with a design doc in
       [Markdown](https://guides.github.com/features/mastering-markdown/)
       format.

1. The design doc should include:
    1. Motivation for the change (a GitHub issue link is acceptable)
    2. An example of the usage of the proposed API/language feature
    3. Description of the proposed change

    [A model example design doc](/designs/skylark/parameterized-aspects.html)
    (although that is probably an overkill).

1. **Skylark Reviewers** respond to a document within *2 business days*

1. **Skylark Reviewers** are responsible for bringing in
 **subject-matter experts** as needed (for example, a change involving
 exposing a certain provider from cc_library rule should be reviewed by
  C++ rules expert as well)

1. **Skylark Reviewers** facilitate the discussion and aim to reach
a "looks good/does not look good" decision within *10 business days*.

1. **Skylark Reviewers** operate by consensus.
