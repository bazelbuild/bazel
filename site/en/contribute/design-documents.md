Project: /_project.yaml
Book: /_book.yaml

# Design Documents

{% include "_buttons.html" %}

If you're planning to add, change, or remove a user-facing feature, or make a
*significant architectural change* to Bazel, you **must** write a design
document and have it reviewed before you can submit the change.

Here are some examples of significant changes:

*   Addition or deletion of native build rules
*   Breaking-changes to native rules
*   Changes to a native build rule semantics that affect the behavior of more
    than a single rule
*   Changes to Bazel's rule definition API
*   Changes to the APIs that Bazel uses to connect to other systems
*   Changes to the Starlark language, semantics, or APIs
*   Changes that could have a pervasive effect on Bazel performance or memory
    usage (for better or for worse)
*   Changes to widely used internal APIs
*   Changes to flags and command-line interface.

## Reasons for design reviews {:#design-reviews}

When you write a design document, you can coordinate with other Bazel developers
and seek guidance from Bazel's core team. For example, when a proposal adds,
removes, or modifies any function or object available in BUILD, MODULE.bazel, or
bzl files, add the [Starlark team](maintainers-guide.md) as reviewers.
Design documents are reviewed before submission because:

*   Bazel is a very complex system; seemingly innocuous local changes can have
    significant global consequences.
*   The team gets many feature requests from users; such requests need to be
    evaluated not only for technical feasibility but importance with regards to
    other feature requests.
*   Bazel features are frequently implemented by people outside the core team;
    such contributors have widely varying levels of Bazel expertise.
*   The Bazel team itself has varying levels of expertise; no single team member
    has a complete understanding of every corner of Bazel.
*   Changes to Bazel must account for backward compatibility and avoid breaking
    changes.

Bazel's design review policy helps to maximize the likelihood that:

*   all feature requests get a baseline level of scrutiny.
*   the right people will weigh in on designs before we've invested in an
    implementation that may not work.

To help you get started, take a look at the design documents in the
[Bazel Proposals Repository](https://github.com/bazelbuild/proposals){: .external}.
Designs are works in progress, so implementation details can change over time
and with feedback. The published design documents capture the initial design,
and *not* the ongoing changes as designs are implemented. Always go to the
documentation for descriptions of current Bazel functionality.

## Contributor Workflow {:#contributor-workflow}

As a contributor, you can write a design document, send pull requests and
request reviewers for your proposal.

### Write the design document {:#write-design-doc}

All design documents must have a header that includes:

*   author
*   date of last major change
*   list of reviewers, including one (and only one)
    [lead reviewer](#lead-reviewer)
*   current status (_draft_, _in review_, _approved_, _rejected_,
    _being implemented_, _implemented_)
*   link to discussion thread (_to be added after the announcement_)

The document can be written either [as a world-readable Google Doc](#gdocs)
or [using Markdown](#markdown). Read below about for a
[Markdown / Google Docs comparison](#markdown-versus-gdocs).

Proposals that have a user-visible impact must have a section documenting the
impact on backward compatibility (and a rollout plan if needed).

### Create a Pull Request {:#pull-request}

Share your design doc by creating a pull request (PR) to add the document to
[the design index](https://github.com/bazelbuild/proposals){: .external}. Add
your markdown file or a document link to your PR.

When possible, [choose a lead reviewer](#lead-reviewer).
and cc other reviewers. If you don't choose a lead reviewer, a Bazel
maintainer will assign one to your PR.

After you create your PR, reviewers can make preliminary comments during the
code review. For example, the lead reviewer can suggest extra reviewers, or
point out missing information. The lead reviewer approves the PR when they
believe the review process can start. This doesn't mean the proposal is perfect
or will be approved; it means that the proposal contains enough information to
start the discussion.

### Announce the new proposal {:#new-proposal}

Send an announcement to
[bazel-dev](https://groups.google.com/forum/#!forum/bazel-dev){: .external} when
the PR is submitted.

You may copy other groups (for example,
[bazel-discuss](https://groups.google.com/forum/#!forum/bazel-discuss){: .external},
to get feedback from Bazel end-users).

### Iterate with reviewers {:#reviewers}

Anyone interested can comment on your proposal. Try to answer questions,
clarify the proposal, and address concerns.

Discussion should happen on the announcement thread. If the proposal is in a
Google Doc, comments may be used instead (Note that anonymous comments are
allowed).

### Update the status {:#update-status}

Create a new PR to update the status of the proposal, when iteration is
complete. Send the PR to the same lead reviewer and cc the other reviewers.

To officially accept the proposal, the lead reviewer approves the PR after
ensuring that the other reviewers agree with the decision.

There must be at least 1 week between the first announcement and the approval of
a proposal. This ensures that users had enough time to read the document and
share their concerns.

Implementation can begin before the proposal is accepted, for example as a
proof-of-concept or an experimentation. However, you cannot submit the change
before the review is complete.

### Choosing a lead reviewer {:#lead-reviewer}

A lead reviewer should be a domain expert who is:

*   Knowledgeable of the relevant subsystems
*   Objective and capable of providing constructive feedback
*   Available for the entire review period to lead the process

Consider checking the contacts for various [team
labels](/contribute/maintainers-guide#team-labels).

## Markdown vs Google Docs {:#markdown-versus-gdocs}

Decide what works best for you, since both are accepted.

Benefits of using Google Docs:

* Effective for brainstorming, since it is easy to get started with.
* Collaborative editing.
* Quick iteration.
* Easy way to suggest edits.

Benefits of using Markdown files:

*   Clean URLs for linking.
*   Explicit record of revisions.
*   No forgetting to set up access rights before publicizing a link.
*   Easily searchable with search engines.
*   Future-proof: Plain text is not at the mercy of any specific tool
    and doesn't require an Internet connection.
*   It is possible to update them even if the author is not around anymore.
*   They can be processed automatically (update/detect dead links, fetch
    list of authors, etc.).

You can choose to first iterate on a Google Doc, and then convert it to
Markdown for posterity.

### Using Google Docs {:#gdocs}

For consistency, use the [Bazel design doc template](
https://docs.google.com/document/d/1cE5zrjrR40RXNg64XtRFewSv6FrLV6slGkkqxBumS1w/edit){: .external}.
It includes the necessary header and creates visual
consistency with other Bazel related documents. To do that, click on **File** >
**Make a copy** or click this link to [make a copy of the design doc
template](https://docs.google.com/document/d/1cE5zrjrR40RXNg64XtRFewSv6FrLV6slGkkqxBumS1w/copy){: .external}.

To make your document readable to the world, click on
**Share** > **Advanced** > **Change…**, and
choose "On - Anyone with the link".  If you allow comments on the document,
anyone can comment anonymously, even without a Google account.

### Using Markdown {:#markdown}

Documents are stored on GitHub and use the
[GitHub flavor of Markdown](https://guides.github.com/features/mastering-markdown/){: .external}
([Specification](https://github.github.com/gfm/){: .external}).

Create a PR to update an existing document. Significant changes should be
reviewed by the document reviewers. Trivial changes (such as typos, formatting)
can be approved by anyone.

## Reviewer workflow {:#reviewer-workflow}

A reviewer comments, reviews and approves design documents.

### General reviewer responsibilities {:#reviewer-responsibilities}

You're responsible for reviewing design documents, asking for additional
information if needed, and approving a design that passes the review process.

#### When you receive a new proposal {:#new-proposal}

1.  Take a quick look at the document.
1.  Comment if critical information is missing, or if the design doesn't fit
    with the goals of the project.
1.  Suggest additional reviewers.
1.  Approve the PR when it is ready for review.

#### During the review process {:#during-review-process}

1. Engage in a dialogue with the design author about issues that are problematic
   or require clarification.
1. If appropriate, invite comments from non-reviewers who should be aware of
   the design.
1. Decide which comments must be addressed by the author as a prerequisite to
   approval.
1. Write "LGTM" (_Looks Good To Me_) in the discussion thread when you are
   happy with the current state of the proposal.

Follow this process for all design review requests. Do not approve designs
affecting Bazel if they are not in the
[design index](https://github.com/bazelbuild/proposals){: .external}.

### Lead reviewer responsibilities {:#lead-reviewer-responsibilities}

You're responsible for making the go / no-go decision on implementation
of a pending design. If you're not able to do this, you should identify a
suitable delegate (reassign the PR to the delegate), or reassign the bug to a
Bazel manager for further disposition.

#### During the review process {:#during-process}

1.  Ensure that the comment and design iteration process moves forward
    constructively.
1.  Prior to approval, ensure that concerns from other reviewers have been
    resolved.

#### After approval by all reviewers {:#after-approval}

1.  Make sure there has been at least 1 week since the announcement on the
    mailing list.
1.  Make sure the PR updates the status.
1.  Approve the PR sent by the proposal author.

#### Rejecting designs {:#reject-designs}

1.  Make sure the PR author sends a PR; or send them a PR.
1.  The PR updates the status of the document.
1.  Add a comment to the document explaining why the design can't be approved in
    its current state, and outlining next steps, if any (such as "revisit invalid
    assumptions and resubmit").
