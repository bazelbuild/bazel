Project: /_project.yaml
Book: /_book.yaml

# Bazel Special Interest Groups

{% include "_buttons.html" %}

Bazel hosts Special Interest Groups (SIGs) to focus collaboration on particular
areas and to support communication and coordination between [Bazel owners,
maintainers, and contributors](/contribute/policy). This policy
applies to [`bazelbuild`](http://github.com/bazelbuild){: .external}.

SIGs do their work in public. The ideal scope for a SIG covers a well-defined
domain, where the majority of participation is from the community. SIGs may
focus on community maintained repositories in `bazelbuild` (such as language
rules) or focus on areas of code in the Bazel repository (such as Remote
Execution).

While not all SIGs will have the same level of energy, breadth of scope, or
governance models, there should be sufficient evidence that there are community
members willing to engage and contribute should the interest group be
established. Before joining, review the group's work, and then get in touch
with the SIG leader. Membership policies vary on a per-SIG basis.

See the complete list of
[Bazel SIGs](https://github.com/bazelbuild/community/tree/main/sigs){: .external}.

### Non-goals: What a SIG is not

SIGs are intended to facilitate collaboration on shared work. A SIG is
therefore:

-   *Not a support forum:* a mailing list and a SIG is not the same thing
-   *Not immediately required:* early on in a project's life, you may not know
    if you have shared work or collaborators
-   *Not free labor:* energy is required to grow and coordinate the work
    collaboratively

Bazel Owners take a conservative approach to SIG creation—thanks to the ease of
starting projects on GitHub, there are many avenues where collaboration can
happen without the need for a SIG.

## SIG lifecycle

This section covers how to create a SIG.

### Research and consultation

To propose a new SIG group, first gather evidence for approval, as specified
below. Some possible avenues to consider are:

-   A well-defined problem or set of problems the group would solve
-   Consultation with community members who would benefit, assessing both the
    benefit and their willingness to commit
-   For existing projects, evidence from issues and PRs that contributors care
    about the topic
-   Potential goals for the group to achieve
-   Resource requirements of running the group

Even if the need for a SIG seems self-evident, the research and consultation is
still important to the success of the group.

### Create the new group

The new group should follow the below process for chartering. In particular, it
must demonstrate:

-   A clear purpose and benefit to Bazel (either around a sub-project or
    application area)
-   Two or more contributors willing to act as group leads, existence of other
    contributors, and evidence of demand for the group
-   Each group needs to use at least one publicly accessible mailing list. A SIG
    may reuse one of the public lists, such as
    [bazel-discuss](https://groups.google.com/g/bazel-discuss), ask for a list
    for @bazel.build, or create their own list
-   Resources the SIG initially requires (usually, mailing list and regular
    video call.)
-   SIGs can serve documents and files from their directory in
    [`bazelbuild/community`](https://github.com/bazelbuild/community){: .external}
    or from their own repository in the
    [`bazelbuild`](https://github.com/bazelbuild){: .external} GitHub
    organization. SIGs may link to external resources if they choose to organize
    their work outside of the `bazelbuild` GitHub organization
-   Bazel Owners approve or reject SIG applications and consult other
    stakeholders as necessary

Before entering the formal parts of the process, you should consult with
the Bazel product team, at product@bazel.build. Most SIGs require conversation
and iteration before approval.

The formal request for the new group is done by submitting a charter as a PR to
[`bazelbuild/community`](https://github.com/bazelbuild/community){: .external},
and including the request in the comments on the PR following the template
below. On approval, the PR for the group is merged and the required resources
created.

### Template Request for New SIG

To request a new SIG, use the template in the community repo:
[SIG-request-template.md](https://github.com/bazelbuild/community/blob/main/governance/SIG-request-template.md){: .external}.

### Chartering

To establish a group, you need a charter and must follow the Bazel
[code of conduct](https://github.com/bazelbuild/bazel/blob/HEAD/CODE_OF_CONDUCT.md){: .external}.
Archives of the group will be public. Membership may either be open to all
without approval, or available on request, pending approval of the group
administrator.

The charter must nominate an administrator. As well as an administrator, the
group must include at least one person as lead (these may be the same person),
who serves as point of contact for coordination as required with the Bazel
product team.

Group creators must post their charter to the group mailing list. The community
repository in the Bazel GitHub organization archives such documents and
policies. As groups evolve their practices and conventions, they should update
their charters within the relevant part of the community repository.

### Collaboration and inclusion

While not mandated, the group should choose to make use of collaboration
via scheduled conference calls or chat channels to conduct meetings. Any such
meetings should be advertised on the mailing list, and notes posted to the
mailing list afterwards. Regular meetings help drive accountability and progress
in a SIG.

Bazel product team members may proactively monitor and encourage the group to
discussion and action as appropriate.

### Launch a SIG

Required activities:

-   Notify Bazel general discussion groups
    ([bazel-discuss](https://groups.google.com/g/bazel-discuss){: .external},
    [bazel-dev](https://groups.google.com/g/bazel-dev){: .external}).

Optional activities:

-   Create a blog post for the Bazel blog

### Health and termination of SIGs

The Bazel owners make a best effort to ensure the health of SIGs. Bazel owners
occasionally request the SIG lead to report on the SIG's work, to inform the
broader Bazel community of the group's activity.

If a SIG no longer has a useful purpose or interested community, it may be
archived and cease operation. The Bazel product team reserves the right to
archive such inactive SIGs to maintain the overall health of the project,
though it is a less preferable outcome. A SIG may also opt to disband if
it recognizes it has reached the end of its useful life.

## Note

*This content has been adopted from Tensorflow’s
[SIG playbook](https://www.tensorflow.org/community/sig_playbook){: .external}
with modifications.*
