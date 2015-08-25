---
layout: community
---

# Support Policy

We generally avoid making backwards-incompatible changes. At Google, we
run all of the tests in the entire depot before every release and
check that there are no regressions. It is much more difficult to do
that outside of Google, because there is no single source repository
that contains everything.

All undocumented features (attributes, rules, "Make" variables, and flags) are subject to change at
any time without prior notice. Features that are documented but marked *experimental* are also
subject to change at any time without prior notice. The Skylark macro and rules language (anything
you write in a `.bzl` file) is still subject to change.

Bugs can be reported in the
[GitHub bugtracker](https://github.com/google/bazel/issues). We will make an effort to triage all
reported issues within 2 business days.

## Releases

We try to do [monthly binary releases of
Bazel](https://github.com/google/bazel/releases). A released version of Bazel
should be free of regression and extensively tested. The release process is the
following:

  - A baseline is tested extensively inside Google. When considered stable
  inside Google, a Bazel release candidate is announced in
  [bazel-discuss](bazel-discuss@googlegroups.com) for testing.
  - Subsequent cherry-pick will be done to create new release candidate if
  regression are discovered on the release candidate.
  - After at least one week since the first candidate and two full business days
  since the last candidate, if no regression were found, a release will be
  emitted and announced in [bazel-discuss](bazel-discuss@googlegroups.com).

Thus, all our releases are tested with the extensive test suite we have inside
Google but also with our public continuous test infrastructure and user tested
both inside and outside Google.

### Release versioning

Version 0.1 is our first release marking the start of our beta phase. Until
version 1.0, each MINOR version increases will be performed when reaching a
[new milestone](http://bazel.io/roadmap.html), otherwise only the PATCH
version will be increased on a new release.

Version 1.0 will be the end of our beta phase and we will label each release
with a version number according to the [semantic version 2.0.0
document](http://semver.org). By the time we reach version 1.0, we will define
clearly what is included in our API.

## Current Status

### Fully Supported
We make no breaking changes to the rules, or provide instructions on how to migrate. We actively fix
issues that are reported, and also keep up with changes in the underlying tools. We ensure that all
the tests pass.

<table class="table table-condensed table-striped table-bordered">
  <colgroup>
    <col width="30%"/>
    <col/>
  </colgroup>
  <thead>
    <tr>
      <th>Rules</th>
      <th>Notes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>C/C++ rules except <code>cc_toolchain</code></td>
      <td></td>
    </tr>
    <tr>
      <td>Java rules</td>
      <td></td>
    </tr>
    <tr>
      <td><code>genrule</code></td>
      <td></td>
    </tr>
    <tr>
      <td><code>test_suite</code></td>
      <td></td>
    </tr>
    <tr>
      <td><code>filegroup</code></td>
      <td></td>
    </tr>
  </tbody>
</table>


### Partially Supported
We avoid breaking changes when possible. We actively fix issues that are reported, but may fall
behind the current state of the tools. We ensure that all the tests pass.

<table class="table table-condensed table-striped table-bordered">
  <colgroup>
    <col width="30%"/>
    <col/>
  </colgroup>
  <thead>
    <tr>
      <th>Rules</th>
      <th>Notes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>cc_toolchain</code></td>
      <td>
        <ul>
          <li>We intend to make significant changes to the way C/C++ toolchains are defined; we will
            keep our published C/C++ toolchain definition(s) up to date, but we make no guarantees for
            custom ones.</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>iOS/Objective C rules</td>
      <td>
        <ul>
          <li>We cannot vouch for changes made by Apple &reg; to the underlying tools and
            infrastructure.</li>
          <li>The rules are fairly new and still subject to change; we try to avoid breaking changes,
            but this may not always be possible.</li>
          <li>No testing support yet.</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>Extra actions (<code>extra_action</code>, <code>action_listener</code>)</td>
      <td>
        <ul>
          <li>Extra actions expose information about Bazel that we consider to be implementation
            details, such as the exact interface between Bazel and the tools we provide; as such,
            users will need to keep up with changes to tools to avoid breakage.</li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

### Built-In Rules and the Internal API For Rules ###
We are planning a number of changes to the APIs between the core of Bazel and the built-in rules,
in order to make it easier for us to develop openly. This has the added benefit of also making it
easier for users to maintain their own rules (if written in Java instead of Skylark), if they don't
want to or cannot check this code into our repository. However, it also means our internal API is
not stable yet. In the long term, we want to move to Skylark wholesale, so we encourage contributors
to use Skylark instead of Java when writing new rules. Rewriting all of our built-in rules is going
to be a lengthy process however.

1. We will fix the friction points that we know about, as well as those that we discover every time
   we make changes that span both the internal and external depots.
2. We will drive a number of pending API cleanups to completion, as well as run anticipated cleanups
   to the APIs, such as disallowing access to the file system from rule implementations (because
   it's not hermetic).
3. We will enumerate the internal rule APIs, and make sure that they are appropriately marked (for
   example with annotations) and documented. Just collecting a list will likely give us good
   suggestions for further improvements, as well as opportunities for a more principled API review
   process.
4. We will automatically check rule implementations against an API whitelist, with the intention
   that API changes are implicitly flagged during code review.
5. We will work on removing (legacy) Google-internal features to reduce the amount of differences
   between the internal and external rule sets.
6. We will encourage our engineers to make changes in the external depot first, and migrate them to
   to the internal one afterwards.
7. We will move more of our rule implementations into the open source repository (Android, Go,
   Python, the remaining C++ rules), even if we don't consider the code to be *ready* or if they are
   still missing tools to work properly.
8. In order to be able to accept external contributions, our highest priority item for Skylark is a
   testing framework. We encourage to write new rules in Skylark rather than in Java.


### Best Effort
We will not break existing tests, but otherwise make no dedicated effort to keep the rules working
or up-to-date.

<table class="table table-condensed table-striped table-bordered">
  <colgroup>
    <col width="30%"/>
    <col/>
  </colgroup>
  <thead>
    <tr>
      <th>Rules</th>
      <th>Notes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>Fileset</code></td>
      <td>
        <ul>
          <li>There are vestiges of Fileset / FilesetEntry in the source code, but we do not intend to
            support them in Bazel, ever.</li>
          <li>They're still widely used internally, and are therefore unlikely to go away in the near
            future.</li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>
