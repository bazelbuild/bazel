# Support Policy

We generally avoid making backward-incompatible changes. At Google, we
run all of the tests in the entire depot before every release and
check that there are no regressions. It is much more difficult to do
that outside of Google, because there is no single source repository
that contains everything.

All undocumented features (attributes, rules, "Make" variables, and flags) are subject to change at
any time without prior notice. Features that are documented but marked *experimental* are also
subject to change at any time without prior notice. The Skylark macro and rules language (anything
you write in a `.bzl` file) is still subject to change.

Bugs can be reported in the
[GitHub bugtracker](https://github.com/google/bazel/issues). We will
make an effort to triage all reported issues within 2 business days; we will measure our triaging
process and regularly report numbers.

## Current Status

### Fully Supported
We make no breaking changes to the rules, or provide instructions on how to migrate. We actively fix
issues that are reported, and also keep up with changes in the underlying tools. We ensure that all
the tests pass.

<table>
<colgroup><col width="30%"/><col/></colgroup>
  <tr>
    <th>Rules</th>
    <th>Notes</th>
  </tr>
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
</table>


### Partially Supported
We avoid breaking changes when possible. We actively fix issues that are reported, but may fall
behind the current state of the tools. We ensure that all the tests pass.

<table>
<colgroup><col width="30%"/><col/></colgroup>
  <tr>
    <th>Rules</th>
    <th>Notes</th>
  </tr>
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
</table>

#### A Note on the Internal Rules API
We are planning a number of changes to the APIs between the core of Bazel and the built-in rules,
in order to make it easier for us to develop openly. This has the added benefit of also making it
easier for users to maintain their custom rules, if they don't want to or can't check this code into
our repository. However, it also means our internal API is not stable yet. In the long term, we
want to move to Skylark wholesale, and we encourage contributors to use Skylark for new rules
development, but rewriting all of our built-in rules is going to be a lengthy process.

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
   testing framework.


### Best Effort
We will not break existing tests, but otherwise make no dedicated effort to keep the rules working
or up-to-date.

<table>
<colgroup><col width="30%"/><col/></colgroup>
  <tr>
    <th>Rules</th>
    <th>Notes</th>
  </tr>
  <tr>
    <td>Go rules (Skylark)</td>
    <td>
      <ul>
        <li>These rules are an experiment with Skylark, and are not the same code as the rules we
          use internally, even though we tried to match the semantics.</li>
        <li>They are not tested very extensively.</li>
      </ul>
    </td>
  </tr>
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
</table>

