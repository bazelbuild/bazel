// Copyright 2014 Google Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.docgen;

import com.google.common.collect.ImmutableMap;

import java.util.Map;

/**
 * A class to contain the base definition of common BUILD rule attributes.
 */
public class PredefinedAttributes {

  public static final Map<String, RuleDocumentationAttribute> COMMON_ATTRIBUTES = ImmutableMap
      .<String, RuleDocumentationAttribute>builder()
      .put("deps", RuleDocumentationAttribute.create("deps", DocgenConsts.COMMON_ATTRIBUTES,
          "A list of dependencies of this rule.\n"
        + "<i>(List of <a href=\"build-ref.html#labels\">labels</a>; optional)</i><br/>\n"
        + "The precise semantics of what it means for this rule to depend on\n"
        + "another using <code>deps</code> are specific to the kind of this rule,\n"
        + "and the rule-specific documentation below goes into more detail.\n"
        + "At a minimum, though, the targets named via <code>deps</code> will\n"
        + "appear in the <code>*.runfiles</code> area of this rule, if it has\n"
        + "one.\n"
        + "<p>Most often, a <code>deps</code> dependency is used to allow one\n"
        + "module to use symbols defined in another module written in the\n"
        + "same programming language and separately compiled.  Cross-language\n"
        + "dependencies are also permitted in many cases: for example,\n"
        + "a <code>java_library</code> rule may depend on C++ code in\n"
        + "a <code>cc_library</code> rule, by declaring the latter in\n"
        + "the <code>deps</code> attribute.  See the definition\n"
        + "of <a href=\"build-ref.html#deps\">dependencies</a> for more\n"
        + "information.</p>\n"
        + "<p>Almost all rules permit a <code>deps</code> attribute, but where\n"
        + "this attribute is not allowed, this fact is documented under the\n"
        + "specific rule.</p>"))
      .put("data", RuleDocumentationAttribute.create("data", DocgenConsts.COMMON_ATTRIBUTES,
          "The list of files needed by this rule at runtime.\n"
        + "<i>(List of <a href=\"build-ref.html#labels\">labels</a>; optional)</i><br/>\n"
        + "Targets named in the <code>data</code> attribute will appear in\n"
        + "the <code>*.runfiles</code> area of this rule, if it has one.  This\n"
        + "may include data files needed by a binary or library, or other\n"
        + "programs needed by it.  See the\n"
        + "<a href=\"build-ref.html#data\">data dependencies</a> section for more\n"
        + "information about how to depend on and use data files.\n"
        + "<p>Almost all rules permit a <code>data</code> attribute, but where\n"
        + "this attribute is not allowed, this fact is documented under the\n"
        + "specific rule.</p>"))
      .put("licenses", RuleDocumentationAttribute.create("licenses",
          DocgenConsts.COMMON_ATTRIBUTES,
          "<i>(List of strings; optional)</i><br/>\n"
        + "A list of license-type strings to be used for this particular build rule.\n"
        + "Overrides the <code>BUILD</code>-file scope defaults defined by the\n"
        + "<a href=\"#licenses\"><code>licenses()</code></a> directive."))
      .put("distribs", RuleDocumentationAttribute.create("distribs",
          DocgenConsts.COMMON_ATTRIBUTES,
          "<i>(List of strings; optional)</i><br/>\n"
        + "A list of distribution-method strings to be used for this particular build rule.\n"
        + "Overrides the <code>BUILD</code>-file scope defaults defined by the\n"
        + "<a href=\"#distribs\"><code>distribs()</code></a> directive."))
      .put("deprecation", RuleDocumentationAttribute.create("deprecation",
          DocgenConsts.COMMON_ATTRIBUTES,
          "<i>(String; optional)</i><br/>\n"
        + "An explanatory warning message associated with this rule.\n"
        + "Typically this is used to notify users that a rule has become obsolete,\n"
        + "or has become superseded by another rule, is private to a package, or is\n"
        + "perhaps \"considered harmful\" for some reason. It is a good idea to include\n"
        + "some reference (like a webpage, a bug number or example migration CLs) so\n"
        + "that one can easily find out what changes are required to avoid the message.\n"
        + "If there is a new target that can be used as a drop in replacement, it is a good idea\n"
        + "to just migrate all users of the old target.\n"
        + "<p>\n"
        + "This attribute has no effect on the way things are built, but it\n"
        + "may affect a build tool's diagnostic output.  The build tool issues a\n"
        + "warning when a rule with a <code>deprecation</code> attribute is\n"
        + "depended upon by another rule.</p>\n"
        + "<p>\n"
        + "Intra-package dependencies are exempt from this warning, so that,\n"
        + "for example, building the tests of a deprecated rule does not\n"
        + "encounter a warning.</p>\n"
        + "<p>\n"
        + "If a deprecated rule depends on another deprecated rule, no warning\n"
        + "message is issued.</p>\n"
        + "<p>\n"
        + "Once people have stopped using it, the package can be removed or marked as\n"
        + "<a href=\"#common.obsolete\"><code>obsolete</code></a>.</p>"))
      .put("obsolete", RuleDocumentationAttribute.create("obsolete",
          DocgenConsts.COMMON_ATTRIBUTES,
          "<i>(Boolean; optional; default 0)</i><br/>\n"
        + "If 1, only obsolete targets can depend on this target. It is an error when\n"
        + "a non-obsolete target depends on an obsolete target.\n"
        + "<p>\n"
        + "As a transition, one can first mark a package as in\n"
        + "<a href=\"#common.deprecation\"><code>deprecation</code></a>.</p>\n"
        + "<p>\n"
        + "This attribute is useful when you want to prevent a target from\n"
        + "being used but are yet not ready to delete the sources.</p>"))
      .put("testonly", RuleDocumentationAttribute.create("testonly",
          DocgenConsts.COMMON_ATTRIBUTES,
          "<i>(Boolean; optional; default 0 except as noted)</i><br />\n"
        + "If 1, only testonly targets (such as tests) can depend on this target.\n"
        + "<p>Equivalently, a rule that is not <code>testonly</code> is not allowed to\n"
        + "depend on any rule that is <code>testonly</code>.</p>\n"
        + "<p>Tests (<code>*_test</code> rules)\n"
        + "and test suites (<a href=\"#test_suite\">test_suite</a> rules)\n"
        + "are <code>testonly</code> by default.</p>\n"
        + "<p>By virtue of\n"
        + "<a href=\"#package.default_testonly\"><code>default_testonly</code></a>,\n"
        + "targets under <code>javatests</code> are <code>testonly</code> by default.</p>\n"
        + "<p>This attribute is intended to mean that the target should not be\n"
        + "contained in binaries that are released to production.</p>\n"
        + "<p>Because testonly is enforced at build time, not run time, and propagates\n"
        + "virally through the dependency tree, it should be applied judiciously. For\n"
        + "example, stubs and fakes that\n"
        + "are useful for unit tests may also be useful for integration tests\n"
        + "involving the same binaries that will be released to production, and\n"
        + "therefore should probably not be marked testonly. Conversely, rules that\n"
        + "are dangerous to even link in, perhaps because they unconditionally\n"
        + "override normal behavior, should definitely be marked testonly.</p>"))
      .put("tags", RuleDocumentationAttribute.create("tags", DocgenConsts.COMMON_ATTRIBUTES,
          "List of arbitrary text tags.  Tags may be any valid string; default is the\n"
        + "empty list.<br/>\n"
        + "<i>Tags</i> can be used on any rule; but <i>tags</i> are most useful\n"
        + "on test and <code>test_suite</code> rules.  Tags on non-test rules\n"
        + "are only useful to humans and/or external programs.\n"
        + "<i>Tags</i> are generally used to annotate a test's role in your debug\n"
        + "and release process.  Typically, tags are most useful for C++ and\n"
        + "Python tests, which\n"
        + "lack any runtime annotation ability.  The use of tags and size elements\n"
        + "gives flexibility in assembling suites of tests based around codebase\n"
        + "check-in policy.\n"
        + "<p>\n"
        + "A few tags have special meaning to the build tool, such as\n"
        + "indicating that a particular test cannot be run remotely, for\n"
        + "example. Consult\n"
        + "the <a href='blaze-user-manual.html#tags_keywords'>Blaze\n"
        + "documentation</a> for details.\n"
        + "</p>"))
      .put("visibility", RuleDocumentationAttribute.create("visibility",
          DocgenConsts.COMMON_ATTRIBUTES,
          "<i>(List of <a href=\"build-ref.html#labels\">"
        + "labels</a>; optional; default private)</i><br/>\n"
        + "<p>The <code>visibility</code> attribute on a rule controls whether\n"
        + "the rule can be used by other packages. Rules are always visible to\n"
        + "other rules declared in the same package.</p>\n"
        + "<p>There are five forms (and one temporary form) a visibility label can take:\n"
        + "<ul>\n"
        + "<li><code>[\"//visibility:public\"]</code>: Anyone can use this rule.</li>\n"
        + "<li><code>[\"//visibility:private\"]</code>: Only rules in this package\n"
        + "can use this rule.  Rules in <code>javatests/foo/bar</code>\n"
        + "can always use rules in <code>java/foo/bar</code>.\n"
        + "</li>\n"
        + "<li><code>[\"//some/package:__pkg__\", \"//other/package:__pkg__\"]</code>:\n"
        + "Only rules in <code>some/package</code> and <code>other/package</code>\n"
        + "(defined in <code>some/package/BUILD</code> and\n"
        + "<code>other/package/BUILD</code>) have access to this rule. Note that\n"
        + "sub-packages do not have access to the rule; for example,\n"
        + "<code>//some/package/foo:bar</code> or\n"
        + "<code>//other/package/testing:bla</code> wouldn't have access.\n"
        + "<code>__pkg__</code> is a special target and must be used verbatim.\n"
        + "It represents all of the rules in the package.\n"
        + "</li>\n"
        + "<li><code>[\"//project:__subpackages__\", \"//other:__subpackages__\"]</code>:\n"
        + "Only rules in packages <code>project</code> or <code>other</code> or\n"
        + "in one of their sub-packages have access to this rule. For example,\n"
        + "<code>//project:rule</code>, <code>//project/library:lib</code> or\n"
        + "<code>//other/testing/internal:munge</code> are allowed to depend on\n"
        + "this rule (but not <code>//independent:evil</code>)\n"
        + "</li>\n"
        + "<li><code>[\"//some/package:my_package_group\"]</code>:\n"
        + "A <a href=\"#package_group\">package group</a> is\n"
        + "a named set of package names. Package groups can also grant access rights\n"
        + "to entire subtrees, e.g.<code>//myproj/...</code>.\n"
        + "</li>\n"
        + "<li><code>[\"//visibility:legacy_public\"]</code>: Anyone can use this\n"
        + "rule (for now). <i>Developer action is needed</i>.\n"
        + "<p>This value has been used during the transition to the new\n"
        + "<code>[\"//visibility:private\"]</code> default, on June 6, 2011.\n"
        + "<i>We will eventually deprecate and then disallow this value.</i>\n"
        + "</li>\n"
        + "</ul>\n"
        + "<p>The visibility specifications of <code>//visibility:public</code>,\n"
        + "<code>//visibility:private</code> and\n"
        + "<code>//visibility:legacy_public</code>\n"
        + "can not be combined with any other visibility specifications.\n"
        + "A visibility specification may contain a combination of package labels\n"
        + "(i.e. //foo:__pkg__) and package_groups.</p>\n"
        + "<p>If a rule does not specify the visibility attribute,\n"
        + "the <code><a href=\"#package\">default_visibility</a></code>\n"
        + "attribute of the <code><a href=\"#package\">package</a></code>\n"
        + "statement in the BUILD file containing the rule is used\n"
        + "(except <a href=\"#exports_files\">exports_files</a> and\n"
        + "<a href=\"#cc_public_library\">cc_public_library</a>, which always default to\n"
        + "public).</p>\n"
        + "<p>If the default visibility for the package is not specified,\n"
        + "the rule is private: on June 6, 2011, in order to prevent teams\n"
        + "from reaching into private code, the default has been changed\n"
        + "to <code>[\"//visibility:private\"]</code>.</p>\n"
        + "<p><b>Example</b>:</p>\n"
        + "<p>\n"
        + "File <code>//frobber/bin/BUILD</code>:\n"
        + "</p>\n"
        + "<pre class=\"code\">\n"
        + "# This rule is visible to everyone\n"
        + "py_binary(\n"
        + "    name = \"executable\",\n"
        + "    visibility = [\"//visibility:public\"],\n"
        + "    deps = [\":library\"],\n"
        + ")\n"
        + "\n"
        + "# This rule is visible only to rules declared in the same package\n"
        + "py_library(\n"
        + "    name = \"library\",\n"
        + "    visibility = [\"//visibility:private\"],\n"
        + ")\n"
        + "\n"
        + "# This rule is visible to rules in package //object and //noun\n"
        + "py_library(\n"
        + "    name = \"subject\",\n"
        + "    visibility = [\n"
        + "        \"//noun:__pkg__\",\n"
        + "        \"//object:__pkg__\",\n"
        + "    ],\n"
        + ")\n"
        + "\n"
        + "# See package group //frobber:friends (below) for who can access this rule.\n"
        + "py_library(\n"
        + "    name = \"thingy\",\n"
        + "    visibility = [\"//frobber:friends\"],\n"
        + ")\n"
        + "</pre>\n"
        + "<p>\n"
        + "File <code>//frobber/BUILD</code>:\n"
        + "</p>\n"
        + "<pre class=\"code\">\n"
        + "# This is the package group declaration to which rule //frobber/bin:thingy refers.\n"
        + "#\n"
        + "# Our friends are packages //frobber, //fribber and any subpackage of //fribber.\n"
        + "package_group(\n"
        + "    name = \"friends\",\n"
        + "    packages = [\n"
        + "        \"//fribber/...\",\n"
        + "        \"//frobber\",\n"
        + "    ],\n"
        + ")\n"
        + "</pre>"))
      .build();

  public static final Map<String, RuleDocumentationAttribute> BINARY_ATTRIBUTES = ImmutableMap.of(
      "args", RuleDocumentationAttribute.create("args", DocgenConsts.BINARY_ATTRIBUTES,
          "Add these arguments to the target when executed by\n"
        + "<code>blaze run</code>.\n"
        + "<i>(List of strings; optional; subject to\n"
        + "<a href=\"#make_variables\">\"Make variable\"</a> substitution and\n"
        + "<a href=\"#sh-tokenization\">Bourne shell tokenization</a>)</i><br/>\n"
        + "These arguments are passed to the target before the target options\n"
        + "specified on the <code>blaze run</code> command line.\n"
        + "<p>Most binary rules permit an <code>args</code> attribute, but where\n"
        + "this attribute is not allowed, this fact is documented under the\n"
        + "specific rule.</p>"),
      "output_licenses", RuleDocumentationAttribute.create("output_licenses",
          DocgenConsts.BINARY_ATTRIBUTES,
          "The licenses of the output files that this binary generates.\n"
        + "<i>(List of strings; optional)</i><br/>\n"
        + "Describes the licenses of the output of the binary generated by\n"
        + "the rule. When a binary is referenced in a host attribute (for\n"
        + "example, the <code>tools</code> attribute of\n"
        + "a <code>genrule</code>), this license declaration is used rather\n"
        + "than the union of the licenses of its transitive closure. This\n"
        + "argument is useful when a binary is used as a tool during the\n"
        + "build of a rule, and it is not desirable for its license to leak\n"
        + "into the license of that rule. If this attribute is missing, the\n"
        + "license computation proceeds as if the host dependency was a\n"
        + "regular dependency.\n"
        + "<p>(For more about the distinction between host and target\n"
        + "configurations,\n"
        + "see <a href=\"blaze-user-manual.html#configurations\">"
        + "Build configurations</a> in the Blaze manual.)\n"
        + "<p><em class=\"harmful\">WARNING: in some cases (specifically, in\n"
        + "genrules) the build tool cannot guarantee that the binary\n"
        + "referenced by this attribute is actually used as a tool, and is\n"
        + "not, for example, copied to the output. In these cases, it is the\n"
        + "responsibility of the user to make sure that this is\n"
        + "true.</em></p>"));

  public static final Map<String, RuleDocumentationAttribute> TEST_ATTRIBUTES = ImmutableMap
      .<String, RuleDocumentationAttribute>builder()
      .put("args", RuleDocumentationAttribute.create("args", DocgenConsts.TEST_ATTRIBUTES,
          "Add these arguments to the <code>--test_arg</code>\n"
        + "when executed by <code>blaze test</code>.\n"
        + "<i>(List of strings; optional; subject to\n"
        + "<a href=\"#make_variables\">\"Make variable\"</a> substitution and\n"
        + "<a href=\"#sh-tokenization\">Bourne shell tokenization</a>)</i><br/>\n"
        + "These arguments are passed before the <code>--test_arg</code> values\n"
        + "specified on the <code>blaze test</code> command line."))
      .put("size", RuleDocumentationAttribute.create("size", DocgenConsts.TEST_ATTRIBUTES,
          "How \"heavy\" the test is\n"
        + "<i>(String \"enormous\", \"large\" \"medium\" or \"small\",\n"
        + "default is \"medium\")</i><br/>\n"
        + "A classification of the test's \"heaviness\": how much time/resources\n"
        + "it needs to run."
        + "Unittests are considered \"small\", integration tests \"medium\", "
        + "and end-to-end tests \"large\" or \"enormous\". "
        + "Blaze uses the size only to determine a default timeout."))
      .put("timeout", RuleDocumentationAttribute.create("timeout", DocgenConsts.TEST_ATTRIBUTES,
          "How long the test is\n"
        + "normally expected to run before returning.\n"
        + "<i>(String \"eternal\", \"long\", \"moderate\", or \"short\"\n"
        + "with the default derived from a test's size attribute)</i><br/>\n"
        + "While a test's size attribute controls resource estimation, a test's\n"
        + "timeout may be set independently.  If not explicitly specified, the\n"
        + "timeout is based on the test's size (with \"small\" &rArr; \"short\",\n"
        + "\"medium\" &rArr; \"moderate\", etc...). "
        + "\"short\" means 1 minute, \"moderate\" 5 minutes, and \"long\" 15 minutes."))
      .put("flaky", RuleDocumentationAttribute.create("flaky", DocgenConsts.TEST_ATTRIBUTES,
          "Marks test as flaky. <i>(Boolean; optional)</i><br/>\n"
        + "If set, executes the test up to 3 times before being declared as failed.\n"
        + "By default this attribute is set to 0 and test is considered to be stable.\n"
        + "Note, that use of this attribute is generally discouraged - we do prefer\n"
        + "all tests to be stable."))
      .put("shard_count", RuleDocumentationAttribute.create("shard_count",
          DocgenConsts.TEST_ATTRIBUTES,
          "Specifies the number of parallel shards\n"
        + "to use to run the test. <i>(Non-negative integer less than or equal to 50;\n"
        + "optional)</i><br/>\n"
        + "This value will override any heuristics used to determine the number of\n"
        + "parallel shards with which to run the test. Note that for some test\n"
        + "rules, this parameter may be required to enable sharding\n"
        + "in the first place. Also see --test_sharding_strategy."))
      .put("local", RuleDocumentationAttribute.create("local", DocgenConsts.TEST_ATTRIBUTES,
          "Forces the test to be run locally. <i>(Boolean; optional)</i><br/>\n"
        + "By default this attribute is set to 0 and the default testing strategy is\n"
        + "used. This is equivalent to providing \"local\" as a tag\n"
        + "(<code>tags=[\"local\"]</code>)."))
      .build();
}
