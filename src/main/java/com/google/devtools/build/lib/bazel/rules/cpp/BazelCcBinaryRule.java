// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.rules.cpp;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.Type.BOOLEAN;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.bazel.rules.cpp.BazelCppRuleClasses.CcBinaryBaseRule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;

/** Rule definition for cc_binary rules. */
public final class BazelCcBinaryRule implements RuleDefinition {

  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
    return builder
        .requiresConfigurationFragments(CppConfiguration.class)
        /*<!-- #BLAZE_RULE(cc_binary).IMPLICIT_OUTPUTS -->
        <ul>
        <li><code><var>name</var>.stripped</code> (only built if explicitly requested): A stripped
          version of the binary. <code>strip -g</code> is run on the binary to remove debug
          symbols.  Additional strip options can be provided on the command line using
          <code>--stripopt=-foo</code>. This output is only built if explicitly requested.</li>
        <li><code><var>name</var>.dwp</code> (only built if explicitly requested): If
          <a href="https://gcc.gnu.org/wiki/DebugFission">Fission</a> is enabled: a debug
          information package file suitable for debugging remotely deployed binaries. Else: an
          empty file.</li>
        </ul>
        <!-- #END_BLAZE_RULE.IMPLICIT_OUTPUTS -->*/
        .setImplicitOutputsFunction(BazelCppRuleClasses.CC_BINARY_IMPLICIT_OUTPUTS)
        /*<!-- #BLAZE_RULE(cc_binary).ATTRIBUTE(linkshared) -->
        Create a shared library.
        To enable this attribute, include <code>linkshared=True</code> in your rule. By default
        this option is off. If you enable it, you must name your binary
        <code>lib<i>foo</i>.so</code> (or whatever is the naming convention of libraries on the
        target platform) for some sensible value of <i>foo</i>.
        <p>
          The presence of this flag means that linking occurs with the <code>-shared</code> flag
          to <code>gcc</code>, and the resulting shared library is suitable for loading into for
          example a Java program. However, for build purposes it will never be linked into the
          dependent binary, as it is assumed that shared libraries built with a
          <a href="#cc_binary">cc_binary</a> rule are only loaded manually by other programs, so
          it should not be considered a substitute for the <a href="#cc_library">cc_library</a>
          rule. For sake of scalability we recommend avoiding this approach altogether and
          simply letting <code>java_library</code> depend on <code>cc_library</code> rules
          instead.
        </p>
        <p>
          If you specify both <code>linkopts=['-static']</code> and <code>linkshared=True</code>,
          you get a single completely self-contained unit. If you specify both
          <code>linkstatic=1</code> and <code>linkshared=True</code>, you get a single, mostly
          self-contained unit.
        </p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(
            attr("linkshared", BOOLEAN)
                .value(false)
                .nonconfigurable("used to *determine* the rule's configuration"))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("cc_binary")
        .ancestors(CcBinaryBaseRule.class, BaseRuleClasses.BinaryBaseRule.class)
        .factoryClass(BazelCcBinary.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = cc_binary, TYPE = BINARY, FAMILY = C / C++) -->

${IMPLICIT_OUTPUTS}

<!-- #END_BLAZE_RULE -->*/

/*<!-- #BLAZE_RULE (NAME = cc_library, TYPE = LIBRARY, FAMILY = C / C++) -->

<h4 id="hdrs">Header inclusion checking</h4>

<p>
  All header files that are used in the build must be declared in the <code>hdrs</code> or
  <code>srcs</code> of <code>cc_*</code> rules. This is enforced.
</p>

<p>
  For <code>cc_library</code> rules, headers in <code>hdrs</code> comprise the public interface of
  the library and can be directly included both from the files in <code>hdrs</code> and
  <code>srcs</code> of the library itself as well as from files in <code>hdrs</code> and
  <code>srcs</code> of <code>cc_*</code> rules that list the library in their <code>deps</code>.
  Headers in <code>srcs</code> must only be directly included from the files in <code>hdrs</code>
  and <code>srcs</code> of the library itself. When deciding whether to put a header into
  <code>hdrs</code> or <code>srcs</code>, you should ask whether you want consumers of this library
  to be able to directly include it. This is roughly the same decision as between
  <code>public</code> and <code>private</code> visibility in programming languages.
</p>

<p>
  <code>cc_binary</code> and <code>cc_test</code> rules do not have an exported interface, so they
  also do not have a <code>hdrs</code> attribute. All headers that belong to the binary or test
  directly should be listed in the <code>srcs</code>.
</p>

<p>
  To illustrate these rules, look at the following example.
</p>

<pre class="code">
cc_binary(
    name = "foo",
    srcs = [
        "foo.cc",
        "foo.h",
    ],
    deps = [":bar"],
)

cc_library(
    name = "bar",
    srcs = [
        "bar.cc",
        "bar-impl.h",
    ],
    hdrs = ["bar.h"],
    deps = [":baz"],
)

cc_library(
    name = "baz",
    srcs = [
        "baz.cc",
        "baz-impl.h",
    ],
    hdrs = ["baz.h"],
)
</pre>

<p>
  The allowed direct inclusions in this example are listed in the table below. For example
  <code>foo.cc</code> is allowed to directly include <code>foo.h</code> and <code>bar.h</code>, but
  not <code>baz.h</code>.
</p>

<table class="table table-striped table-bordered table-condensed">
  <thead>
    <tr><th>Including file</th><th>Allowed inclusions</th></tr>
  </thead>
  <tbody>
    <tr><td>foo.h</td><td>bar.h</td></tr>
    <tr><td>foo.cc</td><td>foo.h bar.h</td></tr>
    <tr><td>bar.h</td><td>bar-impl.h baz.h</td></tr>
    <tr><td>bar-impl.h</td><td>bar.h baz.h</td></tr>
    <tr><td>bar.cc</td><td>bar.h bar-impl.h baz.h</td></tr>
    <tr><td>baz.h</td><td>baz-impl.h</td></tr>
    <tr><td>baz-impl.h</td><td>baz.h</td></tr>
    <tr><td>baz.cc</td><td>baz.h baz-impl.h</td></tr>
  </tbody>
</table>

<p>
  The inclusion checking rules only apply to <em>direct</em>
  inclusions. In the example above <code>foo.cc</code> is allowed to
  include <code>bar.h</code>, which may include <code>baz.h</code>, which in
  turn is allowed to include <code>baz-impl.h</code>. Technically, the
  compilation of a <code>.cc</code> file may transitively include any header
  file in the <code>hdrs</code> or <code>srcs</code> in
  any <code>cc_library</code> in the transitive <code>deps</code> closure. In
  this case the compiler may read <code>baz.h</code> and <code>baz-impl.h</code>
  when compiling <code>foo.cc</code>, but <code>foo.cc</code> must not
  contain <code>#include "baz.h"</code>. For that to be
  allowed, <code>baz</code> must be added to the <code>deps</code>
  of <code>foo</code>.
</p>

<p>
  Unfortunately Bazel currently cannot distinguish between direct and transitive
  inclusions, so it cannot detect error cases where a file illegally includes a
  header directly that is only allowed to be included transitively. For example,
  Bazel would not complain if in the example above <code>foo.cc</code> directly
  includes <code>baz.h</code>. This would be illegal, because <code>foo</code>
  does not directly depend on <code>baz</code>. Currently, no error is produced
  in that case, but such error checking may be added in the future.
</p>

<!-- #END_BLAZE_RULE -->*/
