// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.repository;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;

import com.google.devtools.build.lib.analysis.BaseRuleClasses.BaseRule;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.util.FileTypeSet;

/**
 * Binds an existing target to a target in the virtual //external package.
 */
public final class BindRule implements RuleDefinition {

  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment environment) {
    return builder
        /* <!-- #BLAZE_RULE(bind).ATTRIBUTE(actual) -->
        The target to be aliased.

        <p>This target must exist, but can be any type of rule (including bind).</p>

        <p>If this attribute is omitted, rules referring to this target in <code>//external</code>
        will simply not see this dependency edge. Note that this is different from omitting the
        <code>bind</code> rule completely: it is an error if an <code>//external</code> dependency
        does not have an associated <code>bind</code> rule.
        </p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("actual", LABEL).allowedFileTypes(FileTypeSet.ANY_FILE))
        .setWorkspaceOnly()
        .canHaveAnyProvider()
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("bind")
        .type(RuleClassType.WORKSPACE)
        .ancestors(BaseRule.class)
        .factoryClass(Bind.class)
        .build();
  }
}
/*<!-- #BLAZE_RULE (NAME = bind, FAMILY = Workspace)[GENERIC_RULE] -->

<em><p>Warning: use of <code>bind()</code> is not recommended. See "<a
href="https://github.com/bazelbuild/bazel/issues/1952">Consider removing bind</a>" for a long
discussion of its issues and alternatives.</p></em>
<em><p>Warning: <code>select()</code> cannot be used in <code>bind()</code>. See the <a
href="../configurable-attributes.html#bind-select">Configurable Attributes FAQ</a> for
details.</p></em>

<p>Gives a target an alias in the <code>//external</code> package.</p>

<p>The <code>//external</code> package is not a "normal" package: there is no external/ directory,
  so it can be thought of as a "virtual package" that contains all bound targets.</p>

<h4 id="bind_examples">Examples</h4>

<p>To give a target an alias, <code>bind</code> it in the <i>WORKSPACE</i> file.  For example,
  suppose there is a <code>java_library</code> target called
  <code>//third_party/javacc-v2</code>.  This can be aliased by adding the following to the
  <i>WORKSPACE</i> file:</p>

<pre class="code">
bind(
    name = "javacc-latest",
    actual = "//third_party/javacc-v2",
)
</pre>

<p>Now targets can depend on <code>//external:javacc-latest</code> instead of
  <code>//third_party/javacc-v2</code>. If javacc-v3 is released, the <code>bind</code> rule can be
  updated and all of the BUILD files depending on <code>//external:javacc-latest</code> will now
  depend on javacc-v3 without needing to be edited.</p>

<p>Bind can also be used to make targets in external repositories available to your workspace.
  For example, if there is a remote repository named <code>@my-ssl</code> imported in the
  <i>WORKSPACE</i> file and it has a cc_library target <code>//src:openssl-lib</code>, you can
  create an alias for this target using <code>bind</code>:</p>

<pre class="code">
bind(
    name = "openssl",
    actual = "@my-ssl//src:openssl-lib",
)
</pre>

<p>Then, in a BUILD file in your workspace, the bound target can be used as follows:</p>

<pre class="code">
cc_library(
    name = "sign-in",
    srcs = ["sign_in.cc"],
    hdrs = ["sign_in.h"],
    deps = ["//external:openssl"],
)
</pre>

<p>Within <code>sign_in.cc</code> and <code>sign_in.h</code>, the header files exposed by
  <code>//external:openssl</code> can be referred to using their path relative to their repository
  root.  For example, if the rule definition for <code>@my-ssl//src:openssl-lib</code> looks like
  this:</p>

<pre class="code">
cc_library(
    name = "openssl-lib",
    srcs = ["openssl.cc"],
    hdrs = ["openssl.h"],
)
</pre>

<p>Then <code>sign_in.cc</code>'s includes might look like this:</p>

<pre class="code">
#include "sign_in.h"
#include "src/openssl.h"
</pre>

<!-- #END_BLAZE_RULE -->*/
