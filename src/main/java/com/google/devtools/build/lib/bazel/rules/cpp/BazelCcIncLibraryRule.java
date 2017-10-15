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
import static com.google.devtools.build.lib.packages.BuildType.LABEL;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.rules.cpp.CcIncLibraryRule;
import com.google.devtools.build.lib.rules.cpp.CcToolchain;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppRuleClasses;

/** Rule definition for the cc_inc_library class. */
public final class BazelCcIncLibraryRule implements RuleDefinition {
  @Override
  public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
    return builder
        .requiresConfigurationFragments(CppConfiguration.class)
        .add(
            attr(CcToolchain.CC_TOOLCHAIN_DEFAULT_ATTRIBUTE_NAME, LABEL)
                .value(CppRuleClasses.ccToolchainAttribute(env)))
        .add(attr(":stl", LABEL).value(BazelCppRuleClasses.STL))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("cc_inc_library")
        .ancestors(BaseRuleClasses.RuleBase.class, CcIncLibraryRule.class)
        .factoryClass(BazelCcIncLibrary.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = cc_inc_library, TYPE = LIBRARY, FAMILY = C / C++) -->

<p>
Bazel creates a subdirectory below
<code>includes</code> (relative to WORKSPACE) for each such rule, and makes sure that all
dependent rules have a corresponding <code>-isystem</code> directive to add this
directory into the compiler's header file search path for all compilations. Note
that if a rule has multiple <code>cc_inc_library</code> rules from the same
package in its dependencies, the first such rule will take precedence.
</p>

<p>
One use case for the <code>cc_inc_library</code> rule is to allow C++
<code>#include</code> directives to work with external
libraries that are placed in a version-specific subdirectory, without the
version number. For example, it allows including a header file in
<code>/library/v1/a.h</code> using the path
<code>/library/a.h</code>. This is useful to avoid changing a lot of
code when upgrading to a newer version. In this case, there should be one
<code>cc_inc_library</code> rule per public target. Note that only files that are declared in the
hdrs attribute are available with the rewritten path.
</p>

<p>
In this case, the <code>cc_inc_library</code> represents the interface of the package, and should be
the public top-level rule so that strict header inclusion checks can be performed.
</p>

<pre class="code">
# This rule makes the header file v1/library.h available for inclusion via the
# path /library/library.h.
cc_inc_library(
    name = "library",
    hdrs = ["v1/library.h"],
    prefix = "v1",
    deps = [":library_impl"],
)

cc_library(
    name = "library_impl",
    srcs = [
        "v1/library.c",
        "v1/library.h",
    ],
    visibility = ["//visibility:private"],
)
</pre>

<!-- #END_BLAZE_RULE -->*/
