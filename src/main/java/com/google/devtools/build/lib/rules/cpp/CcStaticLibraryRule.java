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
package com.google.devtools.build.lib.rules.cpp;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static com.google.devtools.build.lib.packages.Type.STRING_LIST;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.StarlarkProviderIdentifier;
import com.google.devtools.build.lib.util.FileTypeSet;

/** A dummy rule for <code>cc_static_library</code> rule. */
public final class CcStaticLibraryRule implements RuleDefinition {

  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
    return builder
        /*<!-- #BLAZE_RULE(cc_static_library).ATTRIBUTE(deps) -->
        The list of targets to combine into a static library, including all their transitive
        dependencies.

        <p>Dependencies that do not provide any object files are not included in the static
        library, but their labels are collected in the file provided by the
        <code>linkdeps</code> output group.</p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(
            attr("deps", LABEL_LIST)
                .skipAnalysisTimeFileTypeCheck()
                .allowedFileTypes(FileTypeSet.NO_FILE)
                .mandatoryProviders(StarlarkProviderIdentifier.forKey(CcInfo.PROVIDER.getKey())))
        .add(
            attr("tags", STRING_LIST)
                .orderIndependent()
                .taggable()
                .nonconfigurable("low-level attribute, used in TargetUtils without configurations"))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return Metadata.builder()
        .name("cc_static_library")
        .factoryClass(BaseRuleClasses.EmptyRuleConfiguredTargetFactory.class)
        .build();
  }
}
/*<!-- #BLAZE_RULE (NAME = cc_static_library, TYPE = LIBRARY, FAMILY = C / C++) -->
Produces a static library from a list of targets and their transitive dependencies.

<p>The resulting static library contains the object files of the targets listed in
<code>deps</code> as well as their transitive dependencies, with preference given to
<code>PIC</code> objects.</p>

<h4 id="cc_static_library_output_groups">Output groups</h4>

<h5><code>linkdeps</code></h5>
<p>A text file containing the labels of those transitive dependencies of targets listed in
<code>deps</code> that did not contribute any object files to the static library, but do
provide at least one static, dynamic or interface library. The resulting static library
may require these libraries to be available at link time.</p>

<h5><code>linkopts</code></h5>
<p>A text file containing the user-provided <code>linkopts</code> of all transitive
dependencies of targets listed in <code>deps</code>.

<h4 id="cc_static_library_symbol_check">Duplicate symbols</h4>
<p>By default, the <code>cc_static_library</code> rule checks that the resulting static
library does not contain any duplicate symbols. If it does, the build fails with an error
message that lists the duplicate symbols and the object files containing them.</p>

<p>This check can be disabled per target or per package by setting
<code>features = ["-symbol_check"]</code> or globally via
<code>--features=-symbol_check</code>.</p>

<h5 id="cc_static_library_symbol_check_toolchain">Toolchain support for <code>symbol_check</code></h5>
<p>The auto-configured C++ toolchains shipped with Bazel support the
<code>symbol_check</code> feature on all platforms. Custom toolchains can add support for
it in one of two ways:</p>
<ul>
  <li>Implementing the <code>ACTION_NAMES.validate_static_library</code> action and
  enabling it with the <code>symbol_check</code> feature. The tool set in the action is
  invoked with two arguments, the static library to check for duplicate symbols and the
  path of a file that must be created if the check passes.</li>
  <li>Having the <code>symbol_check</code> feature add archiver flags that cause the
  action creating the static library to fail on duplicate symbols.</li>
</ul>

<!-- #END_BLAZE_RULE -->*/
