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
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static com.google.devtools.build.lib.packages.Type.BOOLEAN;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.bazel.rules.cpp.BazelCppRuleClasses.CcLibraryBaseRule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.rules.cpp.CcInfo;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;
import com.google.devtools.build.lib.util.FileTypeSet;

/** Rule definition for the cc_library rule. */
public final class BazelCcLibraryRule implements RuleDefinition {
  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
    return builder
        // TODO: Google cc_library overrides documentation for:
        // deps, data, linkopts, defines, srcs; override here too?

        .requiresConfigurationFragments(CppConfiguration.class)
        /*<!-- #BLAZE_RULE(cc_library).ATTRIBUTE(alwayslink) -->
        If 1, any binary that depends (directly or indirectly) on this C++
        library will link in all the object files for the files listed in
        <code>srcs</code>, even if some contain no symbols referenced by the binary.
        This is useful if your code isn't explicitly called by code in
        the binary, e.g., if your code registers to receive some callback
        provided by some service.

        <p>If alwayslink doesn't work with VS 2017 on Windows, that is due to a
        <a href="https://github.com/bazelbuild/bazel/issues/3949">known issue</a>,
        please upgrade your VS 2017 to the latest version.</p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr("alwayslink", BOOLEAN))
        .override(attr("linkstatic", BOOLEAN).value(false))
        /*<!-- #BLAZE_RULE(cc_library).ATTRIBUTE(implementation_deps) -->
        The list of other libraries that the library target depends on. Unlike with
        <code>deps</code>, the headers and include paths of these libraries (and all their
        transitive deps) are only used for compilation of this library, and not libraries that
        depend on it. Libraries specified with <code>implementation_deps</code> are still linked in
        binary targets that depend on this library.
        <p>For now usage is limited to cc_libraries and guarded by the flag
        <code>--experimental_cc_implementation_deps</code>.</p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(
            attr("implementation_deps", LABEL_LIST)
                .allowedFileTypes(FileTypeSet.NO_FILE)
                .mandatoryProviders(CcInfo.PROVIDER.id()))
        /*<!-- #BLAZE_RULE(cc_library).ATTRIBUTE(additional_compiler_inputs) -->
        Any additional files you might want to pass to the compiler command line, such as sanitizer
        ignorelists, for example. Files specified here can then be used in copts with the
        $(location) function.
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr("additional_compiler_inputs", LABEL_LIST).allowedFileTypes(FileTypeSet.ANY_FILE))
        .advertiseStarlarkProvider(CcInfo.PROVIDER.id())
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("cc_library")
        .ancestors(CcLibraryBaseRule.class, BaseRuleClasses.MakeVariableExpandingRule.class)
        .factoryClass(BaseRuleClasses.EmptyRuleConfiguredTargetFactory.class)
        .build();
  }
}
