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

package com.google.devtools.build.lib.rules.objc;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.ToolchainTransitionMode;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppRuleClasses;
import com.google.devtools.build.lib.util.FileType;

/**
 * Rule definition for {@code objc_import}.
 */
public class ObjcImportRule implements RuleDefinition {
  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment environment) {
    return builder
        .requiresConfigurationFragments(
            ObjcConfiguration.class,
            AppleConfiguration.class,
            AppleConfiguration.class,
            CppConfiguration.class)
        /* <!-- #BLAZE_RULE(objc_import).ATTRIBUTE(archives) -->
        The list of <code>.a</code> files provided to Objective-C targets that
        depend on this target.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr("archives", LABEL_LIST).mandatory().nonEmpty().allowedFileTypes(FileType.of(".a")))
        .addRequiredToolchains(CppRuleClasses.ccToolchainTypeAttribute(environment))
        .useToolchainTransition(ToolchainTransitionMode.ENABLED)
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("objc_import")
        .factoryClass(ObjcImport.class)
        .ancestors(
            BaseRuleClasses.NativeBuildRule.class,
            ObjcRuleClasses.CompilingRule.class,
            ObjcRuleClasses.AlwaysLinkRule.class,
            BaseRuleClasses.MakeVariableExpandingRule.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = objc_import, TYPE = LIBRARY, FAMILY = Objective-C) -->

<p>This rule encapsulates an already-compiled static library in the form of an
<code>.a</code> file. It also allows exporting headers and resources using the same
attributes supported by <code>objc_library</code>.</p>

<!-- #END_BLAZE_RULE -->*/
