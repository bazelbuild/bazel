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

package com.google.devtools.build.lib.rules.objc;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;

/**
 * Rule definition for experimental_objc_library.
 */
public class ExperimentalObjcLibraryRule implements RuleDefinition {
  @Override
  public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
    return builder
        .requiresConfigurationFragments(
            ObjcConfiguration.class, AppleConfiguration.class, CppConfiguration.class)
        .setImplicitOutputsFunction(
            ImplicitOutputsFunction.fromFunctions(
                CompilationSupport.FULLY_LINKED_LIB, XcodeSupport.PBXPROJ))
        .cfg(AppleCrosstoolTransition.APPLE_CROSSTOOL_TRANSITION)
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("experimental_objc_library")
        .factoryClass(ExperimentalObjcLibrary.class)
        .ancestors(
            BaseRuleClasses.BaseRule.class,
            ObjcRuleClasses.LinkingRule.class,
            ObjcRuleClasses.AlwaysLinkRule.class,
            ObjcRuleClasses.XcodegenRule.class,
            ObjcRuleClasses.CrosstoolRule.class)
        .build();
  }
}
