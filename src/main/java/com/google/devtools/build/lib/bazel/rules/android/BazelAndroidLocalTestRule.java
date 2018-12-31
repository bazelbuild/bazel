// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.bazel.rules.android;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static com.google.devtools.build.lib.packages.BuildType.TRISTATE;
import static com.google.devtools.build.lib.packages.ImplicitOutputsFunction.fromFunctions;

import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.bazel.rules.java.BazelJavaRuleClasses.BaseJavaBinaryRule;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.packages.SkylarkProviderIdentifier;
import com.google.devtools.build.lib.packages.TriState;
import com.google.devtools.build.lib.rules.android.AndroidFeatureFlagSetProvider;
import com.google.devtools.build.lib.rules.android.AndroidLocalTestBaseRule;
import com.google.devtools.build.lib.rules.config.ConfigFeatureFlagTransitionFactory;
import com.google.devtools.build.lib.rules.java.JavaConfiguration;
import com.google.devtools.build.lib.rules.java.JavaInfo;
import com.google.devtools.build.lib.rules.java.JavaSemantics;
import com.google.devtools.build.lib.rules.java.Jvm;

/** Rule definition for Bazel android_local_test */
public class BazelAndroidLocalTestRule implements RuleDefinition {

  protected static final String JUNIT_TESTRUNNER = "//tools/jdk:TestRunner_deploy.jar";

  private static final ImmutableCollection<String> ALLOWED_RULES_IN_DEPS =
      ImmutableSet.of(
          "aar_import",
          "android_library",
          "java_import",
          "java_library",
          "java_lite_proto_library");

  static final ImplicitOutputsFunction ANDROID_ROBOLECTRIC_IMPLICIT_OUTPUTS =
      fromFunctions(JavaSemantics.JAVA_BINARY_CLASS_JAR, JavaSemantics.JAVA_BINARY_SOURCE_JAR);

  @Override
  public RuleClass build(Builder builder, RuleDefinitionEnvironment environment) {
    return builder
        .requiresConfigurationFragments(JavaConfiguration.class, Jvm.class)
        .setImplicitOutputsFunction(ANDROID_ROBOLECTRIC_IMPLICIT_OUTPUTS)
        .override(
            attr("deps", LABEL_LIST)
                .allowedFileTypes()
                .allowedRuleClasses(ALLOWED_RULES_IN_DEPS)
                .mandatoryProvidersList(
                    ImmutableList.of(
                        ImmutableList.of(
                            SkylarkProviderIdentifier.forKey(JavaInfo.PROVIDER.getKey())))))
        .override(attr("$testsupport", LABEL).value(environment.getToolsLabel(JUNIT_TESTRUNNER)))
        .override(attr("stamp", TRISTATE).value(TriState.NO))
        .removeAttribute("$experimental_testsupport")
        .removeAttribute("classpath_resources")
        .removeAttribute("create_executable")
        .removeAttribute("deploy_manifest_lines")
        .removeAttribute("distribs")
        .removeAttribute("launcher")
        .removeAttribute("main_class")
        .removeAttribute("resources")
        .removeAttribute("use_testrunner")
        .removeAttribute(":java_launcher")
        .cfg(
            new ConfigFeatureFlagTransitionFactory(AndroidFeatureFlagSetProvider.FEATURE_FLAG_ATTR))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("android_local_test")
        .type(RuleClassType.TEST)
        .ancestors(
            AndroidLocalTestBaseRule.class,
            BaseJavaBinaryRule.class,
            BaseRuleClasses.TestBaseRule.class)
        .factoryClass(BazelAndroidLocalTest.class)
        .build();
  }
}
