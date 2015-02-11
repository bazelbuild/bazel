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

package com.google.devtools.build.lib.bazel.rules.java;

import static com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition.HOST;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.ImplicitOutputsFunction.fromFunctions;
import static com.google.devtools.build.lib.packages.Type.BOOLEAN;
import static com.google.devtools.build.lib.packages.Type.LABEL;
import static com.google.devtools.build.lib.packages.Type.LABEL_LIST;
import static com.google.devtools.build.lib.packages.Type.STRING;
import static com.google.devtools.build.lib.packages.Type.STRING_LIST;
import static com.google.devtools.build.lib.packages.Type.TRISTATE;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.BlazeRule;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.bazel.rules.cpp.BazelCppRuleClasses;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.PredicateWithMessage;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.packages.RuleClass.PackageNameConstraint;
import com.google.devtools.build.lib.packages.TriState;
import com.google.devtools.build.lib.rules.java.JavaSemantics;
import com.google.devtools.build.lib.util.FileTypeSet;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.Set;

/**
 * Rule class definitions for Java rules.
 */
public class BazelJavaRuleClasses {

  public static final PredicateWithMessage<Rule> JAVA_PACKAGE_NAMES = new PackageNameConstraint(
      PackageNameConstraint.ANY_SEGMENT, "java", "javatests");

  public static final ImplicitOutputsFunction JAVA_BINARY_IMPLICIT_OUTPUTS =
      fromFunctions(JavaSemantics.JAVA_BINARY_CLASS_JAR, JavaSemantics.JAVA_BINARY_SOURCE_JAR,
          JavaSemantics.JAVA_BINARY_DEPLOY_JAR, JavaSemantics.JAVA_BINARY_DEPLOY_SOURCE_JAR);

  static final ImplicitOutputsFunction JAVA_LIBRARY_IMPLICIT_OUTPUTS =
      fromFunctions(JavaSemantics.JAVA_LIBRARY_CLASS_JAR, JavaSemantics.JAVA_LIBRARY_SOURCE_JAR);

  /**
   * Common attributes for rules that depend on ijar.
   */
  @BlazeRule(name = "$ijar_base_rule",
               type = RuleClassType.ABSTRACT)
  public static final class IjarBaseRule implements RuleDefinition {
    @Override
    public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
      return builder
          .add(attr("$ijar", LABEL).cfg(HOST).exec().value(env.getLabel("//tools/defaults:ijar")))
          .setPreferredDependencyPredicate(JavaSemantics.JAVA_SOURCE)
          .build();
    }
  }


  /**
   * Common attributes for Java rules.
   */
  @BlazeRule(name = "$java_base_rule",
               type = RuleClassType.ABSTRACT,
               ancestors = { IjarBaseRule.class })
  public static final class JavaBaseRule implements RuleDefinition {
    @Override
    public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
      return builder
          .add(attr(":jvm", LABEL).cfg(HOST).value(JavaSemantics.JVM))
          .add(attr(":host_jdk", LABEL).cfg(HOST).value(JavaSemantics.HOST_JDK))
          .add(attr(":java_toolchain", LABEL).value(JavaSemantics.JAVA_TOOLCHAIN))
          .add(attr("$java_langtools", LABEL).cfg(HOST)
              .value(env.getLabel("//tools/defaults:java_langtools")))
          .add(attr("$javac_bootclasspath", LABEL).cfg(HOST)
              .value(env.getLabel(JavaSemantics.JAVAC_BOOTCLASSPATH_LABEL)))
          .add(attr("$javabuilder", LABEL).cfg(HOST)
              .value(env.getLabel(JavaSemantics.JAVABUILDER_LABEL)))
          .add(attr("$singlejar", LABEL).cfg(HOST)
              .value(env.getLabel(JavaSemantics.SINGLEJAR_LABEL)))
          .build();
    }
  }

  static final Set<String> ALLOWED_RULES_IN_DEPS = ImmutableSet.of(
      "cc_binary",  // NB: linkshared=1
      "cc_library",
      "genrule",
      "genproto",  // TODO(bazel-team): we should filter using providers instead (skylark rule).
      "java_import",
      "java_library",
      "sh_binary",
      "sh_library");

  /**
   * Common attributes for Java rules.
   */
  @BlazeRule(name = "$java_rule",
               type = RuleClassType.ABSTRACT,
               ancestors = { BaseRuleClasses.RuleBase.class, JavaBaseRule.class })
  public static final class JavaRule implements RuleDefinition {
    @Override
    public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
      return builder
          .override(builder.copy("deps")
              .allowedFileTypes(JavaSemantics.JAR)
              .allowedRuleClasses(ALLOWED_RULES_IN_DEPS)
              .skipAnalysisTimeFileTypeCheck())
          .add(attr("runtime_deps", LABEL_LIST)
              .allowedFileTypes(JavaSemantics.JAR)
              .allowedRuleClasses(ALLOWED_RULES_IN_DEPS)
              .skipAnalysisTimeFileTypeCheck())
          .add(attr("srcs", LABEL_LIST)
              .orderIndependent()
              .direct_compile_time_input()
              .allowedFileTypes(JavaSemantics.JAVA_SOURCE, JavaSemantics.JAR,
                  JavaSemantics.SOURCE_JAR, JavaSemantics.PROPERTIES))
          .add(attr("resources", LABEL_LIST).orderIndependent()
              .allowedFileTypes(FileTypeSet.ANY_FILE))
          .add(attr("plugins", LABEL_LIST).cfg(HOST).allowedRuleClasses("java_plugin")
              .legacyAllowAnyFileType())
          .add(attr(":java_plugins", LABEL_LIST)
              .cfg(HOST)
              .allowedRuleClasses("java_plugin")
              .silentRuleClassFilter()
              .value(JavaSemantics.JAVA_PLUGINS))
          .add(attr("javacopts", STRING_LIST))
          .build();
    }
  }

  /**
   * Base class for rule definitions producing Java binaries.
   */
  @BlazeRule(name = "$base_java_binary",
               type = RuleClassType.ABSTRACT,
               ancestors = { JavaRule.class,
                             // java_binary and java_test require the crosstool C++ runtime
                             // libraries (libstdc++.so, libgcc_s.so).
                             // TODO(bazel-team): Add tests for Java+dynamic runtime.
                             BazelCppRuleClasses.CcLinkingRule.class })
  public static final class BaseJavaBinaryRule implements RuleDefinition {
    @Override
    public RuleClass build(Builder builder, final RuleDefinitionEnvironment env) {
      return builder
          .add(attr("classpath_resources", LABEL_LIST).legacyAllowAnyFileType())
          .add(attr("jvm_flags", STRING_LIST))
          .add(attr("main_class", STRING))
          .add(attr("create_executable", BOOLEAN).nonconfigurable("internal").value(true))
          .add(attr("deploy_manifest_lines", STRING_LIST))
          .add(attr("stamp", TRISTATE).value(TriState.AUTO))
          .add(attr(":java_launcher", LABEL).value(JavaSemantics.JAVA_LAUNCHER))  // blaze flag
          .build();
    }
  }

  /**
   * Returns the relative path to the WORKSPACE file describing the external dependencies necessary
   * for the Java rules.
   */
  public static PathFragment getDefaultWorkspace() {
    return new PathFragment("jdk.WORKSPACE");
  }
}
