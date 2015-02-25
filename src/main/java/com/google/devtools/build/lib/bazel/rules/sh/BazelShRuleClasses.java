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

package com.google.devtools.build.lib.bazel.rules.sh;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.ImplicitOutputsFunction.fromTemplates;
import static com.google.devtools.build.lib.packages.Type.LABEL_LIST;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.BlazeRule;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.Attribute.AllowedValueSet;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.PredicateWithMessage;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;

import java.util.Collection;
import java.util.Map;

import javax.annotation.Nullable;

/**
 * Rule definitions for rule classes implementing shell support.
 */
public final class BazelShRuleClasses {

  static final Collection<String> ALLOWED_RULES_IN_DEPS_WITH_WARNING = ImmutableSet.of(
      "filegroup", "Fileset", "genrule", "sh_binary", "sh_test", "test_suite");

  /**
   * Common attributes for shell rules.
   */
  @BlazeRule(name = "$sh_target",
               type = RuleClassType.ABSTRACT,
               ancestors = { BaseRuleClasses.RuleBase.class })
  public static final class ShRule implements RuleDefinition {
    @Override
    public RuleClass build(Builder builder, RuleDefinitionEnvironment environment) {
      return builder
          .add(attr("srcs", LABEL_LIST).mandatory().legacyAllowAnyFileType())
          .override(builder.copy("deps")
              .allowedRuleClasses("sh_library", "proto_library")
              .allowedRuleClassesWithWarning(ALLOWED_RULES_IN_DEPS_WITH_WARNING)
              .allowedFileTypes())
          .build();
    }
  }

  /**
   * Defines the file name of an sh_binary's implicit .sar (script package) output.
   */
  static final ImplicitOutputsFunction SAR_PACKAGE_FILENAME =
      fromTemplates("%{name}.sar");

  /**
   * Convenience structure for the bash dependency combinations defined
   * by BASH_BINARY_BINDINGS.
   */
  static class BashBinaryBinding {
    public final String execPath;
    public BashBinaryBinding(@Nullable String execPath) {
      this.execPath = execPath;
    }
  }

  /**
   * Attribute value specifying the local system's bash version.
   */
  static final String SYSTEM_BASH_VERSION = "system";

  static final Map<String, BashBinaryBinding> BASH_BINARY_BINDINGS =
      ImmutableMap.of(
          // "system": don't package any bash with the target, but rather use whatever is
          // available on the system the script is run on.
          SYSTEM_BASH_VERSION, new BashBinaryBinding("/bin/bash")
      );

  static final String DEFAULT_BASH_VERSION = SYSTEM_BASH_VERSION;

  // TODO(bazel-team): refactor sh_binary and sh_base to have a common root
  // with srcs and bash_version attributes
  static final PredicateWithMessage<Object> BASH_VERSION_ALLOWED_VALUES =
      new AllowedValueSet(BASH_BINARY_BINDINGS.keySet());
}
