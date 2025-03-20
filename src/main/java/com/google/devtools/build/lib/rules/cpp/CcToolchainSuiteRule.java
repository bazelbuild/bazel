// Copyright 2015 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.actions.ActionConflictException;
import com.google.devtools.build.lib.analysis.BaseRuleClasses.NativeBuildRule;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.util.FileTypeSet;

/**
 * Definition of the {@code cc_toolchain_suite} rule.
 */
public final class CcToolchainSuiteRule implements RuleDefinition {

  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment environment) {
    return builder
        .add(
            attr("toolchains", BuildType.LABEL_DICT_UNARY)
                .mandatory()
                .allowedFileTypes(FileTypeSet.NO_FILE)
                .nonconfigurable("Used during configuration creation"))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return Metadata.builder()
        .name("cc_toolchain_suite")
        .ancestors(NativeBuildRule.class)
        .factoryClass(CcToolchainSuite.class)
        .build();
  }

  /** A noop rule factory, just used for compatibility. */
  public static final class CcToolchainSuite implements RuleConfiguredTargetFactory {
    @Override
    public ConfiguredTarget create(RuleContext ruleContext)
        throws InterruptedException, ActionConflictException {
      ruleContext.ruleWarning("Rule is a no-op.");
      return new RuleConfiguredTargetBuilder(ruleContext)
          .addProvider(RunfilesProvider.simple(Runfiles.EMPTY))
          .build();
    }
  }
}

/*<!-- #BLAZE_RULE (NAME = cc_toolchain_suite, TYPE = OTHER, FAMILY = C / C++) -->

<p>Deprecated: the rule is a no-op and will be removed.
<!-- #END_BLAZE_RULE -->*/
