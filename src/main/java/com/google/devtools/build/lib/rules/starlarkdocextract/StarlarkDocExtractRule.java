// Copyright 2023 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.starlarkdocextract;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.ImplicitOutputsFunction.fromFunctions;
import static com.google.devtools.build.lib.packages.Type.STRING_LIST;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.util.FileType;
import javax.annotation.Nullable;

/** Definition of the {@code starlark_doc_extract} rule. */
public final class StarlarkDocExtractRule implements RuleDefinition {

  /**
   * Adds {@code starlark_doc_extract} and its dependencies to the provided configured rule class
   * builder.
   */
  public static void register(ConfiguredRuleClassProvider.Builder builder) {
    builder.addConfigurationFragment(StarlarkDocExtract.Configuration.class);
    builder.addRuleDefinition(new StarlarkDocExtractRule());
  }

  @Override
  @Nullable
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
    // TODO(b/276733504): publicly document the rule and its attributes once non-experimental.
    return builder
        .add(
            attr(StarlarkDocExtract.SRC_ATTR, LABEL)
                .allowedFileTypes(FileType.of(".bzl"))
                .mandatory())
        .add(
            attr(StarlarkDocExtract.SYMBOL_NAMES_ATTR, STRING_LIST)
                .value(ImmutableList.<String>of()))
        .setImplicitOutputsFunction(
            fromFunctions(StarlarkDocExtract.BINARYPROTO_OUT, StarlarkDocExtract.TEXTPROTO_OUT))
        .requiresConfigurationFragments(StarlarkDocExtract.Configuration.class)
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        // TODO(b/276733514): add `bazel dump --starlark_doc` command.
        .name("starlark_doc_extract")
        .ancestors(BaseRuleClasses.NativeActionCreatingRule.class)
        .factoryClass(StarlarkDocExtract.class)
        .build();
  }
}
