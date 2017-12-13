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

package com.google.devtools.build.lib.bazel.rules.cpp;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.rules.cpp.CppCompilationContext.Builder;
import com.google.devtools.build.lib.rules.cpp.CppCompileActionBuilder;
import com.google.devtools.build.lib.rules.cpp.CppCompileActionContext;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration.HeadersCheckingMode;
import com.google.devtools.build.lib.rules.cpp.CppSemantics;
import com.google.devtools.build.lib.rules.cpp.IncludeProcessing;
import com.google.devtools.build.lib.rules.cpp.NoProcessing;

/**
 * C++ compilation semantics.
 */
public class BazelCppSemantics implements CppSemantics {
  public static final CppSemantics INSTANCE = new BazelCppSemantics();

  private final IncludeProcessing includeProcessing;

  private BazelCppSemantics() {
    this.includeProcessing = new NoProcessing();
  }

  @Override
  public void finalizeCompileActionBuilder(
      RuleContext ruleContext, CppCompileActionBuilder actionBuilder) {
    actionBuilder
        .setCppConfiguration(ruleContext.getFragment(CppConfiguration.class))
        .setActionContext(CppCompileActionContext.class)
        // Because Bazel does not support include scanning, we need the entire crosstool filegroup,
        // including header files, as opposed to just the "compile" filegroup.
        .addTransitiveMandatoryInputs(actionBuilder.getToolchain().getCrosstool())
        .setShouldScanIncludes(false);
  }

  @Override
  public void setupCompilationContext(RuleContext ruleContext, Builder contextBuilder) {
  }

  @Override
  public NestedSet<Artifact> getAdditionalPrunableIncludes() {
    return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
  }

  @Override
  public HeadersCheckingMode determineHeadersCheckingMode(RuleContext ruleContext) {
    return HeadersCheckingMode.STRICT;
  }

  @Override
  public IncludeProcessing getIncludeProcessing() {
    return includeProcessing;
  }

  @Override
  public boolean needsIncludeScanning(RuleContext ruleContext) {
    return false;
  }

  @Override
  public boolean needsDotdInputPruning() {
    return true;
  }

  @Override
  public void validateAttributes(RuleContext ruleContext) {
  }

  @Override
  public boolean needsIncludeValidation() {
    return true;
  }
}
