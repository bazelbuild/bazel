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

package com.google.devtools.build.lib.rules.cpp;

import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.rules.cpp.CppCompilationContext.Builder;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration.HeadersCheckingMode;
import com.google.devtools.build.lib.vfs.PathFragment;

/**
 * Null-object like {@link CppSemantics} implementation. Only to be used in tests that don't depend
 * on semantics details.
 */
@Immutable
public final class MockCppSemantics implements CppSemantics {

  public static final CppSemantics INSTANCE = new MockCppSemantics();

  private MockCppSemantics() {}

  @Override
  public PathFragment getEffectiveSourcePath(Artifact source) {
    PathFragment sourceRelativeName = source.getRootRelativePath();
    return sourceRelativeName;
  }

  @Override
  public void finalizeCompileActionBuilder(
      RuleContext ruleContext,
      CppCompileActionBuilder actionBuilder,
      FeatureSpecification featureSpecification,
      Predicate<String> coptsFilter,
      ImmutableSet<String> features) {}

  @Override
  public void setupCompilationContext(RuleContext ruleContext, Builder contextBuilder) {}

  @Override
  public NestedSet<Artifact> getAdditionalPrunableIncludes() {
    return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
  }

  @Override
  public IncludeProcessing getIncludeProcessing() {
    return null;
  }

  @Override
  public HeadersCheckingMode determineHeadersCheckingMode(RuleContext ruleContext) {
    return HeadersCheckingMode.LOOSE;
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
  public void validateAttributes(RuleContext ruleContext) {}

  @Override
  public boolean needsIncludeValidation() {
    return true;
  }
}
