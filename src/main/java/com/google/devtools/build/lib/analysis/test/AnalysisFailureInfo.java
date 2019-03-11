// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.test;

import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.skylarkbuildapi.test.AnalysisFailureInfoApi;

/**
 * Implementation of {@link AnalysisFailureInfoApi}.
 *
 * Encapsulates information about analysis-phase errors which would have occurred during a
 * build.
 */
public class AnalysisFailureInfo extends Info implements AnalysisFailureInfoApi<AnalysisFailure> {

  /**
   * Singleton provider instance for {@link AnalysisFailureInfo}.
   */
  public static final AnalysisFailureInfoProvider SKYLARK_CONSTRUCTOR =
      new AnalysisFailureInfoProvider();

  private final NestedSet<AnalysisFailure> causes;

  private AnalysisFailureInfo(NestedSet<AnalysisFailure> causes) {
    super(SKYLARK_CONSTRUCTOR, Location.BUILTIN);
    this.causes = causes;
  }

  /**
   * Constructs and returns an {@link AnalysisFailureInfo} object representing the given failures.
   */
  public static AnalysisFailureInfo forAnalysisFailures(Iterable<AnalysisFailure> failures) {
    return new AnalysisFailureInfo(
        NestedSetBuilder.<AnalysisFailure>stableOrder().addAll(failures).build());
  }

  /**
   * Constructs and returns an {@link AnalysisFailureInfo} object representing the given sets of
   * failures. When combining nested sets of analysis failures, use this method instead of {@link
   * #forAnalysisFailures} so that the sets do not need to be flattened.
   */
  public static AnalysisFailureInfo forAnalysisFailureSets(
      Iterable<NestedSet<AnalysisFailure>> failures) {
    NestedSetBuilder<AnalysisFailure> fullSetBuilder =
        NestedSetBuilder.<AnalysisFailure>stableOrder();
    for (NestedSet<AnalysisFailure> failure : failures) {
      fullSetBuilder.addTransitive(failure);
    }
    return new AnalysisFailureInfo(fullSetBuilder.build());
  }

  @Override
  public NestedSet<AnalysisFailure> getCauses() {
    return causes;
  }

  /**
   * Provider implementation for {@link AnalysisFailureInfo}.
   */
  public static class AnalysisFailureInfoProvider
      extends BuiltinProvider<AnalysisFailureInfo> implements AnalysisFailureInfoProviderApi {

    public AnalysisFailureInfoProvider() {
      super("AnalysisFailureInfo", AnalysisFailureInfo.class);
    }
  }
}
