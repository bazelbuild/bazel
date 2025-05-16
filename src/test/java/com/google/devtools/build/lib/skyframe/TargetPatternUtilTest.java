// Copyright 2025 The Bazel Authors. All rights reserved.
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
// limitations under the License.package com.google.devtools.build.lib.skyframe;
package com.google.devtools.build.lib.skyframe;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.skyframe.EvaluationResultSubjectFactory.assertThatEvaluationResult;
import static java.util.Objects.requireNonNull;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.cmdline.SignedTargetPattern;
import com.google.devtools.build.lib.cmdline.TargetPattern;
import com.google.devtools.build.lib.pkgcache.FilteringPolicies;
import com.google.devtools.build.lib.pkgcache.FilteringPolicy;
import com.google.devtools.build.lib.skyframe.TargetPatternUtil.InvalidTargetPatternException;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.util.SkyframeExecutorTestUtils;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import com.google.testing.junit.testparameterinjector.TestParameters;
import com.google.testing.junit.testparameterinjector.TestParameters.TestParametersValues;
import com.google.testing.junit.testparameterinjector.TestParametersValuesProvider;
import java.io.IOException;
import java.util.Arrays;
import javax.annotation.Nullable;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests for {@link TargetPatternUtil}. */
@RunWith(TestParameterInjector.class)
public class TargetPatternUtilTest extends BuildViewTestCase {

  @Test
  @TestParameters(valuesProvider = ExpansionPatternProvider.class)
  public void expansion(ImmutableList<String> rawPatterns, ImmutableList<Label> expectedLabels)
      throws Exception {
    ExpansionPatternProvider.createBuildFiles(scratch);

    ImmutableList<Label> result = expandTargetPattern(rawPatterns, FilteringPolicies.NO_FILTER);
    assertThat(result).containsExactlyElementsIn(expectedLabels);
  }

  // TODO: blaze-configurability-team - Test errors
  // TODO: blaze-configurability-team - Test relative labels
  // TODO: blaze-configurability-team - Test filtering policies

  private static final class ExpansionPatternProvider extends TestParametersValuesProvider {
    private static TestParametersValues create(String rawPattern, String... rawLabels) {
      return create(ImmutableList.of(rawPattern), rawLabels);
    }

    private static TestParametersValues create(
        ImmutableList<String> rawPatterns, String... rawLabels) {
      ImmutableList<Label> labels =
          Arrays.stream(rawLabels).map(Label::parseCanonicalUnchecked).collect(toImmutableList());

      String name = String.format("%s-%s", rawPatterns, labels);
      return TestParametersValues.builder()
          .name(name)
          .addParameter("rawPatterns", rawPatterns)
          .addParameter("expectedLabels", labels)
          .build();
    }

    @Override
    protected ImmutableList<TestParametersValues> provideValues(Context context) {
      return ImmutableList.of(
          // Single patterns.
          create("//foo/bar:baz", "//foo/bar:baz"),
          create(
              "//wildcard/single/...",
              "//wildcard/single:a",
              "//wildcard/single:b",
              "//wildcard/single:c"),
          create(
              "//wildcard/single:all",
              "//wildcard/single:a",
              "//wildcard/single:b",
              "//wildcard/single:c"),
          create(
              "//wildcard/single:*",
              "//wildcard/single:BUILD",
              "//wildcard/single:a",
              "//wildcard/single:b",
              "//wildcard/single:c"),
          create(
              "//wildcard/deep/...",
              "//wildcard/deep/a",
              "//wildcard/deep/b:b_1",
              "//wildcard/deep/b:b_2",
              "//wildcard/deep/c"),

          // Combinations of patterns
          create(
              ImmutableList.of("//foo/bar:baz", "//foo/bar:quux"),
              "//foo/bar:baz",
              "//foo/bar:quux"),
          create(
              ImmutableList.of("//wildcard/deep/a/...", "//wildcard/deep/c/..."),
              "//wildcard/deep/a",
              "//wildcard/deep/c"),

          // Negative patterns.
          // TODO: blaze-configurability-team - fix handling of negative patterns and re-enable
          create(ImmutableList.of("-//foo/bar:baz", "//foo/bar:quux"), "//foo/bar:quux"),
          create(
              ImmutableList.of("//wildcard/deep/...", "-//wildcard/deep/b/..."),
              "//wildcard/deep/a",
              "//wildcard/deep/c"));
    }

    private static void createBuildFiles(Scratch scratch) throws IOException {
      scratch.file(
          "foo/bar/BUILD",
          """
          filegroup(name = "baz")
          filegroup(name = "quux")
          """);
      scratch.file(
          "wildcard/single/BUILD",
          """
          filegroup(name = "a")
          filegroup(name = "b")
          filegroup(name = "c")
          """);
      scratch.file(
          "wildcard/deep/a/BUILD",
          """
          filegroup(name = "a")
          """);
      scratch.file(
          "wildcard/deep/b/BUILD",
          """
          filegroup(name = "b_1")
          filegroup(name = "b_2")
          """);
      scratch.file(
          "wildcard/deep/c/BUILD",
          """
          filegroup(name = "c")
          """);
    }
  }

  // Test setup and methods.
  private ImmutableList<Label> expandTargetPattern(
      ImmutableList<String> rawPatterns, FilteringPolicy filteringPolicy)
      throws InterruptedException {
    ExpandTargetPatternKey key = new ExpandTargetPatternKey(rawPatterns, filteringPolicy);
    EvaluationResult<ExpandTargetPatternValue> result = expandTargetPattern(key);
    assertThatEvaluationResult(result).hasNoError();
    assertThatEvaluationResult(result).hasEntryThat(key).isNotNull();

    return result.get(key).result();
  }

  private EvaluationResult<ExpandTargetPatternValue> expandTargetPattern(ExpandTargetPatternKey key)
      throws InterruptedException {
    try {
      // Must re-enable analysis for Skyframe functions that create configured targets.
      skyframeExecutor.getSkyframeBuildView().enableAnalysis(true);
      return SkyframeExecutorTestUtils.evaluate(
          skyframeExecutor, key, /* keepGoing= */ false, reporter);
    } finally {
      skyframeExecutor.getSkyframeBuildView().enableAnalysis(false);
    }
  }

  @Override
  protected AnalysisMock getAnalysisMock() {
    return new AnalysisMockWithExpandTargetPatternFunction();
  }

  private static final SkyFunctionName EXPAND_TARGET_PATTERNS_FUNCTION =
      SkyFunctionName.createHermetic("EXPAND_TARGET_PATTERNS_FUNCTION");

  /**
   * An {@link AnalysisMock} that injects {@link ExpandTargetPatternFunction} into the Skyframe
   * executor.
   */
  private static final class AnalysisMockWithExpandTargetPatternFunction
      extends AnalysisMock.Delegate {
    AnalysisMockWithExpandTargetPatternFunction() {
      super(AnalysisMock.get());
    }

    @Override
    public ImmutableMap<SkyFunctionName, SkyFunction> getSkyFunctions(
        BlazeDirectories directories) {
      return ImmutableMap.<SkyFunctionName, SkyFunction>builder()
          .putAll(super.getSkyFunctions(directories))
          .put(EXPAND_TARGET_PATTERNS_FUNCTION, new ExpandTargetPatternFunction())
          .buildOrThrow();
    }
  }

  @AutoCodec
  record ExpandTargetPatternKey(ImmutableList<String> rawPatterns, FilteringPolicy filteringPolicy)
      implements SkyKey {
    ExpandTargetPatternKey {
      requireNonNull(rawPatterns);
      requireNonNull(filteringPolicy);
    }

    @Override
    public SkyFunctionName functionName() {
      return EXPAND_TARGET_PATTERNS_FUNCTION;
    }
  }

  @AutoCodec
  record ExpandTargetPatternValue(ImmutableList<Label> result) implements SkyValue {}

  private static final class ExpandTargetPatternFunction implements SkyFunction {

    @Nullable
    @Override
    public SkyValue compute(SkyKey skyKey, Environment env)
        throws InterruptedException, ExpandTargetPatternFunctionException {
      ExpandTargetPatternKey key = (ExpandTargetPatternKey) skyKey;

      RepositoryMappingValue mainRepoMapping =
          (RepositoryMappingValue) env.getValue(RepositoryMappingValue.key(RepositoryName.MAIN));
      if (env.valuesMissing()) {
        return null;
      }
      TargetPattern.Parser targetPatternParser =
          new TargetPattern.Parser(
              PathFragment.EMPTY_FRAGMENT,
              RepositoryName.MAIN,
              mainRepoMapping.repositoryMapping());

      try {
        ImmutableList<SignedTargetPattern> signedTargetPatterns =
            TargetPatternUtil.parseAllSigned(key.rawPatterns(), targetPatternParser);
        ImmutableList<Label> labels =
            TargetPatternUtil.expandTargetPatterns(
                env, signedTargetPatterns, key.filteringPolicy());
        if (env.valuesMissing()) {
          return null;
        }

        return new ExpandTargetPatternValue(labels);
      } catch (InvalidTargetPatternException e) {
        throw new ExpandTargetPatternFunctionException(e);
      }
    }
  }

  private static final class ExpandTargetPatternFunctionException extends SkyFunctionException {

    private ExpandTargetPatternFunctionException(InvalidTargetPatternException e) {
      super(e, Transience.PERSISTENT);
    }
  }
}
