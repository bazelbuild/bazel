// Copyright 2022 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.test.AnalysisFailure;
import com.google.devtools.build.lib.analysis.test.AnalysisFailureInfo;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.analysis.util.MockRule;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import javax.annotation.Nullable;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkSemantics;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests verifying analysis failure propagation via {@link AnalysisFailureInfo} when {@code
 * --allow_analysis_failures=true}.
 */
// TODO(arostovtsev): move all other `--allow_analysis_failures`-related test cases here.
@RunWith(JUnit4.class)
public final class AnalysisFailureInfoTest extends BuildViewTestCase {

  @Before
  public void setUp() throws Exception {
    useConfiguration("--allow_analysis_failures=true");
  }

  @Test
  public void analysisFailureInfoStarlarkApi() throws Exception {
    Label label = Label.create("test", "test");
    AnalysisFailure failure = new AnalysisFailure(label, "ErrorMessage");
    assertThat(getattr(failure, "label")).isSameInstanceAs(label);
    assertThat(getattr(failure, "message")).isEqualTo("ErrorMessage");

    AnalysisFailureInfo info = AnalysisFailureInfo.forAnalysisFailures(ImmutableList.of(failure));
    // info.causes.to_list()[0] == failure
    NestedSet<AnalysisFailure> causes =
        Depset.cast(getattr(info, "causes"), AnalysisFailure.class, "causes");
    assertThat(causes.toList().get(0)).isSameInstanceAs(failure);
  }

  private static Object getattr(Object x, String name) throws Exception {
    return Starlark.getattr(/*mu=*/ null, StarlarkSemantics.DEFAULT, x, name, null);
  }

  // Regression test for b/154007057
  @Test
  public void nativeRuleExpanderFailure() throws Exception {
    scratch.file(
        "test/BUILD", //
        "genrule(",
        "    name = 'bad_variable',",
        "    outs = ['bad.out'],",
        "    cmd = 'cp $< $@',  # Error to use $< with no srcs",
        ")");

    ConfiguredTarget target = getConfiguredTarget("//test:bad_variable");
    AnalysisFailureInfo info =
        (AnalysisFailureInfo) target.get(AnalysisFailureInfo.STARLARK_CONSTRUCTOR.getKey());
    AnalysisFailure failure = info.getCauses().getSet(AnalysisFailure.class).toList().get(0);
    assertThat(failure.getMessage()).contains("variable '$<' : no input file");
    assertThat(failure.getLabel()).isEqualTo(Label.parseAbsoluteUnchecked("//test:bad_variable"));
  }

  // Regression test for b/154007057
  @Test
  public void nativeRuleConfiguredTargetFactoryCreateReturningNull() throws Exception {
    scratch.file(
        "test/BUILD", //
        "native_rule_with_failing_configured_target_factory(",
        "    name = 'bad_factory',",
        ")");

    ConfiguredTarget target = getConfiguredTarget("//test:bad_factory");
    AnalysisFailureInfo info =
        (AnalysisFailureInfo) target.get(AnalysisFailureInfo.STARLARK_CONSTRUCTOR.getKey());
    AnalysisFailure failure = info.getCauses().getSet(AnalysisFailure.class).toList().get(0);
    assertThat(failure.getMessage()).contains("FailingRuleConfiguredTargetFactory.create() fails");
    assertThat(failure.getLabel()).isEqualTo(Label.parseAbsoluteUnchecked("//test:bad_factory"));
  }

  /** Dummy factory whose {@code create()} method always returns {@code null}. */
  public static final class FailingRuleConfiguredTargetFactory
      implements RuleConfiguredTargetFactory {
    @Override
    @Nullable
    public ConfiguredTarget create(RuleContext ruleContext) {
      ruleContext.ruleError("FailingRuleConfiguredTargetFactory.create() fails");
      return null;
    }
  }

  @Override
  protected ConfiguredRuleClassProvider createRuleClassProvider() {
    ConfiguredRuleClassProvider.Builder builder =
        new ConfiguredRuleClassProvider.Builder()
            .addRuleDefinition(
                ((MockRule)
                    () ->
                        MockRule.factory(FailingRuleConfiguredTargetFactory.class)
                            .define("native_rule_with_failing_configured_target_factory")));
    TestRuleClassProvider.addStandardRules(builder);
    return builder.build();
  }
}
