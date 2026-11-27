// Copyright 2026 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skyframe;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.skyframe.serialization.testutils.Dumper.dumpStructureWithEquivalenceReduction;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.ActionLookupValue;
import com.google.devtools.build.lib.analysis.AnalysisResult;
import com.google.devtools.build.lib.analysis.ConfiguredAspect;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.ConfiguredTargetValue;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.AnalysisTestCase;
import com.google.devtools.build.lib.analysis.util.TestAspects.FileProviderAspect;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationDepsUtils;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import com.google.devtools.build.lib.skyframe.util.SkyframeExecutorTestUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class RemoteActionLookupValueSerializationTest extends AnalysisTestCase {

  @Test
  public void testDeserializedAspect_hasProvidersAndActions() throws Exception {
    FileProviderAspect aspect = new FileProviderAspect();
    setRulesAndAspectsAvailableInTests(ImmutableList.of(aspect), ImmutableList.of());
    scratch.file(
        "a/BUILD",
        "load('//test_defs:foo_binary.bzl', 'foo_binary')",
        "foo_binary(name = 'a', srcs = ['a.sh'])");

    AnalysisResult analysisResult =
        update(new EventBus(), defaultFlags(), ImmutableList.of(aspect.getName()), "//a:a");

    ConfiguredAspect configuredAspect =
        Iterables.getOnlyElement(analysisResult.getAspectsMap().values());
    assertThat(configuredAspect.getActions()).isNotEmpty();
    assertThat(configuredAspect).isInstanceOf(ActionLookupValue.class);
    assertThat(((ActionLookupValue) configuredAspect).getActions()).isNotEmpty();

    var tester = makeSerializationTester(configuredAspect);

    tester.setVerificationFunction(
        (in, out) -> {
          assertThat(out).isInstanceOf(ActionLookupValue.class);
          assertThat(out).isInstanceOf(ConfiguredAspect.class);

          ConfiguredAspect outConfiguredAspect = (ConfiguredAspect) out;

          assertThat(dumpStructureWithEquivalenceReduction(outConfiguredAspect.getActions()))
              .isEqualTo(dumpStructureWithEquivalenceReduction(configuredAspect.getActions()));
          assertThat(dumpStructureWithEquivalenceReduction(outConfiguredAspect.getProviders()))
              .isEqualTo(dumpStructureWithEquivalenceReduction(configuredAspect.getProviders()));
        });

    tester.runTests();
  }

  @Test
  public void ruleConfiguredTargetValue_roundTripsToRemoteConfiguredTargetValue() throws Exception {
    scratch.file(
        "a/BUILD",
        "load('//test_defs:foo_binary.bzl', 'foo_binary')",
        "foo_binary(name = 'a', srcs = ['a.sh'])");

    update("//a:a");
    ConfiguredTarget target = getConfiguredTarget("//a:a");
    BuildConfigurationValue config = getConfiguration(target);
    ConfiguredTargetValue ctValue =
        SkyframeExecutorTestUtils.getExistingConfiguredTargetValue(
            skyframeExecutor, Label.parseCanonical("//a:a"), config);

    var tester = makeSerializationTester(ctValue);

    tester.setVerificationFunction(
        (in, out) -> {
          assertThat(in).isInstanceOf(RuleConfiguredTargetValue.class);
          assertThat(out).isInstanceOf(RemoteConfiguredTargetValue.class);

          RemoteConfiguredTargetValue remoteValue = (RemoteConfiguredTargetValue) out;
          RuleConfiguredTarget ruleTarget = ((RuleConfiguredTargetValue) in).getConfiguredTarget();
          RuleConfiguredTarget remoteTarget =
              (RuleConfiguredTarget) remoteValue.getConfiguredTarget();

          assertThat(dumpStructureWithEquivalenceReduction(remoteTarget))
              .isEqualTo(dumpStructureWithEquivalenceReduction(ruleTarget));

          assertThat(remoteValue.getTargetData().getLabel())
              .isEqualTo(Label.parseCanonicalUnchecked("//a:a"));

          assertThat(remoteValue.isCleared()).isFalse();
          remoteValue.clear(true);
          assertThat(remoteValue.isCleared()).isTrue();
        });

    tester.runTests();
  }

  @Test
  public void nonRuleConfiguredTargetValue_roundTripsToRemoteConfiguredTargetValue()
      throws Exception {
    scratch.file(
        "a/BUILD",
        """
        genrule(
            name = "a",
            srcs = ["a.source"],
            outs = ["a.generated"],
            cmd = "echo a > $@",
        )
        """);
    update("//a:a");

    var inputCtValue =
        SkyframeExecutorTestUtils.getExistingConfiguredTargetValue(
            skyframeExecutor,
            Label.parseCanonical("//a:a.source"),
            getConfiguration(getConfiguredTarget("//a:a.source")));

    var outputCtValue =
        SkyframeExecutorTestUtils.getExistingConfiguredTargetValue(
            skyframeExecutor,
            Label.parseCanonical("//a:a.generated"),
            getConfiguration(getConfiguredTarget("//a:a.generated")));

    var tester = makeSerializationTester(inputCtValue, outputCtValue);

    tester.setVerificationFunction(
        (in, out) -> {
          assertThat(in).isInstanceOf(NonRuleConfiguredTargetValue.class);
          assertThat(out).isInstanceOf(RemoteConfiguredTargetValue.class);

          RemoteConfiguredTargetValue remoteValue = (RemoteConfiguredTargetValue) out;
          ConfiguredTarget configuredTarget =
              ((NonRuleConfiguredTargetValue) in).getConfiguredTarget();
          ConfiguredTarget remoteTarget = remoteValue.getConfiguredTarget();

          assertThat(dumpStructureWithEquivalenceReduction(configuredTarget))
              .isEqualTo(dumpStructureWithEquivalenceReduction(remoteTarget));

          assertThat(remoteValue.getTargetData().getLabel().getPackageName()).isEqualTo("a");

          assertThat(remoteValue.isCleared()).isFalse();
          remoteValue.clear(true);
          assertThat(remoteValue.isCleared()).isTrue();
        });

    tester.runTests();
  }

  private SerializationTester makeSerializationTester(Object... subjects) {
    return new SerializationTester(subjects)
        .makeMemoizingAndAllowFutureBlocking(true)
        .addDependency(RuleClassProvider.class, ruleClassProvider)
        .addDependencies(SerializationDepsUtils.SERIALIZATION_DEPS_FOR_TEST)
        .addDependencies(getCommonSerializationDependencies())
        .addDependency(PrerequisitePackageFunction.class, skyframeExecutor::getExistingPackage)
        .addCodec(RemoteConfiguredTargetValue.codec());
  }
}
