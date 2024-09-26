// Copyright 2024 The Bazel Authors. All rights reserved.
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
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.ConfiguredTargetValue;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.config.BuildOptions.MapBackedChecksumCache;
import com.google.devtools.build.lib.analysis.config.BuildOptions.OptionsChecksumCache;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.AnalysisTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.skyframe.serialization.testutils.Dumper;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationDepsUtils;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import com.google.devtools.build.lib.skyframe.util.SkyframeExecutorTestUtils;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Root;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class RemoteConfiguredTargetValueCodecTest extends AnalysisTestCase {

  // TODO: b/369822761 - Fix this test.
  @Ignore("b/369822761")
  @Test
  public void ruleConfiguredTargetValue_roundTripsToRemoteConfiguredTargetValue() throws Exception {
    scratch.file("a/BUILD", "sh_binary(name = 'a', srcs = ['a.sh'])");

    update("//a:a");
    ConfiguredTarget target = getConfiguredTarget("//a:a");
    BuildConfigurationValue config = getConfiguration(target);
    ConfiguredTargetValue ctValue =
        SkyframeExecutorTestUtils.getExistingConfiguredTargetValue(
            skyframeExecutor, Label.parseCanonical("//a:a"), config);

    var tester =
        new SerializationTester(ctValue)
            .makeMemoizingAndAllowFutureBlocking(true)
            .addDependency(RuleClassProvider.class, ruleClassProvider)
            .addDependencies(SerializationDepsUtils.SERIALIZATION_DEPS_FOR_TEST)
            .addDependency(FileSystem.class, scratch.getFileSystem())
            .addDependency(
                Root.RootCodecDependencies.class,
                new Root.RootCodecDependencies(Root.absoluteRoot(scratch.getFileSystem())))
            .addDependency(OptionsChecksumCache.class, new MapBackedChecksumCache())
            .addDependency(PrerequisitePackageFunction.class, skyframeExecutor::getExistingPackage);

    tester.addCodec(RemoteConfiguredTargetValue.codec());

    tester.setVerificationFunction(
        (in, out) -> {
          assertThat(in).isInstanceOf(RuleConfiguredTargetValue.class);
          assertThat(out).isInstanceOf(RemoteConfiguredTargetValue.class);

          RemoteConfiguredTargetValue remoteValue = (RemoteConfiguredTargetValue) out;
          RuleConfiguredTarget ruleTarget = ((RuleConfiguredTargetValue) in).getConfiguredTarget();
          RuleConfiguredTarget remoteTarget =
              (RuleConfiguredTarget) remoteValue.getConfiguredTarget();

          assertThat(ruleTarget.getActions()).isNotEmpty();
          var exception = assertThrows(NullPointerException.class, remoteTarget::getActions);
          assertThat(exception)
              .hasMessageThat()
              .contains("actions are not available on deserialized instances");

          assertThat(Dumper.dumpStructureWithEquivalenceReduction(ruleTarget))
              .isEqualTo(Dumper.dumpStructureWithEquivalenceReduction(remoteTarget));

          assertThat(remoteValue.getTargetData().getLabel())
              .isEqualTo(Label.parseCanonicalUnchecked("//a:a"));
        });

    // This codec is also stable.
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

    var tester =
        new SerializationTester(inputCtValue, outputCtValue)
            .makeMemoizingAndAllowFutureBlocking(true)
            .addDependency(RuleClassProvider.class, ruleClassProvider)
            .addDependencies(SerializationDepsUtils.SERIALIZATION_DEPS_FOR_TEST)
            .addDependency(FileSystem.class, scratch.getFileSystem())
            .addDependency(
                Root.RootCodecDependencies.class,
                new Root.RootCodecDependencies(Root.absoluteRoot(scratch.getFileSystem())))
            .addDependency(OptionsChecksumCache.class, new MapBackedChecksumCache())
            .addDependency(PrerequisitePackageFunction.class, skyframeExecutor::getExistingPackage);

    tester.addCodec(RemoteConfiguredTargetValue.codec());

    tester.setVerificationFunction(
        (in, out) -> {
          assertThat(in).isInstanceOf(NonRuleConfiguredTargetValue.class);
          assertThat(out).isInstanceOf(RemoteConfiguredTargetValue.class);

          RemoteConfiguredTargetValue remoteValue = (RemoteConfiguredTargetValue) out;
          ConfiguredTarget configuredTarget =
              ((NonRuleConfiguredTargetValue) in).getConfiguredTarget();
          ConfiguredTarget remoteTarget = remoteValue.getConfiguredTarget();

          assertThat(Dumper.dumpStructureWithEquivalenceReduction(configuredTarget))
              .isEqualTo(Dumper.dumpStructureWithEquivalenceReduction(remoteTarget));

          assertThat(remoteValue.getTargetData().getLabel().getPackageName()).isEqualTo("a");
        });

    tester.runTests();
  }
}
