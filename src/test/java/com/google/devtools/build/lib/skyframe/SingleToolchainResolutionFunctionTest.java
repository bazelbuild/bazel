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

package com.google.devtools.build.lib.skyframe;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.skyframe.EvaluationResultSubjectFactory.assertThatEvaluationResult;

import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.rules.platform.ToolchainTestCase;
import com.google.devtools.build.lib.skyframe.util.SkyframeExecutorTestUtils;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyKey;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkSemantics;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link SingleToolchainResolutionValue} and {@link SingleToolchainResolutionFunction}.
 */
@RunWith(JUnit4.class)
public class SingleToolchainResolutionFunctionTest extends ToolchainTestCase {
  ConfiguredTargetKey linuxCtkey;
  ConfiguredTargetKey macCtkey;

  @Before
  public void setUpKeys() {
    // This has to happen here so that targetConfiguration is populated.
    linuxCtkey =
        ConfiguredTargetKey.builder()
            .setLabel(Label.parseCanonicalUnchecked("//platforms:linux"))
            .setConfiguration(getTargetConfiguration())
            .build();
    macCtkey =
        ConfiguredTargetKey.builder()
            .setLabel(Label.parseCanonicalUnchecked("//platforms:mac"))
            .setConfiguration(getTargetConfiguration())
            .build();
  }

  private EvaluationResult<SingleToolchainResolutionValue> invokeToolchainResolution(SkyKey key)
      throws InterruptedException {
    try {
      getSkyframeExecutor().getSkyframeBuildView().enableAnalysis(true);
      return SkyframeExecutorTestUtils.evaluate(
          getSkyframeExecutor(), key, /*keepGoing=*/ false, reporter);
    } finally {
      getSkyframeExecutor().getSkyframeBuildView().enableAnalysis(false);
    }
  }

  @Test
  public void testResolution_singleExecutionPlatform() throws Exception {
    SkyKey key =
        SingleToolchainResolutionValue.key(
            targetConfigKey,
            testToolchainType,
            testToolchainTypeInfo,
            linuxCtkey,
            ImmutableList.of(macCtkey));
    EvaluationResult<SingleToolchainResolutionValue> result = invokeToolchainResolution(key);

    assertThatEvaluationResult(result).hasNoError();

    SingleToolchainResolutionValue singleToolchainResolutionValue = result.get(key);
    assertThat(singleToolchainResolutionValue.availableToolchainLabels())
        .containsExactly(macCtkey, Label.parseCanonicalUnchecked("//toolchain:toolchain_2_impl"));
  }

  @Test
  public void testResolution_multipleExecutionPlatforms() throws Exception {
    addToolchain(
        "extra",
        "extra_toolchain",
        ImmutableList.of("//constraints:linux"),
        ImmutableList.of("//constraints:linux"),
        "baz");
    rewriteWorkspace(
        "register_toolchains(",
        "'//toolchain:toolchain_1',",
        "'//toolchain:toolchain_2',",
        "'//extra:extra_toolchain')");

    SkyKey key =
        SingleToolchainResolutionValue.key(
            targetConfigKey,
            testToolchainType,
            testToolchainTypeInfo,
            linuxCtkey,
            ImmutableList.of(linuxCtkey, macCtkey));
    EvaluationResult<SingleToolchainResolutionValue> result = invokeToolchainResolution(key);

    assertThatEvaluationResult(result).hasNoError();

    SingleToolchainResolutionValue singleToolchainResolutionValue = result.get(key);
    assertThat(singleToolchainResolutionValue.availableToolchainLabels())
        .containsExactly(
            linuxCtkey,
            Label.parseCanonicalUnchecked("//extra:extra_toolchain_impl"),
            macCtkey,
            Label.parseCanonicalUnchecked("//toolchain:toolchain_2_impl"));
  }

  @Test
  public void testResolution_noneFound() throws Exception {
    // Clear the toolchains.
    rewriteWorkspace();

    SkyKey key =
        SingleToolchainResolutionValue.key(
            targetConfigKey,
            testToolchainType,
            testToolchainTypeInfo,
            linuxCtkey,
            ImmutableList.of(macCtkey));
    EvaluationResult<SingleToolchainResolutionValue> result = invokeToolchainResolution(key);

    SingleToolchainResolutionValue singleToolchainResolutionValue = result.get(key);
    assertThat(singleToolchainResolutionValue.availableToolchainLabels()).isEmpty();
  }

  @Test
  public void testToolchainResolutionValue_equalsAndHashCode() {
    new EqualsTester()
        .addEqualityGroup(
            SingleToolchainResolutionValue.create(
                testToolchainTypeInfo,
                ImmutableMap.of(
                    linuxCtkey, Label.parseCanonicalUnchecked("//test:toolchain_impl_1"))),
            SingleToolchainResolutionValue.create(
                testToolchainTypeInfo,
                ImmutableMap.of(
                    linuxCtkey, Label.parseCanonicalUnchecked("//test:toolchain_impl_1"))))
        // Different execution platform, same label.
        .addEqualityGroup(
            SingleToolchainResolutionValue.create(
                testToolchainTypeInfo,
                ImmutableMap.of(
                    macCtkey, Label.parseCanonicalUnchecked("//test:toolchain_impl_1"))))
        // Same execution platform, different label.
        .addEqualityGroup(
            SingleToolchainResolutionValue.create(
                testToolchainTypeInfo,
                ImmutableMap.of(
                    linuxCtkey, Label.parseCanonicalUnchecked("//test:toolchain_impl_2"))))
        // Different execution platform, different label.
        .addEqualityGroup(
            SingleToolchainResolutionValue.create(
                testToolchainTypeInfo,
                ImmutableMap.of(
                    macCtkey, Label.parseCanonicalUnchecked("//test:toolchain_impl_2"))))
        // Multiple execution platforms.
        .addEqualityGroup(
            SingleToolchainResolutionValue.create(
                testToolchainTypeInfo,
                ImmutableMap.<ConfiguredTargetKey, Label>builder()
                    .put(linuxCtkey, Label.parseCanonicalUnchecked("//test:toolchain_impl_1"))
                    .put(macCtkey, Label.parseCanonicalUnchecked("//test:toolchain_impl_1"))
                    .buildOrThrow()))
        .testEquals();
  }

  /** Use custom class instead of mock to make sure that the dynamic codecs lookup is correct. */
  class SerializableConfiguredTarget implements ConfiguredTarget {

    private final PlatformInfo platform;

    SerializableConfiguredTarget(PlatformInfo platform) {
      this.platform = platform;
    }

    @Override
    public ImmutableCollection<String> getFieldNames() {
      return null;
    }

    @Nullable
    @Override
    public String getErrorMessageForUnknownField(String field) {
      return null;
    }

    @Nullable
    @Override
    public Object getValue(String name) {
      return null;
    }

    @Override
    public Label getLabel() {
      return null;
    }

    @Nullable
    @Override
    public BuildConfigurationKey getConfigurationKey() {
      return null;
    }

    @Nullable
    @Override
    public <P extends TransitiveInfoProvider> P getProvider(Class<P> provider) {
      return null;
    }

    @Nullable
    @Override
    public Object get(String providerKey) {
      return null;
    }

    @SuppressWarnings("unchecked")
    @Override
    public <T extends Info> T get(BuiltinProvider<T> provider) {
      if (PlatformInfo.PROVIDER.equals(provider)) {
        return (T) this.platform;
      }
      return provider.getValueClass().cast(get(provider.getKey()));
    }

    @Nullable
    @Override
    public Info get(Provider.Key providerKey) {
      return null;
    }

    @Override
    public void repr(Printer printer) {}

    @Override
    public Object getIndex(StarlarkSemantics semantics, Object key) throws EvalException {
      throw Starlark.errorf("Unknown key '%s'", key);
    }

    @Override
    public boolean containsKey(StarlarkSemantics semantics, Object key) {
      return false;
    }
  }
}
