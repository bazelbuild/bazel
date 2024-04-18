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
package com.google.devtools.build.lib.analysis.producers;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.Fragment;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.RequiresOptions;
import com.google.devtools.build.lib.skyframe.config.BuildConfigurationKey;
import com.google.devtools.build.lib.skyframe.config.PlatformMappingException;
import com.google.devtools.build.lib.skyframe.toolchains.PlatformLookupUtil.InvalidPlatformException;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.skyframe.state.StateMachine;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.Map;
import javax.annotation.Nullable;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests of {@link BuildConfigurationKeyProducer}. */
@RunWith(JUnit4.class)
public class BuildConfigurationKeyProducerTest extends ProducerTestCase {
  /** Extra options for this test. */
  public static class DummyTestOptions extends FragmentOptions {
    public DummyTestOptions() {}

    @Option(
        name = "internal_option",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        defaultValue = "super secret",
        metadataTags = {OptionMetadataTag.INTERNAL})
    public String internalOption;
  }

  /** Test fragment. */
  @RequiresOptions(options = {DummyTestOptions.class})
  public static final class DummyTestOptionsFragment extends Fragment {
    private final BuildOptions buildOptions;

    public DummyTestOptionsFragment(BuildOptions buildOptions) {
      this.buildOptions = buildOptions;
    }

    // Getter required to satisfy AutoCodec.
    public BuildOptions getBuildOptions() {
      return buildOptions;
    }
  }

  @Override
  protected ConfiguredRuleClassProvider createRuleClassProvider() {
    ConfiguredRuleClassProvider.Builder builder = new ConfiguredRuleClassProvider.Builder();
    TestRuleClassProvider.addStandardRules(builder);
    builder.addConfigurationFragment(DummyTestOptionsFragment.class);
    return builder.build();
  }

  @Test
  public void createKey() throws Exception {
    BuildOptions baseOptions = createBuildOptions("--internal_option=from_cmd");
    BuildConfigurationKey result = fetch(baseOptions);

    assertThat(result).isNotNull();
    assertThat(result.getOptions().contains(DummyTestOptions.class)).isTrue();
    assertThat(result.getOptions().get(DummyTestOptions.class).internalOption)
        .isEqualTo("from_cmd");
  }

  @Test
  public void createKey_platformMapping() throws Exception {
    scratch.file(
        "/workspace/platform_mappings",
        "platforms:",
        "  //:sample",
        "    --internal_option=from_mapping");
    scratch.file("BUILD", "platform(name = 'sample')");
    invalidatePackages(false);

    BuildOptions baseOptions = createBuildOptions("--platforms=//:sample");
    BuildConfigurationKey result = fetch(baseOptions);

    assertThat(result).isNotNull();
    assertThat(result.getOptions().contains(DummyTestOptions.class)).isTrue();
    assertThat(result.getOptions().get(DummyTestOptions.class).internalOption)
        .isEqualTo("from_mapping");
  }

  @Test
  public void createKey_platformMapping_invalidFile() throws Exception {
    scratch.file("/workspace/platform_mappings", "not a mapping file");
    scratch.file("BUILD", "platform(name = 'sample')");
    invalidatePackages(false);

    BuildOptions baseOptions = createBuildOptions("--platforms=//:sample");
    assertThrows(PlatformMappingException.class, () -> fetch(baseOptions));
  }

  @Test
  public void createKey_platformMapping_invalidOption() throws Exception {
    scratch.file("/workspace/platform_mappings", "platforms:", "  //:sample", "    --fake_option");
    scratch.file("BUILD", "platform(name = 'sample')");
    invalidatePackages(false);

    BuildOptions baseOptions = createBuildOptions("--platforms=//:sample");
    assertThrows(OptionsParsingException.class, () -> fetch(baseOptions));
  }

  private BuildConfigurationKey fetch(BuildOptions options)
      throws InterruptedException,
          OptionsParsingException,
          PlatformMappingException,
          InvalidPlatformException {
    ImmutableMap<String, BuildConfigurationKey> result = fetch(ImmutableMap.of("only", options));
    return result.get("only");
  }

  private ImmutableMap<String, BuildConfigurationKey> fetch(Map<String, BuildOptions> options)
      throws InterruptedException,
          OptionsParsingException,
          PlatformMappingException,
          InvalidPlatformException {
    Sink sink = new Sink();
    BuildConfigurationKeyProducer producer =
        new BuildConfigurationKeyProducer(sink, StateMachine.DONE, options);
    // Ignore the return value: sink will either return a result or re-throw whatever exception it
    // received from the producer.
    var unused = executeProducer(producer);
    return sink.options();
  }

  /** Receiver for platform info from {@link PlatformInfoProducer}. */
  private static class Sink implements BuildConfigurationKeyProducer.ResultSink {
    @Nullable private OptionsParsingException optionsParsingException;
    @Nullable private PlatformMappingException platformMappingException;
    @Nullable private InvalidPlatformException invalidPlatformException;
    @Nullable private ImmutableMap<String, BuildConfigurationKey> keys;

    @Override
    public void acceptTransitionError(OptionsParsingException e) {
      this.optionsParsingException = e;
    }

    @Override
    public void acceptPlatformMappingError(PlatformMappingException e) {
      this.platformMappingException = e;
    }

    @Override
    public void acceptPlatformFlagsError(InvalidPlatformException e) {
      this.invalidPlatformException = e;
    }

    @Override
    public void acceptTransitionedConfigurations(ImmutableMap<String, BuildConfigurationKey> keys) {
      this.keys = keys;
    }

    ImmutableMap<String, BuildConfigurationKey> options()
        throws OptionsParsingException, PlatformMappingException, InvalidPlatformException {
      if (this.optionsParsingException != null) {
        throw this.optionsParsingException;
      }
      if (this.platformMappingException != null) {
        throw this.platformMappingException;
      }
      if (this.invalidPlatformException != null) {
        throw this.invalidPlatformException;
      }
      if (this.keys != null) {
        return this.keys;
      }
      throw new IllegalStateException("Value and exception not set");
    }
  }
}
