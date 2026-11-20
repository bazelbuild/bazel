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
package com.google.devtools.build.lib.skyframe.config;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.config.OutputPathMnemonicComputer;
import com.google.devtools.build.lib.analysis.config.transitions.BaselineOptionsValue;
import com.google.devtools.build.lib.analysis.util.AnalysisTestCase;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.HashMap;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ConfigurationMnemonicLookup}. */
@RunWith(JUnit4.class)
public final class ConfigurationMnemonicLookupTest extends AnalysisTestCase {

  @Test
  public void getMnemonic() throws Exception {
    scratch.file("test/BUILD", "filegroup(name = 'test')");
    useConfiguration("--compilation_mode=dbg", "--cc_output_directory_tag=tag");
    update("//test:test");
    BuildConfigurationValue config = getTargetConfiguration();

    assertThat(skyframeExecutor.getConfigurationMnemonicLookup().getMnemonic(config.getKey()))
        .isEqualTo(config.getMnemonic());

    // Without the configuration or its baseline in the graph, the mnemonic is unknown, except for
    // the no-config configuration, which has no baseline.
    Map<SkyKey, SkyValue> doneValues = new HashMap<>();
    var lookup = new ConfigurationMnemonicLookup(doneValues::get);
    assertThrows(SerializationException.class, () -> lookup.getMnemonic(config.getKey()));
    BuildConfigurationKey noConfigKey =
        BuildConfigurationKey.create(
            BuildOptions.getDefaultBuildOptionsForFragments(ImmutableList.of(CoreOptions.class)));
    assertThat(lookup.getMnemonic(noConfigKey)).isEqualTo("fastbuild-noconfig");

    // Once the baseline is in the graph (e.g. fetched when deserializing a derived artifact root),
    // getMnemonic computes the mnemonic from the baseline and caches it.
    BaselineOptionsValue.Key baselineKey =
        BuildConfigurationFunction.baselineOptionsKey(config.getOptions());
    var baseline =
        (BaselineOptionsValue) skyframeExecutor.getEvaluator().getExistingValue(baselineKey);
    doneValues.put(baselineKey, baseline);
    assertThat(lookup.getMnemonic(config.getKey())).isEqualTo(config.getMnemonic());

    // If Skyframe re-evaluates BaselineOptionsValue (e.g. after a flag change on an incremental
    // build), the cached mnemonic for the old baseline is invalidated and recomputed.
    useConfiguration("--compilation_mode=dbg", "--cc_output_directory_tag=tag", "--copt=-DFOO");
    update("//test:test");
    var newBaseline =
        (BaselineOptionsValue) skyframeExecutor.getEvaluator().getExistingValue(baselineKey);
    doneValues.put(baselineKey, newBaseline);
    assertThat(lookup.getMnemonic(config.getKey()))
        .isEqualTo(
            OutputPathMnemonicComputer.computeMnemonic(config.getOptions(), newBaseline.toOptions())
                .mnemonic());
    assertThat(lookup.getMnemonic(config.getKey())).isNotEqualTo(config.getMnemonic());
  }
}
