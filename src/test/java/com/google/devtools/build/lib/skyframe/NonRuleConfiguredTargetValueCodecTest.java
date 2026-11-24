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

import com.google.common.collect.ImmutableClassToInstanceMap;
import com.google.devtools.build.lib.actions.Artifact.ArtifactSerializationContext;
import com.google.devtools.build.lib.analysis.config.BuildOptions.MapBackedChecksumCache;
import com.google.devtools.build.lib.analysis.config.BuildOptions.OptionsChecksumCache;
import com.google.devtools.build.lib.analysis.configuredtargets.InputFileConfiguredTarget;
import com.google.devtools.build.lib.analysis.configuredtargets.OutputFileConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestBase;
import com.google.devtools.build.lib.skyframe.serialization.AutoRegistry;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecRegistry;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecs;
import com.google.devtools.build.lib.vfs.Root;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class NonRuleConfiguredTargetValueCodecTest extends BuildViewTestBase {

  @Test
  public void withFileConfiguredTargets() throws Exception {
    scratch.file(
        "foo/BUILD",
        """
        genrule(
            name = "generating_rule",
            srcs = ["a.source"],
            outs = ["generated.source"],
            cmd = "echo hi > $@",
        )
        """);
    update("//foo:generating_rule");
    var generatedSource =
        (OutputFileConfiguredTarget)
            getConfiguredTarget("//foo:generated.source", getExecConfiguration());
    var input =
        (InputFileConfiguredTarget)
            getConfiguredTarget("//foo:a.source", /* configuration= */ null);

    var subject1 =
        new NonRuleConfiguredTargetValue(generatedSource, /* transitivePackages= */ null);
    var subject2 = new NonRuleConfiguredTargetValue(input, /* transitivePackages= */ null);

    ObjectCodecs codecs = prepareCodecs(skyframeExecutor);

    var deserialized1 =
        (NonRuleConfiguredTargetValue)
            codecs.deserializeMemoized(codecs.serializeMemoized(subject1));
    ObjectCodecRegistry registry = codecs.getCodecRegistry();
    assertThat(dumpStructureWithEquivalenceReduction(registry, deserialized1))
        .isEqualTo(dumpStructureWithEquivalenceReduction(registry, subject1));

    var deserialized2 =
        (NonRuleConfiguredTargetValue)
            codecs.deserializeMemoized(codecs.serializeMemoized(subject2));
    assertThat(dumpStructureWithEquivalenceReduction(registry, deserialized2))
        .isEqualTo(dumpStructureWithEquivalenceReduction(registry, subject2));
  }

  /** Builds an {@link ObjectCodecs} instance with minimum dependencies needed to serialize. */
  private static ObjectCodecs prepareCodecs(SkyframeExecutor executor) {
    ObjectCodecRegistry registry =
        AutoRegistry.get()
            .getBuilder()
            .addReferenceConstant(
                executor.getBlazeDirectoriesForTesting().getOutputBase().getFileSystem())
            .build();

    ImmutableClassToInstanceMap<Object> dependencies =
        ImmutableClassToInstanceMap.builder()
            .put(OptionsChecksumCache.class, new MapBackedChecksumCache())
            .put(Root.RootCodecDependencies.class, new Root.RootCodecDependencies())
            .put(
                ArtifactSerializationContext.class,
                executor.getSkyframeBuildView().getArtifactFactory()::getSourceArtifact)
            .build();

    return new ObjectCodecs(registry, dependencies);
  }
}
