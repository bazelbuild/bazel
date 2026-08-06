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

package com.google.devtools.build.lib.actions;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.util.Fingerprint;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** {@link ActionEnvironment}Test */
@RunWith(JUnit4.class)
public final class ActionEnvironmentTest {

  /** Strips the config segment (e.g. "k8-fastbuild") from an output path. */
  private static final PathMapper STRIP_CONFIG =
      execPath -> execPath.subFragment(0, 1).getRelative(execPath.subFragment(2));

  private final Scratch scratch = new Scratch();

  private Artifact createOutputArtifact(String rootRelativePath) throws IOException {
    ArtifactRoot root =
        ArtifactRoot.asDerivedRoot(
            scratch.dir("/exec"), RootType.OUTPUT, "bazel-out", "k8-fastbuild", "bin");
    return ActionsTestUtil.createArtifact(root, rootRelativePath);
  }

  @Test
  public void compoundEnvOrdering() {
    ActionEnvironment env1 =
        ActionEnvironment.create(
            ImmutableMap.of("FOO", "foo1", "BAR", "bar"), ImmutableSet.of("baz"));
    // entries added by env2 override the existing entries
    ActionEnvironment env2 = env1.withAdditionalFixedVariables(ImmutableMap.of("FOO", "foo2"));

    assertThat(env1.getFixedEnv()).containsExactly("FOO", "foo1", "BAR", "bar");
    assertThat(env1.getInheritedEnv()).containsExactly("baz");

    assertThat(env2.getFixedEnv()).containsExactly("FOO", "foo2", "BAR", "bar");
    assertThat(env2.getInheritedEnv()).containsExactly("baz");
  }

  @Test
  public void fixedInheritedInteraction() {
    ActionEnvironment env =
        ActionEnvironment.create(
                ImmutableMap.of("FIXED_ONLY", "fixed"),
                ImmutableSet.of("INHERITED_ONLY", "FIXED_AND_INHERITED"))
            .withAdditionalFixedVariables(ImmutableMap.of("FIXED_AND_INHERITED", "fixed"));
    Map<String, String> clientEnv =
        ImmutableMap.of("INHERITED_ONLY", "inherited", "FIXED_AND_INHERITED", "inherited");
    Map<String, String> result = new HashMap<>();
    env.resolve(result, clientEnv);

    assertThat(result)
        .containsExactly(
            "FIXED_ONLY",
            "fixed",
            "FIXED_AND_INHERITED",
            "inherited",
            "INHERITED_ONLY",
            "inherited");
  }

  @Test
  public void artifactValueResolution() throws Exception {
    Artifact artifact = createOutputArtifact("pkg/file");
    ActionEnvironment env =
        ActionEnvironment.create(ImmutableMap.of("FIXED", "fixed", "ARTIFACT", artifact));

    assertThat(env.getFixedEnv())
        .containsExactly("FIXED", "fixed", "ARTIFACT", "bazel-out/k8-fastbuild/bin/pkg/file");

    Map<String, String> result = new HashMap<>();
    env.resolve(result, ImmutableMap.of());
    assertThat(result)
        .containsExactly("FIXED", "fixed", "ARTIFACT", "bazel-out/k8-fastbuild/bin/pkg/file");
  }

  @Test
  public void resolve_artifactValueAppliesPathMapper() throws Exception {
    Artifact artifact = createOutputArtifact("pkg/file");
    ActionEnvironment env =
        ActionEnvironment.create(ImmutableMap.of("FIXED", "fixed", "ARTIFACT", artifact));

    Map<String, String> result = new HashMap<>();
    env.resolve(result, ImmutableMap.of(), STRIP_CONFIG);
    assertThat(result).containsExactly("FIXED", "fixed", "ARTIFACT", "bazel-out/bin/pkg/file");
  }

  @Test
  public void resolve_mappedArtifactValueOverridesClientEnv() throws Exception {
    Artifact artifact = createOutputArtifact("pkg/file");
    ActionEnvironment env =
        ActionEnvironment.create(
            ImmutableMap.of("ARTIFACT", artifact), ImmutableSet.of("ARTIFACT"));

    Map<String, String> result = new HashMap<>();
    env.resolve(result, ImmutableMap.of("ARTIFACT", "from_client"), STRIP_CONFIG);
    assertThat(result).containsExactly("ARTIFACT", "from_client");

    result.clear();
    env.resolve(result, ImmutableMap.of(), STRIP_CONFIG);
    assertThat(result).containsExactly("ARTIFACT", "bazel-out/bin/pkg/file");
  }

  @Test
  public void withAdditionalFixedVariables_artifactValues() throws Exception {
    Artifact artifact1 = createOutputArtifact("pkg/file1");
    Artifact artifact2 = createOutputArtifact("pkg/file2");
    ActionEnvironment env =
        ActionEnvironment.create(ImmutableMap.of("FOO", "foo", "ARTIFACT", artifact1))
            .withAdditionalFixedVariables(ImmutableMap.of("BAR", "bar", "ARTIFACT", artifact2));

    assertThat(env.getFixedEnv())
        .containsExactly(
            "FOO", "foo", "BAR", "bar", "ARTIFACT", "bazel-out/k8-fastbuild/bin/pkg/file2");
  }

  @Test
  public void addTo_artifactValueAppliesPathMapper() throws Exception {
    Artifact artifact = createOutputArtifact("pkg/file");
    ActionEnvironment env = ActionEnvironment.create(ImmutableMap.of("ARTIFACT", artifact));

    Fingerprint unmapped = new Fingerprint();
    env.addTo(CoreOptions.OutputPathsMode.OFF, unmapped);
    Fingerprint mapped = new Fingerprint();
    env.addTo(CoreOptions.OutputPathsMode.STRIP, mapped);

    assertThat(mapped.hexDigestAndReset()).isNotEqualTo(unmapped.hexDigestAndReset());
  }

  @Test
  public void addTo_artifactValueFingerprintNotSameAsStringValue() throws Exception {
    Artifact artifact = createOutputArtifact("pkg/file");
    ActionEnvironment artifactEnv = ActionEnvironment.create(ImmutableMap.of("ARTIFACT", artifact));
    ActionEnvironment stringEnv =
        ActionEnvironment.create(ImmutableMap.of("ARTIFACT", artifact.getExecPathString()));

    Fingerprint artifactFingerprint = new Fingerprint();
    artifactEnv.addTo(CoreOptions.OutputPathsMode.STRIP, artifactFingerprint);
    Fingerprint stringFingerprint = new Fingerprint();
    stringEnv.addTo(CoreOptions.OutputPathsMode.OFF, stringFingerprint);

    assertThat(artifactFingerprint.hexDigestAndReset())
        .isNotEqualTo(stringFingerprint.hexDigestAndReset());
  }

  @Test
  public void artifactValueInterning() throws Exception {
    Artifact artifact = createOutputArtifact("pkg/file");
    ActionEnvironment env1 =
        ActionEnvironment.create(
            ImmutableMap.of("FOO", "foo", "ARTIFACT", artifact), ImmutableSet.of("baz"));
    ActionEnvironment env2 =
        ActionEnvironment.create(
            ImmutableMap.of("FOO", "foo", "ARTIFACT", artifact), ImmutableSet.of("baz"));
    ActionEnvironment env3 =
        ActionEnvironment.create(
            ImmutableMap.of("FOO", "foo", "ARTIFACT", artifact.getExecPathString()),
            ImmutableSet.of("baz"));

    assertThat(env2).isSameInstanceAs(env1);
    assertThat(env3).isNotEqualTo(env1);
  }

  @Test
  public void emptyEnvironmentInterning() {
    ActionEnvironment emptyEnvironment =
        ActionEnvironment.create(ImmutableMap.of(), ImmutableSet.of());
    assertThat(emptyEnvironment).isSameInstanceAs(ActionEnvironment.EMPTY);

    ActionEnvironment base =
        ActionEnvironment.create(ImmutableMap.of("FOO", "foo1"), ImmutableSet.of("baz"));
    assertThat(base.withAdditionalFixedVariables(ImmutableMap.of())).isSameInstanceAs(base);
  }
}
