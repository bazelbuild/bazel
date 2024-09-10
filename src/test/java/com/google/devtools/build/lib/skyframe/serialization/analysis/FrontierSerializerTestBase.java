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
package com.google.devtools.build.lib.skyframe.serialization.analysis;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.buildtool.BuildTool;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.pkgcache.LoadingFailedException;
import com.google.devtools.build.lib.skyframe.AspectKeyCreator.AspectBaseKey;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.serialization.analysis.FrontierSerializer.SelectionMarking;
import com.google.devtools.build.skyframe.InMemoryGraph;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.perftools.profiles.ProfileProto.Profile;
import com.google.protobuf.ExtensionRegistry;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import org.junit.Before;
import org.junit.Test;

public abstract class FrontierSerializerTestBase extends BuildIntegrationTestCase {

  @Before
  public void setCommonConfiguration() {
    addOptions(
        // Required to use PROJECT.scl.
        "--experimental_enable_scl_dialect",
        // Note that `--nobuild` disables skymeld, because skymeld has no utility
        // without the execution phase
        "--nobuild");
  }

  @Test
  public void serializingFrontierWithNoProjectFile_hasError() throws Exception {
    write(
        "foo/BUILD",
        """
        package_group(name = "empty")
        """);
    addOptions("--serialized_frontier_profile=/tmp/anything");
    LoadingFailedException exception =
        assertThrows(LoadingFailedException.class, () -> buildTarget("//foo:empty"));
    assertThat(exception).hasMessageThat().contains("Failed to find PROJECT.scl file");
  }

  @Test
  public void serializingFrontierWithProjectFile_hasNoError() throws Exception {
    write(
        "foo/BUILD",
        """
        package_group(name = "empty")
        """);
    write(
        "foo/PROJECT.scl",
        """
        active_directories = {"default": ["foo"] }
        """);
    addOptions("--serialized_frontier_profile=/tmp/anything");
    assertThat(buildTarget("//foo:empty").getSuccess()).isTrue();
  }

  @Test
  public void serializingWithMultipleTopLevelProjectFiles_hasError() throws Exception {
    write(
        "foo/BUILD",
        """
        package_group(name = "empty")
        """);
    write(
        "foo/PROJECT.scl",
        """
        active_directories = {"default": ["foo"] }
        """);

    write(
        "bar/BUILD",
        """
        package_group(name = "empty")
        """);
    write(
        "bar/PROJECT.scl",
        """
        active_directories = {"default": ["bar"] }
        """);

    addOptions("--serialized_frontier_profile=/tmp/anything");
    LoadingFailedException exception =
        assertThrows(LoadingFailedException.class, () -> buildTarget("//foo:empty", "//bar:empty"));
    assertThat(exception)
        .hasMessageThat()
        .contains(
            "This build doesn't support automatic project resolution. Targets have different"
                + " project settings.");
  }

  @Test
  public void serializingWithMultipleTargetsResolvingToSameProjectFile_hasNoError()
      throws Exception {
    write(
        "foo/BUILD",
        """
        package_group(name = "empty")
        """);
    write(
        "foo/bar/BUILD",
        """
        package_group(name = "empty")
        """);
    write(
        "foo/PROJECT.scl",
        """
        active_directories = {"default": ["foo"] }
        """);

    addOptions("--serialized_frontier_profile=/tmp/anything");
    assertThat(buildTarget("//foo:empty", "//foo/bar:empty").getSuccess()).isTrue();
  }

  @Test
  public void activeAspect_activatesBaseConfiguredTarget() throws Exception {
    setupScenarioWithAspects("--serialized_frontier_profile=/tmp/anything");
    InMemoryGraph graph = getSkyframeExecutor().getEvaluator().getInMemoryGraph();

    ConfiguredTargetKey generateYKey =
        ConfiguredTargetKey.builder()
            .setLabel(label("//foo:generate_y"))
            .setConfiguration(getTargetConfiguration())
            .build();

    // Determines the UTC of //foo:generate_y.
    Set<ActionLookupKey> utc = Sets.newConcurrentHashSet();
    collectAnalysisUtc(graph, generateYKey, utc);
    assertThat(utc.remove(generateYKey)).isTrue();

    // The strict UTC has 3 keys. One for the two targets in bar and one for the top level aspect.
    assertThat(utc).hasSize(3);
    // Everything in the UTC is an aspect. This means the underlying configured targets are *not*
    // included. This is a dangerous situation. The danger is not present in the actual code because
    // the underlying configured targets _will_ be computed by
    // `FrontierSerializer#computeSelection`.
    for (ActionLookupKey key : utc) {
      assertThat(key).isInstanceOf(AspectBaseKey.class);
    }

    Map<ActionLookupKey, SelectionMarking> selection =
        FrontierSerializer.computeSelection(
            graph,
            BuildTool.getWorkingSetMatcherForSkyfocus(
                // We know exactly which PROJECT file is used, so inject it here.
                Label.parseCanonicalUnchecked("//bar:PROJECT.scl"),
                getSkyframeExecutor(),
                getCommandEnvironment().getReporter()));

    ImmutableSet<ActionLookupKey> activeKeys =
        selection.entrySet().stream()
            .filter(entry -> entry.getValue().equals(SelectionMarking.ACTIVE))
            .map(Map.Entry::getKey)
            .collect(toImmutableSet());

    var expected = new HashSet<ActionLookupKey>();
    expected.add(generateYKey);
    expected.add(ConfiguredTargetKey.builder().setLabel(label("//foo:x.txt")).build());
    expected.addAll(utc);
    for (ActionLookupKey key : utc) {
      // The base configured target keys are included in the active set.
      expected.add(((AspectBaseKey) key).getBaseConfiguredTargetKey());
    }

    assertThat(activeKeys).containsExactlyElementsIn(expected);
  }

  private void collectAnalysisUtc(
      InMemoryGraph graph, ActionLookupKey next, Set<ActionLookupKey> utc) {
    if (!utc.add(next)) {
      return;
    }
    for (SkyKey rdep : graph.getIfPresent(next).getReverseDepsForDoneEntry()) {
      if (!(rdep instanceof ActionLookupKey parent)) {
        continue;
      }
      collectAnalysisUtc(graph, parent, utc);
    }
  }

  @Test
  public void buildCommand_serializedFrontierProfileContainsExpectedClasses() throws Exception {
    @SuppressWarnings("UnnecessarilyFullyQualified") // to avoid confusion with vfs Paths
    Path profilePath = Files.createTempFile(null, "profile");

    setupScenarioWithAspects("--serialized_frontier_profile=" + profilePath);

    // The proto parses successfully from the file.
    var proto =
        Profile.parseFrom(Files.readAllBytes(profilePath), ExtensionRegistry.getEmptyRegistry());

    // The exact contents of the proto could change easily based on the underlying implementation.
    // Constructs a rather coarse assertion on the top-level entries of the proto that is hoped to
    // be relatively robust.

    // Constructs a table of location ID to class name.
    var classNames = new HashMap<Integer, String>();
    proto.getFunctionList().stream()
        .forEach(
            function -> {
              // All the names are formatted <class name>(<codec name>). Keeps just the class name
              // because the codec name could easily change.
              String fullName = proto.getStringTableList().get((int) function.getName());
              int classNameEnd = fullName.indexOf("(");
              if (classNameEnd == -1) {
                return; // not a class name
              }
              classNames.put((int) function.getId(), fullName.substring(0, classNameEnd));
            });

    ImmutableList<String> topLevelClassNames =
        proto.getSampleList().stream()
            .filter(sample -> sample.getLocationIdCount() == 1)
            .map(sample -> classNames.get((int) sample.getLocationId(0)))
            .collect(toImmutableList());

    // These top-level class names should be relatively stable.
    assertThat(topLevelClassNames)
        .containsAtLeast(
            "com.google.devtools.build.lib.actions.Artifact.DerivedArtifact",
            "com.google.devtools.build.lib.actions.Artifact.SourceArtifact",
            "com.google.devtools.build.lib.analysis.ConfiguredTargetValue",
            "com.google.devtools.build.lib.skyframe.ConfiguredTargetKey",
            "com.google.devtools.build.lib.cmdline.Label",
            "java.lang.Object[]");

    assertWithMessage(
            "ConfiguredTargetValue subtypes should be represented in the profile as"
                + " ConfiguredTargetValue")
        .that(topLevelClassNames)
        .doesNotContain("com.google.devtools.build.lib.skyframe.RuleConfiguredTargetValue");

    assertWithMessage(
            "ConfiguredTargetValue subtypes should be represented in the profile as"
                + " ConfiguredTargetValue")
        .that(topLevelClassNames)
        .doesNotContain("com.google.devtools.build.lib.skyframe.NonRuleConfiguredTargetValue");
  }

  private void setupScenarioWithAspects(String... options) throws Exception {
    write(
        "foo/provider.bzl",
        """
FileCountInfo = provider(
    fields = {
        'count' : 'number of files'
    }
)
""");

    write(
        "foo/file_count.bzl",
        """
load("//foo:provider.bzl", "FileCountInfo")
def _file_count_aspect_impl(target, ctx):
    count = 0
    # Make sure the rule has a srcs attribute.
    if hasattr(ctx.rule.attr, 'srcs'):
        # Iterate through the sources counting files.
        for src in ctx.rule.attr.srcs:
            for f in src.files.to_list():
                count = count + 1
    # Get the counts from our dependencies.
    for dep in ctx.rule.attr.deps:
        count = count + dep[FileCountInfo].count
    return [FileCountInfo(count = count)]

file_count_aspect = aspect(
    implementation = _file_count_aspect_impl,
    attr_aspects = ['deps'],
    attrs = {
      "_y" : attr.label(default="//foo:generate_y"),
    },
)
""");

    write(
        "bar/PROJECT.scl",
        """
active_directories = {"default": ["foo"] }
""");

    write(
        "foo/BUILD",
        """
genrule(
    name = "generate_y",
    srcs = ["x.txt"],
    outs = ["y.txt"],
    cmd = "cat $< > $@",
)
""");

    write(
        "bar/BUILD",
        """
java_library(
    name = "one",
    srcs = ["One.java"],
    deps = [":two"],
)

java_library(
    name = "two",
    srcs = ["Two.java", "TwoA.java", "TwoB.java"],
)

# This genrule creates a DerivedArtifact and NestedSet dep to be serialized in the frontier.
#
# Without this, the test fails in the Bazel source tree with missing expected java.lang.Object[]
# and com.google.devtools.build.lib.actions.Artifact.DerivedArtifact from the actual
# topLevelClassNames. In Blaze-land, these classes were contributed by implicit frontier
# dependencies not found in Bazel-land, so this genrule ensures that the test do not rely on
# Blaze-land side-effects.
genrule(
    name = "two_gen",
    outs = ["TwoA.java", "TwoB.java"],
    cmd = "touch $(OUTS)",
)
""");

    addOptions("--aspects=//foo:file_count.bzl%file_count_aspect");
    addOptions(options);
    assertThat(buildTarget("//bar:one").getSuccess()).isTrue();
  }

}

