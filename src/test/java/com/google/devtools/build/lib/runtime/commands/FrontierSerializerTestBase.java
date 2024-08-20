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
package com.google.devtools.build.lib.runtime.commands;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;
import static com.google.devtools.build.lib.skyframe.serialization.testutils.Dumper.dumpStructureWithEquivalenceReduction;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.collect.PathFragmentPrefixTrie;
import com.google.devtools.build.lib.runtime.BlazeCommandResult;
import com.google.devtools.build.lib.runtime.commands.FrontierSerializer.ActiveDirectoryMatcher;
import com.google.devtools.build.lib.runtime.commands.FrontierSerializer.DirectoryMatcherError;
import com.google.devtools.build.lib.runtime.commands.FrontierSerializer.SelectionMarking;
import com.google.devtools.build.lib.skyframe.AspectKeyCreator.AspectBaseKey;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.ProjectValue;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.InMemoryGraph;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.perftools.profiles.ProfileProto.Profile;
import com.google.protobuf.ExtensionRegistry;
import java.io.PrintStream;
import java.nio.file.Files;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import org.junit.Before;
import org.junit.Test;

public abstract class FrontierSerializerTestBase extends BuildIntegrationTestCase {

  @Before
  public void setCommonConfiguration() {
    addOptions("--experimental_enable_scl_dialect", "--nobuild");
  }

  @Test
  public void noProject_infersFromTopLevelTargets() throws Exception {
    write(
        "foo/BUILD",
        """
        genrule(
            name = "target",
            srcs = ["x.txt"],
            outs = ["out"],
            cmd = "cat $< > $@",
        )
        """);
    write(
        "foo/PROJECT.scl",
        """
        active_directories = {"default": ["foo"] }
        """);
    assertThat(buildTarget("//foo:target").getSuccess()).isTrue();

    ActiveDirectoryMatcher matcher =
        (ActiveDirectoryMatcher)
            FrontierSerializer.computeDirectoryMatcher(getSkyframeExecutor(), events.reporter());

    // In this case, the computeDirectoryMatcher step above populates this value in Skyframe.
    ProjectValue projectValue =
        (ProjectValue)
            getSkyframeExecutor()
                .getEvaluator()
                .getInMemoryGraph()
                .getIfPresent(new ProjectValue.Key(label("//foo:PROJECT.scl")))
                .getValue();

    assertMatcherEqualsDefaultProjectMatcher(matcher.matcher(), projectValue);
  }

  @Test
  public void singleProjectFileInGraph_returnsDefaultMatcher() throws Exception {
    // Defines a package to avoid a noBuildFile error.
    write(
        "foo/BUILD",
        """
        package_group(
          name = "empty",
        )
        """);
    write(
        "foo/PROJECT.scl",
        """
        active_directories = {"default": ["foo"] }
        """);
    // Performs a trivial build to prime Bazel state.
    assertThat(buildTarget("//foo:empty").getSuccess()).isTrue();

    var projectKey = new ProjectValue.Key(label("//foo:PROJECT.scl"));
    EvaluationResult<SkyValue> projectResult =
        getSkyframeExecutor()
            .evaluateSkyKeys(
                events.reporter(), ImmutableList.of(projectKey), /* keepGoing= */ false);
    ProjectValue projectValue = (ProjectValue) projectResult.get(projectKey);

    ActiveDirectoryMatcher matcher =
        (ActiveDirectoryMatcher)
            FrontierSerializer.computeDirectoryMatcher(getSkyframeExecutor(), events.reporter());
    assertMatcherEqualsDefaultProjectMatcher(matcher.matcher(), projectValue);
  }

  @Test
  public void multipleProjectFilesInGraph_causesAnError() throws Exception {
    // Defines a package to avoid a BzlLoadFailedException (required to load PROJECT.scl files).
    write(
        "foo/BUILD",
        """
        package_group(
          name = "empty",
        )
        """);
    write(
        "foo/PROJECT.scl",
        """
        active_directories = {"default": ["foo"] }
        """);
    write("bar/BUILD", "");
    write(
        "bar/PROJECT.scl",
        """
        active_directories = {"default": ["bar"] }
        """);
    // Performs a trivial build to prime Bazel state.
    assertThat(buildTarget("//foo:empty").getSuccess()).isTrue();

    // Injects 2 ProjectValues into Skyframe.
    EvaluationResult<SkyValue> seedProjectValuesResult =
        getSkyframeExecutor()
            .evaluateSkyKeys(
                events.reporter(),
                ImmutableList.of(
                    new ProjectValue.Key(label("//foo:PROJECT.scl")),
                    new ProjectValue.Key(label("//bar:PROJECT.scl"))),
                /* keepGoing= */ false);
    assertThat(seedProjectValuesResult.hasError()).isFalse();

    DirectoryMatcherError error =
        (DirectoryMatcherError)
            FrontierSerializer.computeDirectoryMatcher(getSkyframeExecutor(), events.reporter());
    assertThat(error.message())
        .isEqualTo(
            "Dumping the frontier serialization profile does not support more than 1 project"
                + " file.");
  }

  @Test
  public void noTargets_returnsError() throws Exception {
    DirectoryMatcherError error =
        (DirectoryMatcherError)
            FrontierSerializer.computeDirectoryMatcher(getSkyframeExecutor(), events.reporter());
    assertThat(error.message()).isEqualTo("No top level targets could be determined.");
  }

  @Test
  public void noProjectFile_returnsError() throws Exception {
    write(
        "foo/BUILD",
        """
        package_group(
          name = "empty",
        )
        """);
    // Performs a build to populate top level targets.
    assertThat(buildTarget("//foo:empty").getSuccess()).isTrue();

    DirectoryMatcherError error =
        (DirectoryMatcherError)
            FrontierSerializer.computeDirectoryMatcher(getSkyframeExecutor(), events.reporter());
    assertThat(error.message())
        .isEqualTo("No project file could be determined from targets: //foo:empty");
  }

  @Test
  public void activeAspect_activatesBaseConfiguredTarget() throws Exception {
    setupScenarioWithAspects();
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

    ActiveDirectoryMatcher matcher =
        (ActiveDirectoryMatcher)
            FrontierSerializer.computeDirectoryMatcher(getSkyframeExecutor(), events.reporter());

    Map<ActionLookupKey, SelectionMarking> selection =
        FrontierSerializer.computeSelection(graph, matcher);

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
  public void dump_writesProfile() throws Exception {
    setupScenarioWithAspects();

    @SuppressWarnings("UnnecessarilyFullyQualified") // to avoid confusion with vfs Paths
    java.nio.file.Path profilePath = Files.createTempFile(null, "profile");

    Optional<BlazeCommandResult> result =
        FrontierSerializer.dumpFrontierSerializationProfile(
            new PrintStream(outErr.getOutputStream()),
            getCommandEnvironment(),
            profilePath.toString());
    assertThat(result).isEmpty(); // success

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

  private void setupScenarioWithAspects() throws Exception {
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
    assertThat(buildTarget("//bar:one").getSuccess()).isTrue();
  }

  private static void assertMatcherEqualsDefaultProjectMatcher(
      PathFragmentPrefixTrie matcher, ProjectValue projectValue) {
    assertThat(dumpStructureWithEquivalenceReduction(matcher))
        .isEqualTo(
            dumpStructureWithEquivalenceReduction(
                PathFragmentPrefixTrie.of(projectValue.getDefaultActiveDirectory())));
  }
}

