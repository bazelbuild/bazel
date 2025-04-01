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
import static com.google.devtools.build.lib.cmdline.Label.parseCanonicalUnchecked;
import static java.util.stream.Collectors.joining;
import static org.junit.Assert.assertThrows;
import static org.junit.Assume.assumeTrue;

import com.google.common.base.Ascii;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.truth.Correspondence;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.actions.ActionLookupValue;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfiguredTargetValue;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.pkgcache.LoadingFailedException;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.WorkspaceBuilder;
import com.google.devtools.build.lib.runtime.commands.CqueryCommand;
import com.google.devtools.build.lib.runtime.commands.TestCommand;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.RemoteConfiguredTargetValue;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.serialization.FingerprintValueService;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.util.io.RecordingOutErr;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.errorprone.annotations.ForOverride;
import com.google.perftools.profiles.ProfileProto.Profile;
import com.google.protobuf.ExtensionRegistry;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Collection;
import java.util.HashMap;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;
import javax.annotation.Nullable;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TestName;

public abstract class FrontierSerializerTestBase extends BuildIntegrationTestCase {
  @Rule public TestName testName = new TestName();

  protected FingerprintValueService service;
  private final ClearCountingSyscallCache syscallCache = new ClearCountingSyscallCache();

  @Before
  public void setup() {
    // Give each test case a unique instance of the fingerprint value service, so that test cases
    // don't share state. This instance will then last the lifetime of the test case, regardless
    // of the number of command invocations.
    service = createFingerprintValueService();

    // TODO: b/367284400 - replace this with a barebones diffawareness check that works in Bazel
    // integration tests (e.g. making LocalDiffAwareness supported and not return
    // EVERYTHING_MODIFIED) for baseline diffs.
    addOptions("--experimental_frontier_violation_check=disabled_for_testing");
  }

  @ForOverride
  protected FingerprintValueService createFingerprintValueService() {
    return FingerprintValueService.createForTesting();
  }

  @Test
  public void serializingFrontierWithNoProjectFile_hasError() throws Exception {
    write(
        "foo/BUILD",
        """
        package_group(name = "empty")
        """);
    addOptions("--experimental_remote_analysis_cache_mode=upload");
    buildTarget("//foo:empty");
    events.assertContainsInfo("Disabling Skycache due to missing PROJECT.scl: [//foo:empty]");
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
project = {
  "active_directories": { "default": ["foo"] },
}
""");
    addOptions("--experimental_remote_analysis_cache_mode=upload");
    buildTarget("//foo:empty");
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
project = {
  "active_directories": { "default": ["foo"] },
}
""");

    write(
        "bar/BUILD",
        """
        package_group(name = "empty")
        """);
    write(
        "bar/PROJECT.scl",
"""
project = {
  "active_directories": { "default": ["bar"] },
}
""");

    addOptions("--experimental_remote_analysis_cache_mode=upload");
    LoadingFailedException exception =
        assertThrows(LoadingFailedException.class, () -> buildTarget("//foo:empty", "//bar:empty"));
    assertThat(exception)
        .hasMessageThat()
        .contains("Skycache only works on single-project builds. This is a multi-project build");
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
project = {
  "active_directories": { "default": ["foo"] },
}
""");

    addOptions("--experimental_remote_analysis_cache_mode=upload");
    buildTarget("//foo:empty", "//foo/bar:empty");
  }

  @Test
  public void buildCommandWithSkymeld_uploadsFrontierBytesWithUploadMode() throws Exception {
    runSkymeldScenario();
    // Validate that Skymeld did run.
    assertThat(getCommandEnvironment().withMergedAnalysisAndExecutionSourceOfTruth()).isTrue();

    var listener = getCommandEnvironment().getRemoteAnalysisCachingEventListener();
    assertThat(listener.getSerializedKeysCount()).isAtLeast(1);
    assertThat(listener.getSkyfunctionCounts().count(SkyFunctions.CONFIGURED_TARGET)).isAtLeast(1);

    assertContainsEvent("Waiting for write futures took an additional");
  }

  @Test
  public void buildCommandWithSkymeld_doesNotClearCacheMidBuild() throws Exception {
    runSkymeldScenario();

    assertThat(getSyscallCacheClearCount()).isEqualTo(2);
  }

  private void runSkymeldScenario() throws Exception {
    write(
        "foo/PROJECT.scl",
"""
project = {
  "active_directories": { "default": ["foo"] },
}
""");
    write(
        "foo/BUILD",
        """
        genrule(name = "g", srcs = ["//bar"], outs = ["g.out"], cmd = "cp $< $@")
        genrule(name = "h", srcs = ["//bar"], outs = ["h.out"], cmd = "cp $< $@")
        """);
    write(
        "bar/BUILD",
        """
        genrule(name = "bar", outs = ["out"], cmd = "touch $@")
        """);
    addOptions(
        "--experimental_remote_analysis_cache_mode=upload",
        "--build", // overrides --nobuild in setup step.
        "--experimental_merged_skyframe_analysis_execution" // forces Skymeld.
        );
    assertThat(buildTarget("//foo:all").getSuccess()).isTrue();
  }

  @Test
  public void buildCommand_serializedFrontierProfileContainsExpectedClasses() throws Exception {
    @SuppressWarnings("UnnecessarilyFullyQualified") // to avoid confusion with vfs Paths
    java.nio.file.Path profilePath = Files.createTempFile(null, "profile");

    addOptions("--serialized_frontier_profile=" + profilePath);
    setupScenarioWithAspects();
    upload("//bar:one");

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

  @Test
  public void roundTrip_withDifferentWorkspaces_translatesRoots() throws Exception {
    // Performs setup twice to simulate running in two different workspaces. The first setup is for
    // the writer, using the default roots provided by BuildIntegrationTestCase. The second setup
    // populates roots obtained upon recreating the files and mocks.
    setupScenarioWithAspects();
    upload("//bar:one"); // Cache writing build.

    // TODO: b/367287783 - RemoteConfiguredTargetValue cannot be deserialized successfully with
    // Bazel yet. Return early.
    assumeTrue(isBlaze());

    var barOneKey =
        ConfiguredTargetKey.builder()
            .setLabel(parseCanonicalUnchecked("//bar:one"))
            .setConfiguration(getTargetConfiguration())
            .build();

    PathFragment writerWorkspace = directories.getWorkspace().asFragment();

    var graph = getSkyframeExecutor().getEvaluator().getInMemoryGraph();
    var writerValue = graph.getIfPresent(barOneKey).getValue();

    assertThat(writerValue).isInstanceOf(ConfiguredTargetValue.class);
    Root writerRoot = getRootOfFirstInputOfFirstAction((ActionLookupValue) writerValue);
    // The root is the workspace.
    assertThat(writerRoot.asPath().asFragment()).isEqualTo(writerWorkspace);

    // Since createTestRoot is overridden to create unique directories, this creates distinct roots
    // for the reader, simulating a cross-workspace build.
    cleanUp();
    createFilesAndMocks();
    setupScenarioWithAspects();
    addOptions("--experimental_frontier_violation_check=disabled_for_testing");

    download("//bar:one"); // Cache reading build.

    PathFragment readerWorkspace = directories.getWorkspace().asFragment();
    // Sanity checks that recreating the directories results in a distinct workspace.
    assertThat(readerWorkspace).isNotEqualTo(writerWorkspace);

    graph = getSkyframeExecutor().getEvaluator().getInMemoryGraph();
    var readerValue = graph.getIfPresent(barOneKey).getValue();
    assertThat(readerValue).isInstanceOf(RemoteConfiguredTargetValue.class);
    Root readerRoot = getRootOfFirstInputOfFirstAction((ActionLookupValue) readerValue);
    // Asserts that the root was transformed to the reader's workspace.
    assertThat(readerRoot.asPath().asFragment()).isEqualTo(readerWorkspace);
  }

  @Override
  protected Path createTestRoot(FileSystem fileSystem) {
    try {
      return TestUtils.createUniqueTmpDir(fileSystem);
    } catch (IOException e) {
      throw new AssertionError(e);
    }
  }

  private static Root getRootOfFirstInputOfFirstAction(ActionLookupValue value) {
    return value.getAction(0).getInputs().toList().get(0).getRoot().getRoot();
  }

  protected final void setupScenarioWithAspects() throws Exception {
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
project = {
  "active_directories": { "default": ["foo"] },
}
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
load("@rules_java//java:defs.bzl", "java_library")
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

    addOptions("--nobuild", "--aspects=//foo:file_count.bzl%file_count_aspect");
  }

  @Test
  public void buildCommand_downloadsFrontierBytesWithDownloadMode() throws Exception {
    setupScenarioWithConfiguredTargets();
    write(
        "foo/PROJECT.scl",
"""
project = {
  "active_directories": { "default": ["foo"] },
}
""");
    // Cache-writing build.
    upload("//foo:A");

    // TODO: b/367287783 - RemoteConfiguredTargetValue cannot be deserialized successfully with
    // Bazel yet. Return early.
    assumeTrue(isBlaze());

    // Reset the graph.
    getCommandEnvironment().getSkyframeExecutor().resetEvaluator();

    // Cache reading build.
    download("//foo:A");

    var listener = getCommandEnvironment().getRemoteAnalysisCachingEventListener();
    // //bar:C, //bar:H, //bar:E
    assertThat(listener.getAnalysisNodeCacheHits()).isEqualTo(3);
    // //bar:F is not in the project boundary, but it's in the active set, so it wasn't cached.
    assertThat(listener.getAnalysisNodeCacheMisses()).isAtLeast(1);
  }

  @Test
  public void buildCommand_downloadedFrontierContainsRemoteConfiguredTargetValues()
      throws Exception {
    setupScenarioWithConfiguredTargets();
    write(
        "foo/PROJECT.scl",
"""
project = {
  "active_directories": { "default": ["foo"] },
}
""");

    // Cache-writing build.
    upload("//foo:A");

    // TODO: b/367287783 - RemoteConfiguredTargetValue cannot be deserialized successfully with
    // Bazel yet. Return early.
    assumeTrue(isBlaze());

    // Reset the graph.
    getCommandEnvironment().getSkyframeExecutor().resetEvaluator();

    // Cache reading build.
    download("//foo:A");

    var graph = getSkyframeExecutor().getEvaluator().getInMemoryGraph();
    Function<String, ConfiguredTargetKey> ctKeyOfLabel =
        (String label) ->
            ConfiguredTargetKey.builder()
                .setLabel(parseCanonicalUnchecked(label))
                .setConfiguration(getTargetConfiguration())
                .build();

    var expectedFrontier = ImmutableSet.of("//bar:C", "//bar:H", "//bar:E");
    expectedFrontier.forEach(
        label ->
            assertThat(graph.getIfPresent(ctKeyOfLabel.apply(label)).getValue())
                .isInstanceOf(RemoteConfiguredTargetValue.class));

    var expectedActiveSet =
        ImmutableSet.of(
            // //bar:F is in the active set because it's an rdep of //foo:G.
            "//bar:F", "//foo:G", "//foo:A", "//foo:B", "//foo:D");
    expectedActiveSet.forEach(
        label -> {
          var skyValue = graph.getIfPresent(ctKeyOfLabel.apply(label)).getValue();
          assertThat(skyValue).isInstanceOf(ConfiguredTargetValue.class);
          assertThat(skyValue).isNotInstanceOf(RemoteConfiguredTargetValue.class);
        });

    assertThat(graph.getIfPresent(ctKeyOfLabel.apply("//bar:I"))).isNull();
  }

  @Test
  public void undoneNodesFromIncrementalChanges_ignoredForSerialization() throws Exception {
    setupScenarioWithConfiguredTargets();

    write(
        "foo/PROJECT.scl",
"""
project = {
  "active_directories": { "default": ["foo"] },
}
""");

    upload("//foo:A");

    var serializedConfiguredTargetCount =
        getCommandEnvironment()
            .getRemoteAnalysisCachingEventListener()
            .getSkyfunctionCounts()
            .count(SkyFunctions.CONFIGURED_TARGET);

    // Make a small change to the //foo:A's dep graph by cutting the dep on //foo:D.
    // By changing this file, //foo:D will be invalidated as a transitive reverse dependency, but
    // because evaluating //foo:A no longer needs //foo:D's value, it will remain as an un-done
    // value. FrontierSerializer will try to mark //foo:D as active because it's in 'foo', but
    // realizes that it's not done, so it will be ignored.
    write(
        "foo/BUILD",
"""
filegroup(name = "A", srcs = [":B", "//bar:C"])      # unchanged.
filegroup(name = "B", srcs = ["//bar:E", "//bar:F"]) # changed: cut dep on D.
filegroup(name = "D", srcs = ["//bar:H"])            # unchanged.
filegroup(name = "G")                                # unchanged.
""");

    // This will pass only if FrontierSerializer only processes nodes that have finished evaluating.
    buildTarget("//foo:A");
    // There are 2 fewer configured targets. //bar:D is not serialized, as explained above. //bar:H
    // is also not serialized because it is only reached via //bar:D.
    assertThat(
            getCommandEnvironment().getRemoteAnalysisCachingEventListener().getSkyfunctionCounts())
        .hasCount(SkyFunctions.CONFIGURED_TARGET, serializedConfiguredTargetCount - 2);
  }

  @Test
  public void downloadedConfiguredTarget_doesNotDownloadTargetPackage() throws Exception {
    setupScenarioWithConfiguredTargets();
    write(
        "foo/PROJECT.scl",
"""
project = {
  "active_directories": { "default": ["foo"] },
}
""");
    // Cache-writing build.
    upload("//foo:D");

    var graph = getSkyframeExecutor().getEvaluator().getInMemoryGraph();
    var fooKey = PackageIdentifier.createUnchecked(/* repository= */ "", "foo");
    var barKey = PackageIdentifier.createUnchecked(/* repository= */ "", "bar");
    // Building for the first time necessarily loads both package foo and bar.
    assertThat(graph.getIfPresent(fooKey)).isNotNull();
    assertThat(graph.getIfPresent(barKey)).isNotNull();

    // TODO: b/367287783 - RemoteConfiguredTargetValue cannot be deserialized successfully with
    // Bazel yet. Return early.
    assumeTrue(isBlaze());

    // Reset the graph.
    getCommandEnvironment().getSkyframeExecutor().resetEvaluator();

    // Cache reading build.
    download("//foo:D");

    graph = getSkyframeExecutor().getEvaluator().getInMemoryGraph();
    assertThat(graph.getIfPresent(fooKey)).isNotNull();
    // Package bar is not required if //bar:H is downloaded.
    assertThat(graph.getIfPresent(barKey)).isNull();
  }

  @Test
  public void cquery_succeedsAndDoesNotTriggerUpload() throws Exception {
    setupScenarioWithConfiguredTargets();
    addOptions("--experimental_remote_analysis_cache_mode=upload");
    runtimeWrapper.newCommand(CqueryCommand.class);
    buildTarget("//foo:A"); // succeeds, even though there's no PROJECT.scl
    assertThat(
            getCommandEnvironment()
                .getRemoteAnalysisCachingEventListener()
                .getSerializedKeysCount())
        .isEqualTo(0);
  }

  @Test
  public void cquery_succeedsAndDoesNotTriggerUploadWithProjectScl() throws Exception {
    setupScenarioWithConfiguredTargets();
    write(
        "foo/PROJECT.scl",
"""
project = {
  "active_directories": { "default": ["foo"] },
}
""");
    addOptions("--experimental_remote_analysis_cache_mode=upload");
    runtimeWrapper.newCommand(CqueryCommand.class);
    buildTarget("//foo:A");
    assertThat(
            getCommandEnvironment()
                .getRemoteAnalysisCachingEventListener()
                .getSerializedKeysCount())
        .isEqualTo(0);
  }

  @Test
  public void testConfiguration_doesNotAffectSkyValueVersion() throws Exception {
    setupScenarioWithConfiguredTargets();
    write(
        "mytest/PROJECT.scl",
"""
project = {
  "active_directories": { "default": ["mytest"] },
}
""");
    write("mytest/mytest.sh", "exit 0").setExecutable(true);
    write(
        "mytest/BUILD",
        """
        load("//test_defs:foo_test.bzl", "foo_test")
        foo_test(
            name = "mytest",
            srcs = ["mytest.sh"],
            data = ["//foo:A"],
        )
        """);
    addOptions("--experimental_remote_analysis_cache_mode=upload", "--nobuild");

    buildTarget("//mytest");
    assertThat(
            getCommandEnvironment()
                .getRemoteAnalysisCachingEventListener()
                .getSerializedKeysCount())
        .isAtLeast(1);

    var versionFromBuild =
        getCommandEnvironment().getRemoteAnalysisCachingEventListener().getSkyValueVersion();

    getSkyframeExecutor().resetEvaluator();

    runtimeWrapper.newCommand(TestCommand.class);
    buildTarget("//mytest");
    assertThat(
            getCommandEnvironment()
                .getRemoteAnalysisCachingEventListener()
                .getSerializedKeysCount())
        .isAtLeast(1);

    var versionFromTest =
        getCommandEnvironment().getRemoteAnalysisCachingEventListener().getSkyValueVersion();

    // Assert that the top level config checksum subcomponent is equal.
    assertThat(versionFromBuild.getTopLevelConfigFingerprint())
        .isEqualTo(versionFromTest.getTopLevelConfigFingerprint());
    // Then assert that the whole thing is equal.
    assertThat(versionFromBuild).isEqualTo(versionFromTest);
  }

  @Test
  public void dumpUploadManifestOnlyMode_writesManifestToStdOut() throws Exception {
    setupScenarioWithConfiguredTargets();
    addOptions("--experimental_remote_analysis_cache_mode=dump_upload_manifest_only");
    write(
        "foo/PROJECT.scl",
"""
project = {
  "active_directories": { "default": ["foo"] },
}
""");
    RecordingOutErr outErr = new RecordingOutErr();
    this.outErr = outErr;

    buildTarget("//foo:A");

    // BuildConfigurationKey is omitted to avoid too much specificity.
    var expected =
"""
FRONTIER_CANDIDATE: CONFIGURED_TARGET:ConfiguredTargetKey{label=//bar:C,
FRONTIER_CANDIDATE: CONFIGURED_TARGET:ConfiguredTargetKey{label=//bar:E,
FRONTIER_CANDIDATE: CONFIGURED_TARGET:ConfiguredTargetKey{label=//bar:H,
ACTIVE: CONFIGURED_TARGET:ConfiguredTargetKey{label=//bar:F,
ACTIVE: CONFIGURED_TARGET:ConfiguredTargetKey{label=//foo:A,
ACTIVE: CONFIGURED_TARGET:ConfiguredTargetKey{label=//foo:A,
ACTIVE: CONFIGURED_TARGET:ConfiguredTargetKey{label=//foo:B,
ACTIVE: CONFIGURED_TARGET:ConfiguredTargetKey{label=//foo:D,
ACTIVE: CONFIGURED_TARGET:ConfiguredTargetKey{label=//foo:G,
"""
            .lines()
            .collect(toImmutableList());

    expected.forEach(line -> assertThat(outErr.outAsLatin1()).contains(line));

    // The additional line is from the additional --host_platforms analysis node, which has a
    // different label internally and externally.
    assertThat(outErr.outAsLatin1().lines()).hasSize(expected.size() + 1);

    // Nothing serialized
    assertThat(
            getCommandEnvironment()
                .getRemoteAnalysisCachingEventListener()
                .getSerializedKeysCount())
        .isEqualTo(0);
  }

  protected void setupScenarioWithConfiguredTargets() throws Exception {
    // ┌───────┐     ┌───────┐
    // │ bar:C │ ◀── │ foo:A │
    // └───────┘     └───────┘
    //                 │
    //                 │
    //                 ▼
    // ┌───────┐     ┌───────┐     ┌───────┐
    // │ bar:E │ ◀── │ foo:B │ ──▶ │ bar:F │
    // └───────┘     └───────┘     └───────┘
    //   │             │             │
    //   │             │             │
    //   ▼             ▼             ▼
    // ┌───────┐     ┌───────┐     ┌───────┐
    // │ bar:I │     │ foo:D │     │ foo:G │
    // └───────┘     └───────┘     └───────┘
    //                 │
    //                 │
    //                 ▼
    //               ┌───────┐
    //               │ bar:H │
    //               └───────┘
    write(
        "foo/BUILD",
"""
filegroup(name = "A", srcs = [":B", "//bar:C"])
filegroup(name = "B", srcs = [":D", "//bar:E", "//bar:F"])
filegroup(name = "D", srcs = ["//bar:H"])
filegroup(name = "G")
""");
    write(
        "bar/BUILD",
"""
filegroup(name = "C")
filegroup(name = "E", srcs = [":I"])
filegroup(name = "F", srcs = ["//foo:G"])
filegroup(name = "H")
filegroup(name = "I")
""");
  }

  protected static <T> ImmutableSet<T> filterKeys(Set<SkyKey> from, Class<? extends T> klass) {
    return from.stream().filter(klass::isInstance).map(klass::cast).collect(toImmutableSet());
  }

  protected static ImmutableSet<Label> getLabels(Set<ActionLookupKey> from) {
    return from.stream().map(ActionLookupKey::getLabel).collect(toImmutableSet());
  }

  protected static ImmutableSet<Label> getOwningLabels(Set<ActionLookupData> from) {
    return from.stream()
        .map(data -> data.getActionLookupKey().getLabel())
        .collect(toImmutableSet());
  }

  @Test
  public void actionLookupKey_underTheFrontier_areNotUploaded() throws Exception {
    setupGenruleGraph();
    upload("//A");
    var serializedKeys =
        getCommandEnvironment().getRemoteAnalysisCachingEventListener().getSerializedKeys();
    ImmutableSet<Label> labels = getLabels(filterKeys(serializedKeys, ActionLookupKey.class));

    // Active set
    assertThat(labels).contains(parseCanonicalUnchecked("//A"));

    // Frontier
    assertThat(labels)
        .containsAtLeast(
            parseCanonicalUnchecked("//C:C.txt"), // output file CT
            parseCanonicalUnchecked("//E"));

    // Under the frontier
    assertThat(labels).doesNotContain(parseCanonicalUnchecked("//C"));
    assertThat(labels).doesNotContain(parseCanonicalUnchecked("//D"));

    // Different top level target
    assertThat(labels).doesNotContain(parseCanonicalUnchecked("//B"));
  }

  @Test
  public void frontierSelectionSucceeds_forTopLevelGenruleConfiguredTargetWithUniqueName()
      throws Exception {
    setupGenruleGraph();
    write(
        "A/BUILD",
        """
        genrule(
            name = "copy_of_A", # renamed
            srcs = ["in.txt", "//C:C.txt", "//E"],
            outs = ["A"],
            cmd = "cat $(SRCS) > $@",
        )
        """);
    upload("//A");
    var serializedKeys =
        getCommandEnvironment().getRemoteAnalysisCachingEventListener().getSerializedKeys();
    ImmutableSet<Label> labels = getLabels(filterKeys(serializedKeys, ActionLookupKey.class));

    // Active set
    assertThat(labels)
        .containsAtLeast(
            parseCanonicalUnchecked("//A"),
            parseCanonicalUnchecked("//A:copy_of_A"),
            parseCanonicalUnchecked("//A:in.txt"));

    // Frontier
    assertThat(labels)
        .containsAtLeast(parseCanonicalUnchecked("//C:C.txt"), parseCanonicalUnchecked("//E"));

    // Under the frontier
    assertThat(labels).doesNotContain(parseCanonicalUnchecked("//C"));
    assertThat(labels).doesNotContain(parseCanonicalUnchecked("//D"));

    // Different top level target
    assertThat(labels).doesNotContain(parseCanonicalUnchecked("//B"));
  }

  @Test
  public void dumpUploadManifestOnlyMode_forTopLevelGenruleConfiguredTarget() throws Exception {
    setupGenruleGraph();
    write(
        "A/BUILD",
        """
        genrule(
            name = "copy_of_A",
            srcs = ["in.txt", "//C:C.txt", "//E"],
            outs = ["A"],
            cmd = "cat $(SRCS) > $@",
        )
        """);

    addOptions("--experimental_remote_analysis_cache_mode=dump_upload_manifest_only");

    RecordingOutErr outErr = new RecordingOutErr();
    this.outErr = outErr;

    buildTarget("//A");

    // Note that there are two //A:A - one each for target and exec configuration. The
    // BuildConfigurationKey is omitted because it's too specific, but we test for the
    // exact number of entries in the manifest later, so the two //A:A configured targets will be
    // counted correctly.
    var expectedActiveSet =
"""
ACTIVE: CONFIGURED_TARGET:ConfiguredTargetKey{label=//A:copy_of_A, config=
ACTIVE: CONFIGURED_TARGET:ConfiguredTargetKey{label=//A:A, config=
ACTIVE: CONFIGURED_TARGET:ConfiguredTargetKey{label=//A:A, config=
ACTIVE: CONFIGURED_TARGET:ConfiguredTargetKey{label=//A:in.txt, config=null}
ACTION_EXECUTION:ActionLookupData0{actionLookupKey=ConfiguredTargetKey{label=//A:copy_of_A, config=
"""
            .lines()
            .collect(toImmutableList());

    var actualActiveSet =
        outErr.outAsLatin1().lines().filter(l -> l.startsWith("ACTIVE:")).collect(joining("\n"));

    expectedActiveSet.forEach(line -> assertThat(actualActiveSet).contains(line));

    assertThat(actualActiveSet.lines()).hasSize(expectedActiveSet.size());
  }

  @Test
  public void actionLookupData_ownedByActiveSet_areUploaded() throws Exception {
    setupGenruleGraph();
    upload("//A");
    var serializedKeys =
        getCommandEnvironment().getRemoteAnalysisCachingEventListener().getSerializedKeys();
    var actionLookupDatas = filterKeys(serializedKeys, ActionLookupData.class);
    var owningLabels = getOwningLabels(actionLookupDatas);

    // Active set
    assertThat(owningLabels).contains(parseCanonicalUnchecked("//A"));

    // Frontier
    assertThat(owningLabels)
        .containsAtLeast(parseCanonicalUnchecked("//C"), parseCanonicalUnchecked("//E"));

    // Under the frontier
    assertThat(owningLabels).contains(parseCanonicalUnchecked("//D"));

    // Different top level target
    assertThat(owningLabels).doesNotContain(parseCanonicalUnchecked("//B"));
  }

  @Test
  public void disjointDirectoriesWithCanonicalProject_uploadsSuccessfully() throws Exception {
    setupGenruleGraph();
    write(
        "B/PROJECT.scl",
"""
project = { "actual": "//A:PROJECT.scl" }
""");
    upload("//A", "//B");
  }

  @Test
  public void activeToInactiveToActiveGraph_roundTrips() throws Exception {
    // In this test case, a depends on b depends on c. Crucially, a depends on c's output,
    // transitively. Furthermore, both a and c are in the active directories, while b is outside.
    //
    // This test guards against the serialization of b's ActionLookupValue. If b is serialized, when
    // a requests b, it'll get b from cache during analysis, without ever analyzing c (which can't
    // be fetched from the remote cache because it's in the active directories). Thus, at execution,
    // when a requests c's output, it'll trigger c's analysis, causing a Not-yet-present artifact
    // owner exception.
    //
    // TODO: b/364831651 - this test case remains salient with granular invalidation, but the
    // mechanism changes and the comment should be updated.
    write(
        "pkg_c/BUILD",
"""
genrule(
    name = "c",
    outs = ["c.out"],
    cmd = "echo 'C contents' > $@",
)
""");
    write(
        "pkg_b/BUILD",
"""
filegroup(
    name = "b",
    srcs = ["//pkg_c:c"], # Depends on c, collects its default outputs.
)
""");
    write(
        "pkg_a/BUILD",
"""
genrule(
    name = "a",
    srcs = ["//pkg_b:b"],
    outs = ["a.out"],
    # This command reads the content of the file(s) provided by target B. Since B wraps C's output,
    # this effectively reads C's output via B.
    cmd = "echo 'A received this via B:' > $@ && cat $(locations //pkg_b:b) >> $@",
)
""");
    write(
        "pkg_a/PROJECT.scl",
"""
project = { "active_directories": {"default": ["pkg_a", "pkg_c"]}}
""");
    upload("//pkg_a:a");

    // TODO: b/367287783 - RemoteConfiguredTargetValue cannot be deserialized successfully with
    // Bazel yet. Return early.
    assumeTrue(isBlaze());

    getSkyframeExecutor().resetEvaluator();
    download("//pkg_a:a");
  }

  protected final void setupGenruleGraph() throws IOException {
    // /--> E
    // A -> C -> D
    // B ---^
    write("A/in.txt", "A");
    write(
        "A/BUILD",
        """
        genrule(
            name = "A",
            srcs = ["in.txt", "//C:C.txt", "//E"],
            outs = ["A"],
            cmd = "cat $(SRCS) > $@",
        )
        """);
    write("B/in.txt", "B");
    write(
        "B/BUILD",
        """
        genrule(
            name = "B",
            srcs = ["in.txt", "//C:C.txt"],
            outs = ["B"],
            cmd = "cat $(SRCS) > $@",
        )
        """);
    write("C/in.txt", "C");
    write(
        "C/BUILD",
        """
        genrule(
            name = "C",
            srcs = ["in.txt", "//D:D.txt"],
            outs = ["C.txt"],
            cmd = "cat $(SRCS) > $@",
        )
        """);
    write("D/in.txt", "D");
    write(
        "D/BUILD",
        """
        genrule(
            name = "D",
            srcs = ["in.txt"],
            outs = ["D.txt"],
            cmd = "cat $(SRCS) > $@",
        )
        """);
    write("E/in.txt", "E");
    write(
        "E/BUILD",
        """
        genrule(
            name = "E",
            srcs = ["in.txt"],
            outs = ["E.txt"],
            cmd = "cat $(SRCS) > $@",
        )
        """);
    write(
        "A/PROJECT.scl",
"""
project = { "active_directories": {"default": ["A"]} }
""");
  }

  protected static final void assertContainsExactlyPrefixes(
      ImmutableList<String> strings, String... prefixes) {
    Correspondence<String, String> prefixCorrespondence =
        Correspondence.from(String::startsWith, "starts with");
    for (String prefix : prefixes) {
      assertThat(strings).comparingElementsUsing(prefixCorrespondence).contains(prefix);
    }
  }

  protected static final void assertContainsPrefixes(
      ImmutableList<String> strings, String... prefixes) {
    Correspondence<String, String> prefixCorrespondence =
        Correspondence.from(String::startsWith, "starts with");
    for (String prefix : prefixes) {
      assertThat(strings).comparingElementsUsing(prefixCorrespondence).contains(prefix);
    }
  }

  protected static final void assertDoesNotContainPrefixes(
      ImmutableList<String> strings, String... prefixes) {
    Correspondence<String, String> prefixCorrespondence =
        Correspondence.from(String::startsWith, "starts with");
    assertThat(strings).comparingElementsUsing(prefixCorrespondence).containsNoneIn(prefixes);
  }

  protected final ImmutableList<String> uploadManifest(String... targets) throws Exception {
    getSkyframeExecutor().resetEvaluator();
    addOptions("--experimental_remote_analysis_cache_mode=dump_upload_manifest_only");
    RecordingOutErr outErr = new RecordingOutErr();
    this.outErr = outErr;
    buildTarget(targets);
    return outErr
        .outAsLatin1()
        .lines()
        .filter(s -> s.startsWith("ACTIVE: ") || s.startsWith("FRONTIER_CANDIDATE:"))
        .collect(toImmutableList());
  }

  protected final void roundtrip(String... targets) throws Exception {
    getSkyframeExecutor().resetEvaluator();
    upload(targets);
    getSkyframeExecutor().resetEvaluator();
    download(targets);
  }

  protected final void upload(String... targets) throws Exception {
    addOptions("--experimental_remote_analysis_cache_mode=upload");
    buildTarget(targets);
    assertWithMessage("expected to serialize at least one Skyframe node")
        .that(getCommandEnvironment().getRemoteAnalysisCachingEventListener().getSerializedKeys())
        .isNotEmpty();
  }

  protected final void download(String... targets) throws Exception {
    addOptions("--experimental_remote_analysis_cache_mode=download");
    buildTarget(targets);
    assertWithMessage("expected to deserialize at least one Skyframe node")
        .that(getCommandEnvironment().getRemoteAnalysisCachingEventListener().getCacheHits())
        .isNotEmpty();
  }

  @Override
  protected BlazeRuntime.Builder getRuntimeBuilder() throws Exception {
    var builder = super.getRuntimeBuilder();
    if (testUsesSyscallCacheClearCount()) {
      // There isn't really a good way to apply this conditionally during @Before in Junit.
      builder.addBlazeModule(new SyscallCacheInjectingModule());
    }
    return builder;
  }

  boolean isBlaze() {
    return Ascii.equalsIgnoreCase(getCommandEnvironment().getRuntime().getProductName(), "blaze");
  }

  boolean testUsesSyscallCacheClearCount() {
    return testName.getMethodName().equals("buildCommandWithSkymeld_doesNotClearCacheMidBuild");
  }

  int getSyscallCacheClearCount() {
    return syscallCache.clearCount.get();
  }

  static class ClearCountingSyscallCache implements SyscallCache {
    private final AtomicInteger clearCount = new AtomicInteger(0);

    @Override
    public Collection<Dirent> readdir(Path path) throws IOException {
      return path.readdir(Symlinks.NOFOLLOW);
    }

    @Nullable
    @Override
    public FileStatus statIfFound(Path path, Symlinks symlinks) throws IOException {
      return path.statIfFound(symlinks);
    }

    @Override
    public DirentTypeWithSkip getType(Path path, Symlinks symlinks) {
      return DirentTypeWithSkip.FILESYSTEM_OP_SKIPPED;
    }

    @Override
    public void clear() {
      clearCount.incrementAndGet();
    }
  }

  class SyscallCacheInjectingModule extends BlazeModule {
    @Override
    public void workspaceInit(
        BlazeRuntime runtime, BlazeDirectories directories, WorkspaceBuilder builder) {
      builder.setSyscallCache(syscallCache);
    }
  }
}
