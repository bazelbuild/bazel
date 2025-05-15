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
import static java.util.Arrays.stream;
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
import com.google.devtools.build.lib.analysis.BuildView;
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
import com.google.devtools.build.lib.util.AbruptExitException;
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

  protected static final String UPLOAD_MODE_OPTION =
      "--experimental_remote_analysis_cache_mode=upload";
  protected static final String DOWNLOAD_MODE_OPTION =
      "--experimental_remote_analysis_cache_mode=download";
  protected static final String DUMP_MANIFEST_MODE_OPTION =
      "--experimental_remote_analysis_cache_mode=dump_upload_manifest_only";

  @Rule public TestName testName = new TestName();

  /**
   * A unique instance of the fingerprint value service per test case.
   *
   * <p>This ensures that test cases don't share state. The instance will then last the lifetime of
   * the test case, regardless of the number of command invocations.
   */
  protected FingerprintValueService service = createFingerprintValueService();

  private final ClearCountingSyscallCache syscallCache = new ClearCountingSyscallCache();

  @Before
  public void setup() {
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
  public void serializingFrontierWithNoProjectFile_hasNoError_withNothingSerialized()
      throws Exception {
    write(
        "foo/BUILD",
        """
        package_group(name = "empty")
        """);
    addOptions(UPLOAD_MODE_OPTION);
    buildTarget("//foo:empty");
    assertThat(
            getCommandEnvironment()
                .getRemoteAnalysisCachingEventListener()
                .getSerializedKeysCount())
        .isEqualTo(0);
  }

  @Test
  public void serializingFrontierWithNoProjectFile_withActiveDirectoriesFlag_serializesKeys()
      throws Exception {
    setupScenarioWithConfiguredTargets();

    addOptions("--experimental_active_directories=foo");
    addOptions(UPLOAD_MODE_OPTION);

    buildTarget("//foo:A");

    assertThat(
            getCommandEnvironment()
                .getRemoteAnalysisCachingEventListener()
                .getSerializedKeysCount())
        .isAtLeast(1);
  }

  @Test
  public void activeDirectoriesMatcher_activeDirectoriesFlag_takesPrecedenceOverProjectFile()
      throws Exception {
    // Tests that the result of the active directories matcher is the same regardless of whether
    // the matcher is obtained from the active directories flag or the PROJECT.scl file.
    setupScenarioWithConfiguredTargets();

    writeProjectSclWithActiveDirs("foo");
    addOptions(UPLOAD_MODE_OPTION);
    buildTarget("//foo:A");
    var serializedKeysWithProjectScl =
        getCommandEnvironment().getRemoteAnalysisCachingEventListener().getSerializedKeys();
    assertThat(serializedKeysWithProjectScl).isNotEmpty();

    getSkyframeExecutor().resetEvaluator();

    addOptions("--experimental_active_directories=bar"); // overrides the PROJECT.scl file.
    buildTarget("//foo:A");
    var serializedKeysWithActiveDirectories =
        getCommandEnvironment().getRemoteAnalysisCachingEventListener().getSerializedKeys();
    assertThat(serializedKeysWithActiveDirectories).isNotEmpty();

    assertThat(serializedKeysWithActiveDirectories).isNotEqualTo(serializedKeysWithProjectScl);
    assertContainsEvent(
        "Specifying --experimental_active_directories will override the active directories"
            + " specified in the PROJECT.scl file");
  }

  @Test
  public void activeDirectoriesMatcher_withProjectSclOrActiveDirectories_areEquivalent()
      throws Exception {
    // Tests that the result of the active directories matcher is the same regardless of whether
    // the matcher is obtained from the active directories flag or the PROJECT.scl file.
    setupScenarioWithConfiguredTargets();

    writeProjectSclWithActiveDirs(
        /* path= */ "foo",
        /* activeDirs...= */
        "foo",
        "-bar",
        "baz/qux",
        "-baz/qux/quux",
        "-zee",
        "zee/yee");
    addOptions(UPLOAD_MODE_OPTION);
    buildTarget("//foo:A");
    assertThat(
            getCommandEnvironment()
                .getRemoteAnalysisCachingEventListener()
                .getSerializedKeysCount())
        .isAtLeast(1);

    RemoteAnalysisCachingDependenciesProvider providerWithProjectScl =
        getSkyframeExecutor().getRemoteAnalysisCachingDependenciesProvider();

    getSkyframeExecutor().resetEvaluator();

    addOptions("--experimental_active_directories=foo,-bar,baz/qux,-baz/qux/quux,-zee,zee/yee");
    buildTarget("//foo:A");
    assertThat(
            getCommandEnvironment()
                .getRemoteAnalysisCachingEventListener()
                .getSerializedKeysCount())
        .isAtLeast(1);

    RemoteAnalysisCachingDependenciesProvider providerWithActiveDirectories =
        getSkyframeExecutor().getRemoteAnalysisCachingDependenciesProvider();

    assertThat(providerWithActiveDirectories).isNotSameInstanceAs(providerWithProjectScl);

    ImmutableList<PackageIdentifier> testCases =
        ImmutableList.of(
            PackageIdentifier.createInMainRepo("foo"),
            PackageIdentifier.createInMainRepo("foo/bar"),
            PackageIdentifier.createInMainRepo("bar"),
            PackageIdentifier.createInMainRepo("baz/qux"),
            PackageIdentifier.createInMainRepo("baz/qux/quux"),
            PackageIdentifier.createInMainRepo("zee"),
            PackageIdentifier.createInMainRepo("zee/yee"),
            PackageIdentifier.createInMainRepo(""),
            PackageIdentifier.createInMainRepo("nonexistent"));

    for (PackageIdentifier testCase : testCases) {
      var activeDirectoriesResult = providerWithActiveDirectories.withinActiveDirectories(testCase);
      var projectSclResult = providerWithProjectScl.withinActiveDirectories(testCase);
      assertWithMessage(
              "for %s: active directories: %s, projectScl: %s",
              testCase, activeDirectoriesResult, projectSclResult)
          .that(activeDirectoriesResult)
          .isEqualTo(projectSclResult);
    }
  }

  @Test
  public void serializingFrontierWithProjectFile_hasNoError() throws Exception {
    write(
        "foo/BUILD",
        """
        package_group(name = "empty")
        """);
    writeProjectSclWithActiveDirs("foo");

    addOptions(UPLOAD_MODE_OPTION);
    buildTarget("//foo:empty");
  }

  @Test
  public void serializingWithMultipleTopLevelProjectFiles_hasError() throws Exception {
    write(
        "foo/BUILD",
        """
        package_group(name = "empty")
        """);
    writeProjectSclWithActiveDirs("foo");

    write(
        "bar/BUILD",
        """
        package_group(name = "empty")
        """);
    writeProjectSclWithActiveDirs("bar");

    addOptions(UPLOAD_MODE_OPTION);
    LoadingFailedException exception =
        assertThrows(LoadingFailedException.class, () -> buildTarget("//foo:empty", "//bar:empty"));
    assertThat(exception).hasMessageThat().contains("This is a multi-project build");
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
    writeProjectSclWithActiveDirs("foo");

    addOptions(UPLOAD_MODE_OPTION);
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
    writeProjectSclWithActiveDirs("foo");
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
        UPLOAD_MODE_OPTION,
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
    assertUploadSuccess("//bar:one");

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
    assertUploadSuccess("//bar:one"); // Cache writing build.

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

    assertDownloadSuccess("//bar:one"); // Cache reading build.

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

    writeProjectSclWithActiveDirs("bar", "foo");

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
    writeProjectSclWithActiveDirs("foo");
    // Cache-writing build.
    assertUploadSuccess("//foo:A");

    // TODO: b/367287783 - RemoteConfiguredTargetValue cannot be deserialized successfully with
    // Bazel yet. Return early.
    assumeTrue(isBlaze());

    // Reset the graph.
    getCommandEnvironment().getSkyframeExecutor().resetEvaluator();

    // Cache reading build.
    assertDownloadSuccess("//foo:A");

    var listener = getCommandEnvironment().getRemoteAnalysisCachingEventListener();
    // //bar:C, //bar:H, //bar:E
    assertThat(listener.getCacheHits()).hasSize(3);
    // //bar:F is not in the project boundary, but it's in the active set, so it wasn't cached.
    assertThat(listener.getCacheMisses()).isNotEmpty();
  }

  @Test
  public void buildCommand_downloadedFrontierContainsRemoteConfiguredTargetValues()
      throws Exception {
    setupScenarioWithConfiguredTargets();
    writeProjectSclWithActiveDirs("foo");

    // Cache-writing build.
    assertUploadSuccess("//foo:A");

    // TODO: b/367287783 - RemoteConfiguredTargetValue cannot be deserialized successfully with
    // Bazel yet. Return early.
    assumeTrue(isBlaze());

    // Reset the graph.
    getCommandEnvironment().getSkyframeExecutor().resetEvaluator();

    // Cache reading build.
    assertDownloadSuccess("//foo:A");

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
  public void errorOnWarmSkyframeUploadBuilds() throws Exception {
    setupScenarioWithConfiguredTargets();

    writeProjectSclWithActiveDirs("foo");

    assertUploadSuccess("//foo:A");
    var exception = assertThrows(AbruptExitException.class, () -> buildTarget("//foo:A"));
    assertThat(exception).hasMessageThat().contains(BuildView.UPLOAD_BUILDS_MUST_BE_COLD);
  }

  @Test
  public void downloadedConfiguredTarget_doesNotDownloadTargetPackage() throws Exception {
    setupScenarioWithConfiguredTargets();
    writeProjectSclWithActiveDirs("foo");
    // Cache-writing build.
    assertUploadSuccess("//foo:D");

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
    assertDownloadSuccess("//foo:D");

    graph = getSkyframeExecutor().getEvaluator().getInMemoryGraph();
    assertThat(graph.getIfPresent(fooKey)).isNotNull();
    // Package bar is not required if //bar:H is downloaded.
    assertThat(graph.getIfPresent(barKey)).isNull();
  }

  @Test
  public void cquery_succeedsAndDoesNotTriggerUpload() throws Exception {
    setupScenarioWithConfiguredTargets();
    addOptions(UPLOAD_MODE_OPTION);
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
    writeProjectSclWithActiveDirs("foo");
    addOptions(UPLOAD_MODE_OPTION);
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
    writeProjectSclWithActiveDirs("mytest");
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
    addOptions(UPLOAD_MODE_OPTION, "--nobuild");

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
    addOptions(DUMP_MANIFEST_MODE_OPTION);
    writeProjectSclWithActiveDirs("foo");

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

  @Test
  public void actionLookupKey_underTheFrontier_areNotUploaded() throws Exception {
    setupGenruleGraph();
    assertUploadSuccess("//A");
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
    assertUploadSuccess("//A");
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

    addOptions(DUMP_MANIFEST_MODE_OPTION);

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
    assertUploadSuccess("//A");
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
    write("B/PROJECT.scl", "project = { \"actual\": \"//A:PROJECT.scl\" }");
    assertUploadSuccess("//A", "//B");
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
    writeProjectSclWithActiveDirs("pkg_a", "pkg_a", "pkg_c");
    assertUploadSuccess("//pkg_a:a");

    // TODO: b/367287783 - RemoteConfiguredTargetValue cannot be deserialized successfully with
    // Bazel yet. Return early.
    assumeTrue(isBlaze());

    getSkyframeExecutor().resetEvaluator();
    assertDownloadSuccess("//pkg_a:a");
  }

  @Test
  public void roundTripWithActionOwnerCacheHitAndActionCacheMiss_succeeds() throws Exception {
    // This test case sets up a scenario where there's an uncached action under the frontier, but
    // there's a cached analysis value in the frontier. This is possible when an underlying target,
    // in this case //A:output_generator has two Actions and only one of them is consumed by the
    // writer.
    //
    // If a reader builds a different top level target that requires the action that is not
    // consumed, it causes analysis to occur during the execution phase, requiring certain normally
    // applicable checks to be bypassed.
    write(
        "A/defs.bzl",
"""
MultiOutputInfo = provider(
    fields = {
        "output1": "The file generated by action 1.",
        "output2": "The file generated by action 2.",
    },
)

def _multi_output_impl(ctx):
    output_file1 = ctx.actions.declare_file(ctx.label.name + "_out1.txt")
    output_file2 = ctx.actions.declare_file(ctx.label.name + "_out2.txt")

    ctx.actions.run_shell(
        outputs = [output_file1],
        command = "echo Action1 > {output1}".format(output1 = output_file1.path),
        progress_message = "Running Action 1 for %{label}",
    )

    ctx.actions.run_shell(
        outputs = [output_file2],
        command = "echo Action2 > {output2}".format(output2 = output_file2.path),
        progress_message = "Running Action 2 for %{label}",
    )

    return [
        DefaultInfo(files = depset([output_file1, output_file2])),
        MultiOutputInfo(
            output1 = output_file1,
            output2 = output_file2,
        ),
    ]

multi_output_rule = rule(
    implementation = _multi_output_impl,
    doc = "A rule that runs two separate actions producing two distinct outputs.",
)

def _forwarding_impl(ctx):
    # Rule that forwards MultiOutputInfo.

    # Get the provider instance from the dependency
    dep_multi_output_info = ctx.attr.dep[MultiOutputInfo]

    # Simply return the provider instance received from the dependency.
    # We can also forward DefaultInfo if needed, but for this specific case,
    # just forwarding the required provider is enough.
    return [dep_multi_output_info]

forwarding_rule = rule(
    implementation = _forwarding_impl,
    attrs = {
        "dep": attr.label(
            providers = [MultiOutputInfo], # Ensure the dependency *has* the provider to forward
            mandatory = True,
            doc = "The target whose MultiOutputInfo should be forwarded.",
        ),
    },
    doc = "A rule that forwards the MultiOutputInfo provider from its dependency.",
)

def _consumer_impl(ctx):
    dep_info = ctx.attr.dep[MultiOutputInfo]
    input_file = None
    if ctx.attr.output_key == "output1":
        input_file = dep_info.output1
    elif ctx.attr.output_key == "output2":
        input_file = dep_info.output2
    else:
        # This case should be prevented by the 'values' check in the rule definition
        fail("Invalid output_key: '{}'. Must be 'output1' or 'output2'".format(ctx.attr.output_key))

    output_file = ctx.actions.declare_file(ctx.label.name + ".txt")

    # Added label to progress message for clarity
    ctx.actions.run_shell(
        inputs = [input_file],
        outputs = [output_file],
        command = "cat {input} > {output}".format(
            input = input_file.path,
            output = output_file.path,
        ),
        progress_message = "Running consumer %{{label}} using {} from {}".format(
            ctx.attr.output_key, ctx.attr.dep.label),
    )

    return [DefaultInfo(files = depset([output_file]))]

consumer_rule = rule(
    implementation = _consumer_impl,
    attrs = {
        "dep": attr.label(
            providers = [MultiOutputInfo],
            mandatory = True,
            doc = "The target providing the MultiOutputInfo (directly or indirectly).",
        ),
        "output_key": attr.string(
            values = ["output1", "output2"],
            mandatory = True,
            doc = "Which output to consume ('output1' or 'output2').",
        ),
    },
    doc = "Rule that consumes a specific output advertised via MultiOutputInfo.",
)
""");
    write(
        "A/BUILD",
"""
load("//A:defs.bzl", "multi_output_rule")

# Original source of the two outputs.
multi_output_rule(
    name = "output_generator",
)
""");

    write(
        "B/BUILD",
"""
load("//A:defs.bzl", "forwarding_rule")

# Depends on output_generator and forwards its MultiOutputInfo.
forwarding_rule(
    name = "forwarder",
    dep = "//A:output_generator",
)
""");

    write(
        "C/BUILD",
"""
load("//A:defs.bzl", "consumer_rule")

consumer_rule(
    name = "consumer_one",
    dep = "//B:forwarder",
    output_key = "output1",
)

consumer_rule(
    name = "consumer_two",
    dep = "//B:forwarder",
    output_key = "output2",
)
""");

    writeProjectSclWithActiveDirs("C", "C");
    assertUploadSuccess("//C:consumer_one");

    // TODO: b/367287783 - RemoteConfiguredTargetValue cannot be deserialized successfully with
    // Bazel yet. Return early.
    assumeTrue(isBlaze());

    getSkyframeExecutor().resetEvaluator();
    assertDownloadSuccess("//C:consumer_two");
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
    writeProjectSclWithActiveDirs("A");
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
    addOptions(DUMP_MANIFEST_MODE_OPTION);
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
    assertUploadSuccess(targets);
    getSkyframeExecutor().resetEvaluator();
    assertDownloadSuccess(targets);
  }

  protected final void assertUploadSuccess(String... targets) throws Exception {
    addOptions(UPLOAD_MODE_OPTION);
    buildTarget(targets);
    assertWithMessage("expected to serialize at least one Skyframe node")
        .that(getCommandEnvironment().getRemoteAnalysisCachingEventListener().getSerializedKeys())
        .isNotEmpty();
  }

  protected final void assertDownloadSuccess(String... targets) throws Exception {
    addOptions(DOWNLOAD_MODE_OPTION);
    buildTarget(targets);
    assertWithMessage("expected to deserialize at least one Skyframe node")
        .that(getCommandEnvironment().getRemoteAnalysisCachingEventListener().getCacheHits())
        .isNotEmpty();
  }

  protected final void writeProjectSclWithActiveDirs(String path, String... activeDirs)
      throws IOException {
    String activeDirsString = stream(activeDirs).map(s -> "\"" + s + "\"").collect(joining(", "));
    write(
        path + "/PROJECT.scl",
        String.format(
            "project = { \"active_directories\": { \"default\": [%s] } }", activeDirsString));
  }

  protected final void writeProjectSclWithActiveDirs(String path) throws IOException {
    // Overload for the common case where the path is the only active directory.
    writeProjectSclWithActiveDirs(path, path);
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
}
