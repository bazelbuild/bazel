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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.truth.Correspondence;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.BuildView;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.pkgcache.LoadingFailedException;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.WorkspaceBuilder;
import com.google.devtools.build.lib.runtime.commands.CqueryCommand;
import com.google.devtools.build.lib.runtime.commands.TestCommand;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.io.RecordingOutErr;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.perftools.profiles.ProfileProto.Profile;
import com.google.protobuf.ExtensionRegistry;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Collection;
import java.util.HashMap;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import javax.annotation.Nullable;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TestName;

public abstract class SkycacheIntegrationTestBase extends BuildIntegrationTestCase
    implements SkycacheIntegrationTestHelpers {

  protected static final String UPLOAD_MODE_OPTION =
      "--experimental_remote_analysis_cache_mode=upload";
  protected static final String DOWNLOAD_MODE_OPTION =
      "--experimental_remote_analysis_cache_mode=download";
  protected static final String DUMP_MANIFEST_MODE_OPTION =
      "--experimental_remote_analysis_cache_mode=dump_upload_manifest_only";

  @Rule public TestName testName = new TestName();

  private final ClearCountingSyscallCache syscallCache = new ClearCountingSyscallCache();

  @Before
  public void setup() {
    // TODO: b/367284400 - replace this with a barebones diffawareness check that works in Bazel
    // integration tests (e.g. making LocalDiffAwareness supported and not return
    // EVERYTHING_MODIFIED) for baseline diffs.
    addOptions("--experimental_frontier_violation_check=disabled_for_testing");
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

  @Override
  protected Path createTestRoot(FileSystem fileSystem) {
    try {
      return TestUtils.createUniqueTmpDir(fileSystem);
    } catch (IOException e) {
      throw new AssertionError(e);
    }
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
  public void errorOnWarmSkyframeUploadBuilds() throws Exception {
    setupScenarioWithConfiguredTargets();

    writeProjectSclWithActiveDirs("foo");

    assertUploadSuccess("//foo:A");
    var exception = assertThrows(AbruptExitException.class, () -> buildTarget("//foo:A"));
    assertThat(exception).hasMessageThat().contains(BuildView.UPLOAD_BUILDS_MUST_BE_COLD);
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
        getCommandEnvironment()
            .getRemoteAnalysisCachingEventListener()
            .getRemoteAnalysisCachingState()
            .version();

    getSkyframeExecutor().resetEvaluator();

    runtimeWrapper.newCommand(TestCommand.class);
    buildTarget("//mytest");
    assertThat(
            getCommandEnvironment()
                .getRemoteAnalysisCachingEventListener()
                .getSerializedKeysCount())
        .isAtLeast(1);

    var versionFromTest =
        getCommandEnvironment()
            .getRemoteAnalysisCachingEventListener()
            .getRemoteAnalysisCachingState()
            .version();

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

  @Override
  public CommandEnvironment getCommandEnvironment() {
    return super.getCommandEnvironment();
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

  protected static ImmutableSet<String> getLabelStrings(Set<ActionLookupKey> from) {
    return getLabels(from).stream().map(Label::toString).collect(toImmutableSet());
  }

  protected static ImmutableSet<Label> getOwningLabels(Set<ActionLookupData> from) {
    return from.stream()
        .map(data -> data.getActionLookupKey().getLabel())
        .collect(toImmutableSet());
  }
}
