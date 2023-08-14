// Copyright 2021 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.view.java;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.rules.java.DeployArchiveBuilder.Compression.COMPRESSED;
import static com.google.devtools.build.lib.rules.java.DeployArchiveBuilder.Compression.UNCOMPRESSED;
import static com.google.devtools.build.lib.rules.java.DeployArchiveBuilder.defaultSingleJarCommandLine;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.CommandLine;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.rules.java.JavaConfiguration.OneVersionEnforcementLevel;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.testutil.MoreAsserts;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for the command line of the single jar action. */
@RunWith(JUnit4.class)
public class SingleJarCommandLineTest extends FoundationTestCase {
  @Test
  public void testIncludeBuildData() throws Exception {
    Artifact dummy =
        ActionsTestUtil.createArtifactWithExecPath(
            ArtifactRoot.asSourceRoot(Root.fromPath(scratch.dir("/a"))), PathFragment.create("b"));
    CommandLine command =
        defaultSingleJarCommandLine(
                dummy,
                Label.parseCanonicalUnchecked("//dummy"),
                null,
                ImmutableList.<String>of(),
                ImmutableList.<Artifact>of(),
                ImmutableList.<Artifact>of(),
                null,
                true,
                UNCOMPRESSED,
                null,
                OneVersionEnforcementLevel.OFF,
                null,
                /* multiReleaseDeployJars= */ false,
                /* javaHome= */ null,
                /* libModules= */ null,
                /* hermeticInputs= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER),
                /* addExports= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER),
                /* addOpens= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER))
            .build();

    assertThat(command.arguments()).doesNotContain("--exclude_build_data");
  }

  @Test
  public void testExcludeBuildData() throws Exception {
    Artifact dummy =
        ActionsTestUtil.createArtifactWithExecPath(
            ArtifactRoot.asSourceRoot(Root.fromPath(scratch.dir("/a"))), PathFragment.create("b"));
    CommandLine command =
        defaultSingleJarCommandLine(
                dummy,
                Label.parseCanonicalUnchecked("//dummy"),
                null,
                ImmutableList.<String>of(),
                ImmutableList.<Artifact>of(),
                ImmutableList.<Artifact>of(),
                null,
                false,
                UNCOMPRESSED,
                null,
                OneVersionEnforcementLevel.OFF,
                null,
                /* multiReleaseDeployJars= */ false,
                /* javaHome= */ null,
                /* libModules= */ null,
                /* hermeticInputs= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER),
                /* addExports= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER),
                /* addOpens= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER))
            .build();

    assertThat(command.arguments()).contains("--exclude_build_data");
  }

  @Test
  public void testIncludeJavaLauncher() throws Exception {
    Artifact dummy =
        ActionsTestUtil.createArtifactWithExecPath(
            ArtifactRoot.asSourceRoot(Root.fromPath(scratch.dir("/a"))), PathFragment.create("b"));
    CommandLine command =
        defaultSingleJarCommandLine(
                dummy,
                Label.parseCanonicalUnchecked("//dummy"),
                null,
                ImmutableList.<String>of(),
                ImmutableList.<Artifact>of(),
                ImmutableList.<Artifact>of(),
                null,
                false,
                UNCOMPRESSED,
                dummy,
                OneVersionEnforcementLevel.OFF,
                null,
                /* multiReleaseDeployJars= */ false,
                /* javaHome= */ null,
                /* libModules= */ null,
                /* hermeticInputs= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER),
                /* addExports= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER),
                /* addOpens= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER))
            .build();
    assertThat(command.arguments()).contains("--java_launcher");
  }

  @Test
  public void testIncludeCompression() throws Exception {
    Artifact dummy =
        ActionsTestUtil.createArtifactWithExecPath(
            ArtifactRoot.asSourceRoot(Root.fromPath(scratch.dir("/a"))), PathFragment.create("b"));
    CommandLine command =
        defaultSingleJarCommandLine(
                dummy,
                Label.parseCanonicalUnchecked("//dummy"),
                null,
                ImmutableList.<String>of(),
                ImmutableList.<Artifact>of(),
                ImmutableList.<Artifact>of(),
                null,
                false,
                COMPRESSED,
                dummy,
                OneVersionEnforcementLevel.OFF,
                null,
                /* multiReleaseDeployJars= */ false,
                /* javaHome= */ null,
                /* libModules= */ null,
                /* hermeticInputs= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER),
                /* addExports= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER),
                /* addOpens= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER))
            .build();
    assertThat(command.arguments()).contains("--compression");
  }

  @Test
  public void testExcludeCompression() throws Exception {
    Artifact dummy =
        ActionsTestUtil.createArtifactWithExecPath(
            ArtifactRoot.asSourceRoot(Root.fromPath(scratch.dir("/a"))), PathFragment.create("b"));
    CommandLine command =
        defaultSingleJarCommandLine(
                dummy,
                Label.parseCanonicalUnchecked("//dummy"),
                null,
                ImmutableList.<String>of(),
                ImmutableList.<Artifact>of(),
                ImmutableList.<Artifact>of(),
                null,
                false,
                UNCOMPRESSED,
                dummy,
                OneVersionEnforcementLevel.OFF,
                null,
                /* multiReleaseDeployJars= */ false,
                /* javaHome= */ null,
                /* libModules= */ null,
                /* hermeticInputs= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER),
                /* addExports= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER),
                /* addOpens= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER))
            .build();
    assertThat(command.arguments()).doesNotContain("--compression");
  }

  @Test
  public void testOneVersionArgs() throws Exception {
    Artifact dummy =
        ActionsTestUtil.createArtifactWithExecPath(
            ArtifactRoot.asSourceRoot(Root.fromPath(scratch.dir("/a"))), PathFragment.create("b"));
    Artifact dummyOneVersion =
        ActionsTestUtil.createArtifactWithExecPath(
            ArtifactRoot.asSourceRoot(Root.fromPath(scratch.dir("/a"))),
            PathFragment.create("whitelistfile"));
    CommandLine command =
        defaultSingleJarCommandLine(
                dummy,
                Label.parseCanonicalUnchecked("//dummy"),
                null,
                ImmutableList.<String>of(),
                ImmutableList.<Artifact>of(),
                ImmutableList.<Artifact>of(),
                null,
                false,
                UNCOMPRESSED,
                dummy,
                OneVersionEnforcementLevel.WARNING,
                dummyOneVersion,
                /* multiReleaseDeployJars= */ false,
                /* javaHome= */ null,
                /* libModules= */ null,
                /* hermeticInputs= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER),
                /* addExports= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER),
                /* addOpens= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER))
            .build();
    assertThat(command.arguments())
        .containsAtLeast("--enforce_one_version", "--succeed_on_found_violations");

    // --one_version_whitelist and the execpath to the whitelist are two different args, but they
    // need to be next to each other.
    MoreAsserts.assertContainsSublist(
        Lists.newArrayList(command.arguments()), "--one_version_whitelist", "whitelistfile");
  }
}
