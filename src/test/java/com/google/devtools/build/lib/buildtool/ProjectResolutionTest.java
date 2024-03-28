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
package com.google.devtools.build.lib.buildtool;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.skyframe.PackageLookupFunction.PROJECT_FILE_NAME;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.pkgcache.LoadingFailedException;
import com.google.devtools.build.lib.pkgcache.PackageOptions;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.runtime.QuiescingExecutorsImpl;
import com.google.devtools.build.lib.skyframe.BazelSkyframeExecutorConstants;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.common.options.Options;
import java.util.Set;
import java.util.UUID;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests how Bazel finds the right {@link Project} for a build.
 *
 * <p>This is an integration test between {@link Project} and the build process. It specifically
 * tests how builds call {@link Project} and use the results meaningfully. For direct unit tests on
 * projects, use {@link ProjectTest}.
 */
@RunWith(JUnit4.class)
// TODO b/331316530: Temporarily removed to avoid build memory regressions. Re-enable as opt in.
@Ignore
public class ProjectResolutionTest extends BuildIntegrationTestCase {
  @Before
  public void setupSkyframePackageSemantics() {
    getSkyframeExecutor()
        .preparePackageLoading(
            new PathPackageLocator(
                getOutputBase(),
                ImmutableList.of(Root.fromPath(getWorkspace())),
                BazelSkyframeExecutorConstants.BUILD_FILES_BY_PRIORITY),
            Options.getDefaults(PackageOptions.class),
            Options.getDefaults(BuildLanguageOptions.class),
            UUID.randomUUID(),
            ImmutableMap.of(),
            QuiescingExecutorsImpl.forTesting(),
            new TimestampGranularityMonitor(null));
  }

  @Test
  public void buildWithNoProjectFiles() throws Exception {
    write("pkg/BUILD", "genrule(name='f', cmd = '', srcs=[], outs=['a.out'])");

    assertThat(
            BuildTool.getProjectFile(
                ImmutableList.of(Label.parseCanonical("//pkg:f")),
                getSkyframeExecutor(),
                events.reporter()))
        .isNull();
  }

  @Test
  public void buildWithOneProjectFile() throws Exception {
    write("pkg/BUILD", "genrule(name='f', cmd = '', srcs=[], outs=['a.out'])");
    write("pkg/" + PROJECT_FILE_NAME);

    assertThat(
            BuildTool.getProjectFile(
                ImmutableList.of(Label.parseCanonical("//pkg:f")),
                getSkyframeExecutor(),
                events.reporter()))
        .isEqualTo(PathFragment.create("pkg/" + PROJECT_FILE_NAME));
  }

  @Test
  public void buildWithTwoProjectFiles() throws Exception {
    write("foo/bar/BUILD", "genrule(name='f', cmd = '', srcs=[], outs=['a.out'])");
    write("foo/BUILD");
    write("foo/" + PROJECT_FILE_NAME);
    write("foo/bar/" + PROJECT_FILE_NAME);

    var thrown =
        assertThrows(
            LoadingFailedException.class,
            () ->
                BuildTool.getProjectFile(
                    ImmutableList.of(Label.parseCanonical("//foo/bar:f")),
                    getSkyframeExecutor(),
                    events.reporter()));
    assertThat(thrown)
        .hasMessageThat()
        .contains(
            String.format(
                "Multiple project files found: [foo/%s, foo/bar/%s]",
                PROJECT_FILE_NAME, PROJECT_FILE_NAME));
  }

  @Test
  public void twoTargetsSameProjectFile() throws Exception {
    write("foo/bar/BUILD", "genrule(name='child', cmd = '', srcs=[], outs=['c.out'])");
    write("foo/BUILD", "genrule(name='parent', cmd = '', srcs=[], outs=['p.out'])");
    write("foo/" + PROJECT_FILE_NAME);

    assertThat(
            BuildTool.getProjectFile(
                ImmutableList.of(
                    Label.parseCanonical("//foo:parent"), Label.parseCanonical("//foo/bar:child")),
                getSkyframeExecutor(),
                events.reporter()))
        .isEqualTo(PathFragment.create("foo/" + PROJECT_FILE_NAME));
  }

  @Test
  public void twoTargetsDifferentProjectFiles() throws Exception {
    write("foo/BUILD", "genrule(name='f', cmd = '', srcs=[], outs=['f.out'])");
    write("bar/BUILD", "genrule(name='g', cmd = '', srcs=[], outs=['g.out'])");
    write("foo/" + PROJECT_FILE_NAME);
    write("bar/" + PROJECT_FILE_NAME);

    var thrown =
        assertThrows(
            LoadingFailedException.class,
            () ->
                BuildTool.getProjectFile(
                    ImmutableList.of(
                        Label.parseCanonical("//foo:f"), Label.parseCanonical("//bar:g")),
                    getSkyframeExecutor(),
                    events.reporter()));
    assertThat(thrown)
        .hasMessageThat()
        .contains(
            String.format(
                "Targets have different project settings. "
                    + "For example:  [foo/%s]: //foo:f [bar/%s]: //bar:g",
                PROJECT_FILE_NAME, PROJECT_FILE_NAME));
  }

  @Test
  public void ignoredProjectFileInNonPackages() throws Exception {
    write("foo/bar/baz/" + PROJECT_FILE_NAME);
    write("foo/bar/BUILD", "genrule(name='f', cmd = '', srcs=[], outs=['f.out'])");
    write("foo/bar/" + PROJECT_FILE_NAME);
    write("foo/" + PROJECT_FILE_NAME);

    assertThat(
            BuildTool.getProjectFile(
                ImmutableList.of(Label.parseCanonical("//foo/bar:f")),
                getSkyframeExecutor(),
                events.reporter()))
        .isEqualTo(PathFragment.create("foo/bar/" + PROJECT_FILE_NAME));

    Set<RootedPath> projectRootedPaths = Sets.newConcurrentHashSet();
    getSkyframeExecutor()
        .getEvaluator()
        .getInMemoryGraph()
        .parallelForEach(
            k -> {
              if (k.getKey().argument() instanceof RootedPath rp) {
                if (rp.getRootRelativePath().getBaseName().equals(PROJECT_FILE_NAME)) {
                  projectRootedPaths.add(rp);
                }
              }
            });
    assertThat(projectRootedPaths).hasSize(1);
  }
}
