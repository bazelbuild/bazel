// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.objc;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.PathFragment;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** A test for {@link ObjcCommon}. */
@RunWith(JUnit4.class)
public class ObjcCommonTest extends BuildViewTestCase {

  @Test
  public void testObjcLibraryExternalStructuredResourcesNoErrors() throws Exception {
    FileSystemUtils.appendIsoLatin1(
        scratch.resolve("WORKSPACE"),
        "local_repository(",
        "    name = 'pkg',",
        "    path = '/foo')");
    getSkyframeExecutor()
        .invalidateFilesUnderPathForTesting(
            reporter,
            new ModifiedFileSet.Builder().modify(PathFragment.create("WORKSPACE")).build(),
            rootDirectory);
    FileSystemUtils.createDirectoryAndParents(scratch.resolve("/foo/bar/nested"));
    scratch.file("/foo/bar/nested/file.txt");
    scratch.file("/foo/WORKSPACE", "workspace(name = 'pkg')");
    scratch.file(
        "/foo/bar/BUILD",
        "objc_library(name = 'lib',",
        "             srcs = ['foo.cc'],",
        "             structured_resources = ['nested/file.txt'])");
    Label label = Label.parseAbsolute("@pkg//bar:lib");
    ConfiguredTarget target = view.getConfiguredTargetForTesting(reporter, label, targetConfig);
    Artifact artifact = getSharedArtifact("external/pkg/bar/nested/file.txt", target);

    // Verify that the xcodeStructuredResourceDirs call doesn't throw an exception and that the
    // PathFragment it returns has the expected suffix, which includes the repository directory.
    Iterable<Artifact> artifacts = ImmutableList.of(artifact);
    Iterable<PathFragment> pathFragments = ObjcCommon.xcodeStructuredResourceDirs(artifacts);

    assertThat(pathFragments.iterator().hasNext()).isTrue();
    PathFragment fragment = pathFragments.iterator().next();
    assertThat(fragment.endsWith(PathFragment.create("external/pkg/bar/nested"))).isTrue();
  }
}
