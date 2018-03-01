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

package com.google.devtools.build.lib.rules.proto;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.util.LabelArtifactOwner;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Unit tests for {@link ProtoCommon}.
 */
@RunWith(JUnit4.class)
public class ProtoCommonTest {

  private Scratch scratch;
  private Path execDir;
  private ArtifactRoot rootDir;

  @Before
  public final void setRootDir() throws Exception {
    scratch = new Scratch();
    execDir = scratch.dir("/exec");
    rootDir = ArtifactRoot.asDerivedRoot(execDir, scratch.dir("/exec/root"));
  }

  @Test
  public void getPathIgnoringRepository_main() throws IOException, LabelSyntaxException {
    Path f1 = scratch.file("/exec/root/foo/bar");

    PackageIdentifier ownerPackage =
        PackageIdentifier.create(RepositoryName.MAIN, PathFragment.create("//foo"));

    LabelArtifactOwner owner = new LabelArtifactOwner(Label.create(ownerPackage, "owner_a"));
    Artifact a1 = new Artifact(rootDir, f1.relativeTo(execDir), owner);
    PathFragment pathIgnoringRepository = ProtoCommon.getPathIgnoringRepository(a1);
    assertThat(pathIgnoringRepository).isEqualTo(PathFragment.create("foo/bar"));
  }

  @Test
  public void getPathIgnoringRepository_external() throws IOException, LabelSyntaxException {
    Path f1 = scratch.file("/exec/root/external/repo_a/foo/bar");

    PackageIdentifier ownerPackage =
        PackageIdentifier.create("@repo_a", PathFragment.create("//foo"));

    LabelArtifactOwner owner = new LabelArtifactOwner(Label.create(ownerPackage, "owner_a"));
    Artifact a1 = new Artifact(rootDir, f1.relativeTo(execDir), owner);
    PathFragment pathIgnoringRepository = ProtoCommon.getPathIgnoringRepository(a1);
    assertThat(pathIgnoringRepository).isEqualTo(PathFragment.create("foo/bar"));
  }
}
