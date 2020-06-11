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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Unit test for {@link J2ObjcSource}.
 */
@RunWith(JUnit4.class)
public class J2ObjcSourceTest {
  private ArtifactRoot rootDir;

  @Before
  public final void setRootDir() throws Exception  {
    Scratch scratch = new Scratch();
    Path execRoot = scratch.getFileSystem().getPath("/exec");
    String outSegment = "root";
    execRoot.getChild(outSegment).createDirectoryAndParents();
    rootDir = ArtifactRoot.asDerivedRoot(execRoot, outSegment);
  }

  @Test
  public void testEqualsAndHashCode() throws Exception {
    new EqualsTester()
        .addEqualityGroup(
            getJ2ObjcSource("//a/b:c", "sourceA", J2ObjcSource.SourceType.JAVA),
            getJ2ObjcSource("//a/b:c", "sourceA", J2ObjcSource.SourceType.JAVA))
        .addEqualityGroup(
            getJ2ObjcSource("//a/b:d", "sourceA", J2ObjcSource.SourceType.JAVA),
            getJ2ObjcSource("//a/b:d", "sourceA", J2ObjcSource.SourceType.JAVA))
        .addEqualityGroup(
            getJ2ObjcSource("//a/b:d", "sourceC", J2ObjcSource.SourceType.JAVA),
            getJ2ObjcSource("//a/b:d", "sourceC", J2ObjcSource.SourceType.JAVA))
        .addEqualityGroup(
            getJ2ObjcSource("//a/b:d", "sourceC", J2ObjcSource.SourceType.PROTO),
            getJ2ObjcSource("//a/b:d", "sourceC", J2ObjcSource.SourceType.PROTO))
        .testEquals();
  }

  private J2ObjcSource getJ2ObjcSource(String label, String fileName,
      J2ObjcSource.SourceType sourceType) throws Exception {
    Label ruleLabel = Label.parseAbsolute(label, ImmutableMap.of());
    PathFragment path = ruleLabel.toPathFragment();
    return new J2ObjcSource(
        ruleLabel,
        ImmutableList.of(getArtifactForTest(path.getRelative(fileName + ".m").toString())),
        ImmutableList.of(getArtifactForTest(path.getRelative(fileName + ".h").toString())),
        path,
        sourceType,
        ImmutableList.of(path));
  }

  private Artifact getArtifactForTest(String path) throws Exception {
    return ActionsTestUtil.createArtifact(rootDir, path);
  }
}
