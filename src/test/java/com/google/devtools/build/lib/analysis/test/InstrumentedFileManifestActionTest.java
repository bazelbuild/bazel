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
package com.google.devtools.build.lib.analysis.test;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.util.ActionTester;
import com.google.devtools.build.lib.analysis.util.ActionTester.ActionCombinationFactory;
import com.google.devtools.build.lib.analysis.util.AnalysisTestCase;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.vfs.Path;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for the instrumented file manifest creation
 */
@RunWith(JUnit4.class)
public class InstrumentedFileManifestActionTest extends AnalysisTestCase {

  @Before
  public final void initializeConfiguration() throws Exception {
    useConfiguration("--collect_code_coverage");
  }

  /** regression test for b/9607864. */
  @Test
  public void testInstrumentedFileManifestConflicts() throws Exception {
    scratch.file(
        "foo/BUILD",
        """
        java_library(
            name = "foo.so",
            srcs = ["Bar.java"],
        )

        java_library(
            name = "foo",
            srcs = ["Foo.java"],
        )
        """);

    update("//foo:foo", "//foo:foo.so");
  }

  private Artifact createArtifact(String rootRelativePath) {
    Path execRoot = outputBase.getRelative("exec");
    String rootSegment = "out";
    Path root = execRoot.getRelative(rootSegment);
    return ActionsTestUtil.createArtifact(
        ArtifactRoot.asDerivedRoot(execRoot, RootType.Output, rootSegment),
        root.getRelative(rootRelativePath));
  }

  private enum KeyAttributes {
    FILE_A,
    FILE_B
  }

  /** Regression test for b/28261106. */
  @Test
  public void testCacheKey() throws Exception {
    final Artifact a = createArtifact("foo/a");
    final Artifact b = createArtifact("foo/b");
    ActionTester.runTest(
        KeyAttributes.class,
        new ActionCombinationFactory<KeyAttributes>() {
          @Override
          public Action generate(ImmutableSet<KeyAttributes> attributesToFlip) {
            NestedSetBuilder<Artifact> files = NestedSetBuilder.stableOrder();
            if (attributesToFlip.contains(KeyAttributes.FILE_A)) {
              files.add(a);
            }
            if (attributesToFlip.contains(KeyAttributes.FILE_B)) {
              files.add(b);
            }
            Artifact output = createArtifact("foo/manifest");
            return new InstrumentedFileManifestAction(
                ActionOwner.SYSTEM_ACTION_OWNER, files.build(), output);
          }
        },
        actionKeyContext);
  }
}
