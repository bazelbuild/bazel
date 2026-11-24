// Copyright 2026 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.java;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.vfs.Path;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link JavaTargetAttributes}. */
@RunWith(JUnit4.class)
public class JavaTargetAttributesTest extends FoundationTestCase {

  // test that direct jars always precede transitive dependencies on the compilation classpath
  @Test
  public void directJarOrderingTest() {
    Path execRoot = outputBase.getRelative("exec");
    String outSegment = "out";
    Path outputRoot = execRoot.getRelative(outSegment);
    ArtifactRoot root = ArtifactRoot.asDerivedRoot(execRoot, RootType.OUTPUT, outSegment);
    Artifact liba = ActionsTestUtil.createArtifact(root, outputRoot.getRelative("liba.jar"));
    Artifact libb = ActionsTestUtil.createArtifact(root, outputRoot.getRelative("libb.jar"));
    Artifact libc = ActionsTestUtil.createArtifact(root, outputRoot.getRelative("libc.jar"));
    Artifact libd = ActionsTestUtil.createArtifact(root, outputRoot.getRelative("libd.jar"));

    JavaTargetAttributes.Builder builder = new JavaTargetAttributes.Builder();

    builder.addCompileTimeClassPathEntries(
        NestedSetBuilder.create(Order.NAIVE_LINK_ORDER, liba, libb));
    builder.addDirectJars(NestedSetBuilder.create(Order.NAIVE_LINK_ORDER, liba));

    builder.addCompileTimeClassPathEntries(
        NestedSetBuilder.create(Order.NAIVE_LINK_ORDER, libc, libd));
    builder.addDirectJars(NestedSetBuilder.create(Order.NAIVE_LINK_ORDER, libc));

    JavaTargetAttributes attributes = builder.build();

    assertThat(attributes.getDirectJars().toSet()).containsExactly(liba, libc).inOrder();

    assertThat(attributes.getCompileTimeClassPath().toSet())
        .containsExactly(liba, libc, libb, libd)
        .inOrder();
  }
}
