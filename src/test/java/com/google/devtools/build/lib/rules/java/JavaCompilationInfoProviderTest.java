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

package com.google.devtools.build.lib.rules.java;

import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link JavaInfo}. */
@RunWith(JUnit4.class)
public class JavaCompilationInfoProviderTest {

  /**
   * Tests the {@link JavaCompilationInfoProvider} {@code equals} and {@code hashcode}
   * implementations.
   *
   * <p>The key thing we are testing is that the {@link JavaCompilationInfoProvider#bootClasspath}
   * matches for different {@link NestedSet} instances as long as they have the same content. Other
   * fields, such as {@link JavaCompilationInfoProvider#compilationClasspath} are required to be the
   * exact same {@link NestedSet} instance.
   */
  @Test
  public void compilationInfo_equalityTests() throws Exception {
    Artifact jar = createArtifact("foo.jar");
    NestedSet<Artifact> fixedNestedSet = NestedSetBuilder.create(Order.STABLE_ORDER, jar);
    JavaCompilationInfoProvider empty1 = new JavaCompilationInfoProvider.Builder().build();
    JavaCompilationInfoProvider empty2 = new JavaCompilationInfoProvider.Builder().build();
    JavaCompilationInfoProvider withBootCpNewNestedSet1 =
        new JavaCompilationInfoProvider.Builder()
            .setBootClasspath(
                BootClassPathInfo.create(NestedSetBuilder.create(Order.STABLE_ORDER, jar)))
            .build();
    JavaCompilationInfoProvider withBootCpNewNestedSet2 =
        new JavaCompilationInfoProvider.Builder()
            .setBootClasspath(
                BootClassPathInfo.create(NestedSetBuilder.create(Order.STABLE_ORDER, jar)))
            .build();
    JavaCompilationInfoProvider withBootCpNewEmptyNestedSet1 =
        new JavaCompilationInfoProvider.Builder()
            .setBootClasspath(
                BootClassPathInfo.create(NestedSetBuilder.emptySet(Order.STABLE_ORDER)))
            .build();
    JavaCompilationInfoProvider withBootCpNewEmptyNestedSet2 =
        new JavaCompilationInfoProvider.Builder()
            .setBootClasspath(
                BootClassPathInfo.create(NestedSetBuilder.emptySet(Order.STABLE_ORDER)))
            .build();
    JavaCompilationInfoProvider withCompileCpNewNestedSet1 =
        new JavaCompilationInfoProvider.Builder()
            .setCompilationClasspath(NestedSetBuilder.create(Order.STABLE_ORDER, jar))
            .build();
    JavaCompilationInfoProvider withCompileCpNewNestedSet2 =
        new JavaCompilationInfoProvider.Builder()
            .setCompilationClasspath(NestedSetBuilder.create(Order.STABLE_ORDER, jar))
            .build();
    JavaCompilationInfoProvider withCompileCpFixedNestedSet1 =
        new JavaCompilationInfoProvider.Builder().setCompilationClasspath(fixedNestedSet).build();
    JavaCompilationInfoProvider withCompileCpFixedNestedSet2 =
        new JavaCompilationInfoProvider.Builder().setCompilationClasspath(fixedNestedSet).build();

    new EqualsTester()
        .addEqualityGroup(empty1, empty2)
        .addEqualityGroup(withBootCpNewNestedSet1, withBootCpNewNestedSet2)
        .addEqualityGroup(withBootCpNewEmptyNestedSet1, withBootCpNewEmptyNestedSet2)
        .addEqualityGroup(withCompileCpNewNestedSet1)
        .addEqualityGroup(withCompileCpNewNestedSet2)
        .addEqualityGroup(withCompileCpFixedNestedSet1, withCompileCpFixedNestedSet2)
        .testEquals();
  }

  private static Artifact createArtifact(String path) throws IOException {
    Path execRoot = new Scratch().dir("/");
    ArtifactRoot root = ArtifactRoot.asDerivedRoot(execRoot, RootType.Output, "fake-root");
    return ActionsTestUtil.createArtifact(root, path);
  }
}
