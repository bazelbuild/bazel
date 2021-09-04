// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.cpp;

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.ExecutionRequirements;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.clock.JavaClock;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.ArgumentMatchers;
import org.mockito.Mockito;

/** Tests for {@link CppCompileAction#createSpawn}. */
@RunWith(JUnit4.class)
public final class SpawnGccStrategyTest {
  private FileSystem fs;
  private ArtifactRoot ar;
  private Path execRoot;

  @Before
  public void setup() {
    fs = new InMemoryFileSystem(new JavaClock(), DigestHashFunction.SHA256);
    execRoot = fs.getPath("/exec/root");
    ar = ArtifactRoot.asDerivedRoot(execRoot, RootType.Output, "out");
  }

  @Test
  public void testInMemoryDotdFileAndExecutionRequirement() throws Exception {
    // Test that when in memory dotd files are enabled the execution requirement to inline the dotd
    // file is added to the spawn.

    // arrange
    Artifact dotdFile = ActionsTestUtil.createArtifact(ar, ar.getRoot().asPath().getChild("dot.d"));
    CppCompileAction action = Mockito.mock(CppCompileAction.class);
    when(action.getOutputs()).thenReturn(ImmutableSet.of());
    when(action.getMandatoryInputs()).thenReturn(NestedSetBuilder.emptySet(Order.STABLE_ORDER));
    when(action.getAdditionalInputs()).thenReturn(NestedSetBuilder.emptySet(Order.STABLE_ORDER));
    when(action.getExecutionInfo()).thenReturn(ImmutableMap.of());
    when(action.getArguments()).thenReturn(ImmutableList.of());
    when(action.getEffectiveEnvironment(ArgumentMatchers.any())).thenReturn(ImmutableMap.of());
    when(action.getDotdFile()).thenReturn(dotdFile);
    when(action.useInMemoryDotdFiles()).thenReturn(true);
    when(action.shouldParseShowIncludes()).thenReturn(false);
    when(action.createSpawn(any(), any())).thenCallRealMethod();

    // act
    Spawn spawn = action.createSpawn(execRoot, ImmutableMap.of());

    ImmutableMap<String, String> execInfo = spawn.getExecutionInfo();
    assertThat(execInfo.get(ExecutionRequirements.REMOTE_EXECUTION_INLINE_OUTPUTS))
        .isEqualTo(action.getDotdFile().getExecPathString());
  }
}
