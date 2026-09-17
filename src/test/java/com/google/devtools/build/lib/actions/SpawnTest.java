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
package com.google.devtools.build.lib.actions;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.exec.util.SpawnBuilder;
import com.google.devtools.build.lib.vfs.PathFragment;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link Spawn}. */
@RunWith(JUnit4.class)
public final class SpawnTest {

  @Test
  public void getArgumentsWithExpandedParamFiles_noParamFiles() throws Exception {
    Spawn spawn = new SpawnBuilder("/bin/gcc", "-c", "foo.c", "-o", "foo.o").build();
    assertThat(spawn.getArgumentsWithExpandedParamFiles())
        .containsExactly("/bin/gcc", "-c", "foo.c", "-o", "foo.o")
        .inOrder();
  }

  @Test
  public void getArgumentsWithExpandedParamFiles_expandsParamFileReference() throws Exception {
    ParamFileActionInput paramFile =
        new ParamFileActionInput(
            PathFragment.create("output/foo-0.params"),
            "@output/foo-0.params",
            ImmutableList.of("-lfoo", "-lbar", "-lbaz"),
            ParameterFileType.UNQUOTED);
    Spawn spawn =
        new SpawnBuilder("/bin/gcc", "@output/foo-0.params", "-o", "foo.o")
            .withInput(paramFile)
            .build();
    assertThat(spawn.getArgumentsWithExpandedParamFiles())
        .containsExactly("/bin/gcc", "-lfoo", "-lbar", "-lbaz", "-o", "foo.o")
        .inOrder();
  }

  @Test
  public void getArgumentsWithExpandedParamFiles_expandsCustomParamFileReference()
      throws Exception {
    ParamFileActionInput paramFile =
        new ParamFileActionInput(
            PathFragment.create("output/foo-0.params"),
            "--param=output/foo-0.params",
            ImmutableList.of("-lfoo", "-lbar", "-lbaz"),
            ParameterFileType.UNQUOTED);
    Spawn spawn =
        new SpawnBuilder("/bin/gcc", "--param=output/foo-0.params", "-o", "foo.o")
            .withInput(paramFile)
            .build();
    assertThat(spawn.getArgumentsWithExpandedParamFiles())
        .containsExactly("/bin/gcc", "-lfoo", "-lbar", "-lbaz", "-o", "foo.o")
        .inOrder();
  }

  @Test
  public void getArgumentsWithExpandedParamFiles_multipleParamFiles() throws Exception {
    ParamFileActionInput paramFile1 =
        new ParamFileActionInput(
            PathFragment.create("output/foo-0.params"),
            "@output/foo-0.params",
            ImmutableList.of("-lfoo", "-lbar"),
            ParameterFileType.UNQUOTED);
    ParamFileActionInput paramFile2 =
        new ParamFileActionInput(
            PathFragment.create("output/foo-1.params"),
            "@output/foo-1.params",
            ImmutableList.of("src1.o", "src2.o"),
            ParameterFileType.UNQUOTED);
    Spawn spawn =
        new SpawnBuilder("/bin/gcc", "@output/foo-0.params", "@output/foo-1.params", "-o", "foo.o")
            .withInput(paramFile1)
            .withInput(paramFile2)
            .build();
    assertThat(spawn.getArgumentsWithExpandedParamFiles())
        .containsExactly("/bin/gcc", "-lfoo", "-lbar", "src1.o", "src2.o", "-o", "foo.o")
        .inOrder();
  }

  @Test
  public void getArgumentsWithExpandedParamFiles_unmatchedAtSignNotExpanded() throws Exception {
    // An argument starting with @ that doesn't match any param file should be left alone.
    Spawn spawn = new SpawnBuilder("/bin/gcc", "@some/other/file", "-o", "foo.o").build();
    assertThat(spawn.getArgumentsWithExpandedParamFiles())
        .containsExactly("/bin/gcc", "@some/other/file", "-o", "foo.o")
        .inOrder();
  }

  private static final ResourceSet DECLARED_RESOURCES =
      ResourceSet.createWithRamCpu(/* memoryMb= */ 250, /* cpu= */ 1);

  @Test
  public void getLocalResources_appliesResourceOverrides() throws Exception {
    Spawn spawn =
        new SpawnBuilder("/bin/true")
            .withLocalResources(DECLARED_RESOURCES)
            .withExecutionInfo("resources:cpu:4", "")
            .withCombinedExecProperties(ImmutableMap.of("resources:memory", "8000"))
            .build();
    ResourceSet resources = spawn.getLocalResources();

    // Execution info and exec properties both override the declared amounts.
    assertThat(resources.getCpuUsage()).isEqualTo(4.0);
    assertThat(resources.getMemoryMb()).isEqualTo(8000.0);
  }

  @Test
  public void getLocalResources_fixedDeclaration_ignoresResourceOverrides() throws Exception {
    Spawn spawn =
        new SpawnBuilder("/bin/true")
            .withLocalResources(ResourceSetOrBuilder.ignoringOverrides(DECLARED_RESOURCES))
            .withExecutionInfo("resources:cpu:4", "")
            .withCombinedExecProperties(ImmutableMap.of("resources:memory", "8000"))
            .build();

    assertThat(spawn.getLocalResources()).isEqualTo(DECLARED_RESOURCES);
    // The declarations still reach the spawn, so strategy selection is unaffected.
    assertThat(spawn.getExecutionInfo()).containsKey("resources:cpu:4");
    assertThat(spawn.getCombinedExecProperties()).containsEntry("resources:memory", "8000");
  }

  // NullAction's owner has no exec properties, so these only exercise the execution-info route.
  // Both routes share ResourceSetOrBuilder#buildLocalResources, which the cases above cover.
  private static BaseSpawn baseSpawn(ResourceSetOrBuilder localResources) {
    return new BaseSpawn(
        ImmutableList.of("/bin/true"),
        /* environment= */ ImmutableMap.of(),
        /* executionInfo= */ ImmutableMap.of("resources:cpu:4", ""),
        new ActionsTestUtil.NullAction(),
        localResources);
  }

  @Test
  public void baseSpawn_getLocalResources_appliesResourceOverrides() throws Exception {
    ResourceSet resources = baseSpawn(DECLARED_RESOURCES).getLocalResources();

    assertThat(resources.getCpuUsage()).isEqualTo(4.0);
  }

  @Test
  public void baseSpawn_getLocalResources_fixedDeclaration_keepsDeclaredResources()
      throws Exception {
    BaseSpawn spawn = baseSpawn(ResourceSetOrBuilder.ignoringOverrides(DECLARED_RESOURCES));

    assertThat(spawn.getLocalResources()).isEqualTo(DECLARED_RESOURCES);
    assertThat(spawn.getExecutionInfo()).containsKey("resources:cpu:4");
  }
}
