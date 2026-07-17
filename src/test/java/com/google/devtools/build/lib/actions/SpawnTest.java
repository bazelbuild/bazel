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
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
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
}
