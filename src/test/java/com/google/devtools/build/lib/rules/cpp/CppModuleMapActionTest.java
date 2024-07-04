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
package com.google.devtools.build.lib.rules.cpp;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link CppModuleMapAction}. */
@RunWith(JUnit4.class)
public final class CppModuleMapActionTest {

  private ArtifactRoot outputRoot;
  private final ActionKeyContext actionKeyContext = new ActionKeyContext();

  @Before
  public void createOutputRoot() throws IOException {
    Scratch scratch = new Scratch();
    outputRoot = ArtifactRoot.asDerivedRoot(scratch.dir("/execroot"), RootType.Output, "out");
  }

  @Test
  public void getKey_dependencyWithEqualExecPathAndNewName_returnsDifferentKey() throws Exception {
    CppModuleMap map = new CppModuleMap(createOutputArtifact("foo.cppmap"), "foo");
    Artifact depCppMap = createOutputArtifact("dep.cppmap");
    CppModuleMap dep = new CppModuleMap(depCppMap, "oldName");
    CppModuleMap depWithDifferentName = new CppModuleMap(depCppMap, "newName");
    CppModuleMapAction action1 = createCppModuleMapAction(map, dep);
    CppModuleMapAction action2 = createCppModuleMapAction(map, depWithDifferentName);

    assertThat(action1.getKey(actionKeyContext, /*artifactExpander=*/ null))
        .isNotEqualTo(action2.getKey(actionKeyContext, /*artifactExpander=*/ null));
  }

  private static CppModuleMapAction createCppModuleMapAction(
      CppModuleMap cppModuleMap, CppModuleMap... dependencies) {
    return new CppModuleMapAction(
        ActionsTestUtil.NULL_ACTION_OWNER,
        cppModuleMap,
        /* privateHeaders= */ ImmutableList.of(),
        /* publicHeaders= */ ImmutableList.of(),
        ImmutableList.copyOf(dependencies),
        /* additionalExportedHeaders= */ ImmutableList.of(),
        /* separateModuleHeaders= */ ImmutableList.of(),
        /* compiledModule= */ false,
        /* moduleMapHomeIsCwd= */ false,
        /* generateSubmodules= */ false,
        /* externDependencies= */ false,
        CoreOptions.OutputPathsMode.OFF,
        (executionInfo, mnemonic) -> executionInfo);
  }

  private Artifact createOutputArtifact(String rootRelativePath) {
    return ActionsTestUtil.createArtifactWithRootRelativePath(
        outputRoot, PathFragment.create(rootRelativePath));
  }
}
