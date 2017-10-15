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

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandAction;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.rules.objc.ObjcCommandLineOptions.ObjcCrosstoolMode;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test case for discovery of used headers during compilation. */
@RunWith(JUnit4.class)
public class HeaderDiscoveryTest extends ObjcRuleTestCase {

  @Override
  protected void useConfiguration(String... args) throws Exception {
    // Don't test crosstool for header discovery
    useConfiguration(ObjcCrosstoolMode.OFF, args);
  }

  private NestedSet<Artifact> getDiscoveredInputsForDotdFile(String... lines) throws Exception {
    createLibraryTargetWriter("//objc:lib")
        .setList("srcs", "a.m")
        .setList("hdrs", "used.h", "not_used.h")
        .write();
    CommandAction commandAction = compileAction("//objc:lib", "a.o");
    assertThat(commandAction).isInstanceOf(ObjcCompileAction.class);
    ObjcCompileAction objcCompileAction = (ObjcCompileAction) commandAction;

    Artifact dotdFile = ActionsTestUtil.getOutput(objcCompileAction, "a.d");
    scratch.file(dotdFile.getPath().toString(), lines);
    return objcCompileAction.discoverInputsFromDotdFiles(
        directories.getExecRoot(), view.getArtifactFactory());
  }

  @Test
  public void testObjcHeaderDiscoveryFindsInputs() throws Exception {
    NestedSet<Artifact> discoveredInputs =
        getDiscoveredInputsForDotdFile("a.o: \\", "  objc/a.m \\", "  objc/used.h \\");
    assertThat(Artifact.toExecPaths(discoveredInputs)).containsExactly("objc/a.m", "objc/used.h");
  }

  @Test
  public void testObjcHeaderDiscoveryIgnoresOutsideExecRoot() throws Exception {
    NestedSet<Artifact> discoveredInputs =
        getDiscoveredInputsForDotdFile("a.o: \\", "  /foo/a.h \\", "  /bar/b.h");
    assertThat(Artifact.toExecPaths(discoveredInputs)).isEmpty();
  }

  @Test
  public void testInputsArePruned() throws Exception {
    NestedSet<Artifact> discoveredInputs =
        getDiscoveredInputsForDotdFile("a.o: \\", "  objc/a.m \\", "  objc/used.h \\");
    ObjcCompileAction compileAction = (ObjcCompileAction) compileAction("//objc:lib", "a.o");
    compileAction.updateActionInputs(discoveredInputs);

    assertThat(Artifact.toExecPaths(compileAction.getInputs())).doesNotContain("objc/not_used.h");
  }

  @Test
  public void testSrcsAreMandatoryInputs() throws Exception {
    NestedSet<Artifact> discoveredInputs =
        getDiscoveredInputsForDotdFile("a.o: \\", "  objc/used.h");
    ObjcCompileAction compileAction = (ObjcCompileAction) compileAction("//objc:lib", "a.o");
    compileAction.updateActionInputs(discoveredInputs);

    assertThat(Artifact.toExecPaths(compileAction.getInputs())).contains("objc/a.m");
  }
}
