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

package com.google.devtools.build.lib.rules.cpp;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.base.Predicate;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Variables;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests that {@code CppCompileAction} is populated with the correct build variables. */
@RunWith(JUnit4.class)
public class CompileBuildVariablesTest extends BuildViewTestCase {

  private CppCompileAction getCppCompileAction(final String label, final String name) throws
      Exception {
    return (CppCompileAction)
        getGeneratingAction(
            Iterables.find(
                getGeneratingAction(
                    Iterables.getOnlyElement(getFilesToBuild(getConfiguredTarget(label))))
                    .getInputs(),
                new Predicate<Artifact>() {
                  @Override
                  public boolean apply(Artifact artifact) {
                    return artifact.getExecPath().getBaseName().startsWith(name);
                  }
                }));
  }

  /** Returns active build variables for a compile action of given type for given target. */
  protected Variables getCompileBuildVariables(String label, String name) throws Exception {
    return getCppCompileAction(label, name).getCompileCommandLine().getVariables();
  }

  @Test
  public void testPresenceOfBasicVariables() throws Exception {
    scratch.file("x/BUILD", "cc_binary(name = 'bin', srcs = ['bin.cc'])");
    scratch.file("x/bin.cc");

    Variables variables = getCompileBuildVariables("//x:bin", "bin");

    assertThat(variables.getStringVariable(CppModel.SOURCE_FILE_VARIABLE_NAME))
        .contains("x/bin.cc");
    assertThat(variables.getStringVariable(CppModel.OUTPUT_FILE_VARIABLE_NAME))
        .contains("x/bin");
  }
}
