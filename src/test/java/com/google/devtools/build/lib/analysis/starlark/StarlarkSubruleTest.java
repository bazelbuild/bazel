// Copyright 2023 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis.starlark;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.starlark.util.BazelEvaluationTestCase;
import com.google.devtools.build.lib.starlarkbuildapi.StarlarkSubruleApi;
import net.starlark.java.eval.BuiltinFunction;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class StarlarkSubruleTest extends BuildViewTestCase {

  private final BazelEvaluationTestCase ev = new BazelEvaluationTestCase("//subrule_testing:label");
  private final BazelEvaluationTestCase evOutsideAllowlist =
      new BazelEvaluationTestCase("//foo:bar");

  @Test
  public void testSubruleFunctionSymbol_notVisibleInBUILD() throws Exception {
    scratch.file("foo/BUILD", "subrule");

    checkLoadingPhaseError("//foo", "'subrule' is not defined");
  }

  @Test
  // checks that 'subrule' symbol visibility in bzl files, not whether it's callable
  public void testSubruleFunctionSymbol_isVisibleInBzl() throws Exception {
    Object subruleFunction = ev.eval("subrule");

    assertNoEvents();
    assertThat(subruleFunction).isNotNull();
    assertThat(subruleFunction).isInstanceOf(BuiltinFunction.class);
  }

  @Test
  public void testSubruleInstantiation_inAllowlistedPackage_succeeds() throws Exception {
    Object subrule = ev.eval("subrule(implementation = lambda : 0 )");

    assertThat(subrule).isNotNull();
    assertThat(subrule).isInstanceOf(StarlarkSubruleApi.class);
  }

  @Test
  public void testSubrule_isCallable() throws Exception {
    ev.exec("x = subrule(implementation = lambda : 'dummy result' )");

    Object result = ev.eval("x()");

    assertThat(result).isNotNull();
    assertThat(result).isEqualTo("dummy result");
  }

  @Test
  public void testSubruleInstantiation_outsideAllowlist_failsWithPrivateAPIError()
      throws Exception {
    evOutsideAllowlist.checkEvalErrorContains(
        "'//foo:bar' cannot use private API", "subrule(implementation = lambda: 0 )");
  }
}
