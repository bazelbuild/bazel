// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.syntax;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.syntax.StarlarkSemantics.FlagIdentifier;
import com.google.devtools.build.lib.syntax.util.EvaluationTestCase;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Starlark evaluation tests which verify the infrastructure which toggles build API methods and
 * parameters with semantic flags.
 */
@RunWith(JUnit4.class)
public final class StarlarkFlagGuardingTest {

  private EvaluationTestCase ev = new EvaluationTestCase();

  /** Mock containing exposed methods for flag-guarding tests. */
  @StarlarkBuiltin(name = "Mock", doc = "")
  public static class Mock implements StarlarkValue {

    @StarlarkMethod(
        name = "positionals_only_method",
        documented = false,
        parameters = {
          @Param(name = "a", positional = true, named = false, type = Integer.class),
          @Param(
              name = "b",
              positional = true,
              named = false,
              type = Boolean.class,
              enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_SIBLING_REPOSITORY_LAYOUT,
              valueWhenDisabled = "False"),
          @Param(name = "c", positional = true, named = false, type = Integer.class),
        },
        useStarlarkThread = true)
    public String positionalsOnlyMethod(Integer a, boolean b, Integer c, StarlarkThread thread) {
      return "positionals_only_method(" + a + ", " + b + ", " + c + ")";
    }

    @StarlarkMethod(
        name = "keywords_only_method",
        documented = false,
        parameters = {
          @Param(name = "a", positional = false, named = true, type = Integer.class),
          @Param(
              name = "b",
              positional = false,
              named = true,
              type = Boolean.class,
              enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_SIBLING_REPOSITORY_LAYOUT,
              valueWhenDisabled = "False"),
          @Param(name = "c", positional = false, named = true, type = Integer.class),
        },
        useStarlarkThread = true)
    public String keywordsOnlyMethod(Integer a, boolean b, Integer c, StarlarkThread thread) {
      return "keywords_only_method(" + a + ", " + b + ", " + c + ")";
    }

    @StarlarkMethod(
        name = "mixed_params_method",
        documented = false,
        parameters = {
          @Param(name = "a", positional = true, named = false, type = Integer.class),
          @Param(
              name = "b",
              positional = true,
              named = false,
              type = Boolean.class,
              enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_SIBLING_REPOSITORY_LAYOUT,
              valueWhenDisabled = "False"),
          @Param(
              name = "c",
              positional = false,
              named = true,
              type = Integer.class,
              enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_SIBLING_REPOSITORY_LAYOUT,
              valueWhenDisabled = "3"),
          @Param(name = "d", positional = false, named = true, type = Boolean.class),
        },
        useStarlarkThread = true)
    public String mixedParamsMethod(
        Integer a, boolean b, Integer c, boolean d, StarlarkThread thread) {
      return "mixed_params_method(" + a + ", " + b + ", " + c + ", " + d + ")";
    }

    @StarlarkMethod(
        name = "keywords_multiple_flags",
        documented = false,
        parameters = {
          @Param(name = "a", positional = false, named = true, type = Integer.class),
          @Param(
              name = "b",
              positional = false,
              named = true,
              type = Boolean.class,
              disableWithFlag = FlagIdentifier.INCOMPATIBLE_NO_ATTR_LICENSE,
              valueWhenDisabled = "False"),
          @Param(
              name = "c",
              positional = false,
              named = true,
              type = Integer.class,
              enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_SIBLING_REPOSITORY_LAYOUT,
              valueWhenDisabled = "3"),
        },
        useStarlarkThread = true)
    public String keywordsMultipleFlags(Integer a, boolean b, Integer c, StarlarkThread thread) {
      return "keywords_multiple_flags(" + a + ", " + b + ", " + c + ")";
    }
  }

  @Test
  public void testPositionalsOnlyGuardedMethod() throws Exception {
    ev.new Scenario("--experimental_sibling_repository_layout=true")
        .update("mock", new Mock())
        .testEval(
            "mock.positionals_only_method(1, True, 3)", "'positionals_only_method(1, true, 3)'");

    ev.new Scenario("--experimental_sibling_repository_layout=true")
        .update("mock", new Mock())
        .testIfErrorContains(
            "in call to positionals_only_method(), parameter 'b' got value of type 'int', want"
                + " 'bool'",
            "mock.positionals_only_method(1, 3)");

    ev.new Scenario("--experimental_sibling_repository_layout=false")
        .update("mock", new Mock())
        .testEval("mock.positionals_only_method(1, 3)", "'positionals_only_method(1, false, 3)'");

    ev.new Scenario("--experimental_sibling_repository_layout=false")
        .update("mock", new Mock())
        .testIfErrorContains(
            "in call to positionals_only_method(), parameter 'c' got value of type 'bool', want"
                + " 'int'",
            "mock.positionals_only_method(1, True, 3)");
  }

  @Test
  public void testKeywordOnlyGuardedMethod() throws Exception {
    ev.new Scenario("--experimental_sibling_repository_layout=true")
        .update("mock", new Mock())
        .testEval(
            "mock.keywords_only_method(a=1, b=True, c=3)", "'keywords_only_method(1, true, 3)'");

    ev.new Scenario("--experimental_sibling_repository_layout=true")
        .update("mock", new Mock())
        .testIfErrorContains(
            "keywords_only_method() missing 1 required named argument: b",
            "mock.keywords_only_method(a=1, c=3)");

    ev.new Scenario("--experimental_sibling_repository_layout=false")
        .update("mock", new Mock())
        .testEval("mock.keywords_only_method(a=1, c=3)", "'keywords_only_method(1, false, 3)'");

    ev.new Scenario("--experimental_sibling_repository_layout=false")
        .update("mock", new Mock())
        .testIfErrorContains(
            "parameter 'b' is experimental and thus unavailable with the current "
                + "flags. It may be enabled by setting "
                + "--experimental_sibling_repository_layout",
            "mock.keywords_only_method(a=1, b=True, c=3)");
  }

  @Test
  public void testMixedParamsMethod() throws Exception {
    // def mixed_params_method(a, b, c = ?, d = ?)
    ev.new Scenario("--experimental_sibling_repository_layout=true")
        .update("mock", new Mock())
        .testEval(
            "mock.mixed_params_method(1, True, c=3, d=True)",
            "'mixed_params_method(1, true, 3, true)'");

    ev.new Scenario("--experimental_sibling_repository_layout=true")
        .update("mock", new Mock())
        .testIfErrorContains(
            // Missing named arguments (d) are not reported
            // if there are missing positional arguments.
            "mixed_params_method() missing 1 required positional argument: b",
            "mock.mixed_params_method(1, c=3)");

    // def mixed_params_method(a, b disabled = False, c disabled = 3, d = ?)
    ev.new Scenario("--experimental_sibling_repository_layout=false")
        .update("mock", new Mock())
        .testEval(
            "mock.mixed_params_method(1, d=True)", "'mixed_params_method(1, false, 3, true)'");

    ev.new Scenario("--experimental_sibling_repository_layout=false")
        .update("mock", new Mock())
        .testIfErrorContains(
            "mixed_params_method() accepts no more than 1 positional argument but got 2",
            "mock.mixed_params_method(1, True, d=True)");

    ev.new Scenario("--experimental_sibling_repository_layout=false")
        .update("mock", new Mock())
        .testIfErrorContains(
            "mixed_params_method() accepts no more than 1 positional argument but got 2",
            "mock.mixed_params_method(1, True, c=3, d=True)");
  }

  @Test
  public void testKeywordsMultipleFlags() throws Exception {
    ev
        .new Scenario(
            "--experimental_sibling_repository_layout=true", "--incompatible_no_attr_license=false")
        .update("mock", new Mock())
        .testEval(
            "mock.keywords_multiple_flags(a=42, b=True, c=0)",
            "'keywords_multiple_flags(42, true, 0)'");

    ev
        .new Scenario(
            "--experimental_sibling_repository_layout=true", "--incompatible_no_attr_license=false")
        .update("mock", new Mock())
        .testIfErrorContains(
            "keywords_multiple_flags() missing 2 required named arguments: b, c",
            "mock.keywords_multiple_flags(a=42)");

    ev
        .new Scenario(
            "--experimental_sibling_repository_layout=false", "--incompatible_no_attr_license=true")
        .update("mock", new Mock())
        .testEval("mock.keywords_multiple_flags(a=42)", "'keywords_multiple_flags(42, false, 3)'");

    ev
        .new Scenario(
            "--experimental_sibling_repository_layout=false", "--incompatible_no_attr_license=true")
        .update("mock", new Mock())
        .testIfErrorContains(
            "parameter 'b' is deprecated and will be removed soon. It may be "
                + "temporarily re-enabled by setting --incompatible_no_attr_license=false",
            "mock.keywords_multiple_flags(a=42, b=True, c=0)");
  }

  @Test
  public void testExperimentalFlagGuardedValue() throws Exception {
    // This test uses an arbitrary experimental flag to verify this functionality. If this
    // experimental flag were to go away, this test may be updated to use any experimental flag.
    // The flag itself is unimportant to the test.

    // clumsy way to predeclare
    ev =
        new EvaluationTestCase() {
          @Override
          protected Object newModuleHook(ImmutableMap.Builder<String, Object> predeclared) {
            predeclared.put(
                "GlobalSymbol",
                FlagGuardedValue.onlyWhenExperimentalFlagIsTrue(
                    FlagIdentifier.EXPERIMENTAL_SIBLING_REPOSITORY_LAYOUT, "foo"));
            return null; // no client data
          }
        };

    String errorMessage =
        "GlobalSymbol is experimental and thus unavailable with the current "
            + "flags. It may be enabled by setting --experimental_sibling_repository_layout";

    ev.new Scenario("--experimental_sibling_repository_layout=true")
        .setUp("var = GlobalSymbol")
        .testLookup("var", "foo");

    ev.new Scenario("--experimental_sibling_repository_layout=false")
        .testIfErrorContains(errorMessage, "var = GlobalSymbol");

    ev.new Scenario("--experimental_sibling_repository_layout=false")
        .testIfErrorContains(errorMessage, "def my_function():", "  var = GlobalSymbol");

    ev.new Scenario("--experimental_sibling_repository_layout=false")
        .setUp("GlobalSymbol = 'other'", "var = GlobalSymbol")
        .testLookup("var", "other");
  }

  @Test
  public void testIncompatibleFlagGuardedValue() throws Exception {
    // This test uses an arbitrary incompatible flag to verify this functionality. If this
    // incompatible flag were to go away, this test may be updated to use any incompatible flag.
    // The flag itself is unimportant to the test.

    ev =
        new EvaluationTestCase() {
          @Override
          protected Object newModuleHook(ImmutableMap.Builder<String, Object> predeclared) {
            predeclared.put(
                "GlobalSymbol",
                FlagGuardedValue.onlyWhenIncompatibleFlagIsFalse(
                    FlagIdentifier.INCOMPATIBLE_LINKOPTS_TO_LINKLIBS, "foo"));
            return null; // no client data
          }
        };

    String errorMessage =
        "GlobalSymbol is deprecated and will be removed soon. It may be "
            + "temporarily re-enabled by setting --incompatible_linkopts_to_linklibs=false";

    ev.new Scenario("--incompatible_linkopts_to_linklibs=false")
        .setUp("var = GlobalSymbol")
        .testLookup("var", "foo");

    ev.new Scenario("--incompatible_linkopts_to_linklibs=true")
        .testIfErrorContains(errorMessage, "var = GlobalSymbol");

    ev.new Scenario("--incompatible_linkopts_to_linklibs=true")
        .testIfErrorContains(errorMessage, "def my_function():", "  var = GlobalSymbol");

    ev.new Scenario("--incompatible_linkopts_to_linklibs=true")
        .setUp("GlobalSymbol = 'other'", "var = GlobalSymbol")
        .testLookup("var", "other");
  }
}
