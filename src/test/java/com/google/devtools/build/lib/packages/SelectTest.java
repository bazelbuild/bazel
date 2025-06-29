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
package com.google.devtools.build.lib.packages;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.starlark.StarlarkGlobalsImpl;
import com.google.devtools.build.lib.cmdline.BazelModuleContext;
import com.google.devtools.build.lib.cmdline.BazelModuleKey;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.util.List;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Module;
import net.starlark.java.eval.Mutability;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkInt;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.syntax.FileOptions;
import net.starlark.java.syntax.ParserInput;
import net.starlark.java.syntax.SyntaxError;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests of {@code select} function and data type. */
@RunWith(TestParameterInjector.class)
public class SelectTest {

  private static Object eval(
      String expr, StarlarkSemantics semantics, @Nullable BazelModuleContext bazelModuleContext)
      throws SyntaxError.Exception, EvalException, InterruptedException {
    ParserInput input = ParserInput.fromLines(expr);
    Module module =
        Module.withPredeclaredAndData(
            semantics, StarlarkGlobalsImpl.INSTANCE.getUtilToplevels(), bazelModuleContext);
    try (Mutability mu = Mutability.create()) {
      StarlarkThread thread = StarlarkThread.createTransient(mu, semantics);
      return Starlark.eval(input, FileOptions.DEFAULT, module, thread);
    }
  }

  private static Object eval(String expr)
      throws SyntaxError.Exception, EvalException, InterruptedException {
    return eval(expr, StarlarkSemantics.DEFAULT, /* bazelModuleContext= */ null);
  }

  private static void assertFails(String expr, String wantError) {
    EvalException ex = assertThrows(EvalException.class, () -> eval(expr));
    assertThat(ex).hasMessageThat().contains(wantError);
  }

  @Test
  public void testSelect() throws Exception {
    SelectorList result = (SelectorList) eval("select({'a': 1})");
    assertThat(((SelectorValue) Iterables.getOnlyElement(result.getElements())).getDictionary())
        .containsExactly("a", StarlarkInt.of(1));
  }

  @Test
  public void testPlus() throws Exception {
    SelectorList x = (SelectorList) eval("select({'foo': ['FOO'], 'bar': ['BAR']}) + []");
    List<Object> elements = x.getElements();
    assertThat(elements).hasSize(2);
    assertThat(elements.get(0)).isInstanceOf(SelectorValue.class);
    assertThat((Iterable<?>) elements.get(1)).isEmpty();
  }

  @Test
  public void testPlusIncompatibleType() throws Exception {

    assertFails(
        "select({'foo': ['FOO'], 'bar': ['BAR']}) + 1",
        "Cannot combine incompatible types (select of list, int)");
    assertFails(
        "select({'foo': ['FOO']}) + select({'bar': 2})",
        "Cannot combine incompatible types (select of list, select of int)");

    assertFails(
        "select({'foo': ['FOO']}) + select({'bar': {'a': 'a'}})",
        "Cannot combine incompatible types (select of list, select of dict)");
    assertFails(
        "select({'bar': {'a': 'a'}}) + select({'foo': ['FOO']})",
        "Cannot combine incompatible types (select of dict, select of list)");
    assertFails(
        "['FOO'] + select({'bar': {'a': 'a'}})",
        "Cannot combine incompatible types (list, select of dict)");
    assertFails(
        "select({'bar': {'a': 'a'}}) + ['FOO']",
        "Cannot combine incompatible types (select of dict, list)");
    assertFails(
        "select({'foo': ['FOO']}) + {'a': 'a'}", "unsupported binary operation: select + dict");
    assertFails(
        "{'a': 'a'} + select({'foo': ['FOO']})", "unsupported binary operation: dict + select");
  }

  @Test
  public void testUnionIncompatibleType() throws Exception {
    assertFails(
        "select({'foo': ['FOO']}) | select({'bar': {'a': 'a'}})",
        "Cannot combine incompatible types (select of list, select of dict)");
    assertFails(
        "select({'bar': {'a': 'a'}}) | select({'foo': ['FOO']})",
        "Cannot combine incompatible types (select of dict, select of list)");
    assertFails(
        "['FOO'] | select({'bar': {'a': 'a'}})", "unsupported binary operation: list | select");
    assertFails(
        "select({'bar': {'a': 'a'}}) | ['FOO']", "unsupported binary operation: select | list");
    assertFails(
        "select({'foo': ['FOO']}) | {'a': 'a'}",
        "Cannot combine incompatible types (select of list, dict)");
    assertFails(
        "{'a': 'a'} | select({'foo': ['FOO']})",
        "Cannot combine incompatible types (dict, select of list)");
  }

  @Test
  public void testRepr() throws Exception {
    assertThat(eval("repr(select({'foo': ['FOO']})+['BAR'])"))
        .isEqualTo("select({\"foo\": [\"FOO\"]}) + [\"BAR\"]");

    assertThat(eval("repr(['FOO']+select({'bar': ['BAR']}))"))
        .isEqualTo("[\"FOO\"] + select({\"bar\": [\"BAR\"]})");

    assertThat(eval("repr(select({'foo': ['FOO']})+select({'bar': ['BAR']}))"))
        .isEqualTo("select({\"foo\": [\"FOO\"]}) + select({\"bar\": [\"BAR\"]})");

    assertThat(eval("repr(select({'foo': {'FOO': 123}})|{'BAR': 456})"))
        .isEqualTo("select({\"foo\": {\"FOO\": 123}}) | {\"BAR\": 456}");

    assertThat(eval("repr({'FOO': 123}|select({'bar': {'BAR': 456}}))"))
        .isEqualTo("{\"FOO\": 123} | select({\"bar\": {\"BAR\": 456}})");

    assertThat(eval("repr(select({'foo': {'FOO': 123}})|select({'bar': {'BAR': 456}}))"))
        .isEqualTo("select({\"foo\": {\"FOO\": 123}}) | select({\"bar\": {\"BAR\": 456}})");
  }

  @Test
  public void testKeyResolution(@TestParameter boolean resolveSelectKeysEagerly) throws Exception {
    var ctx =
        BazelModuleContext.create(
            BazelModuleKey.createFakeModuleKeyForTesting(
                Label.parseCanonicalUnchecked("//other/pkg:def.bzl")),
            RepositoryMapping.create(
                ImmutableMap.of(
                    "",
                    RepositoryName.MAIN,
                    "other_repo",
                    RepositoryName.createUnvalidated("other_repo+")),
                RepositoryName.MAIN),
            "other/pkg/def.bzl",
            /* loads= */ ImmutableList.of(),
            /* bzlTransitiveDigest= */ new byte[0]);
    var semantics =
        StarlarkSemantics.builder()
            .setBool(
                BuildLanguageOptions.INCOMPATIBLE_RESOLVE_SELECT_KEYS_EAGERLY,
                resolveSelectKeysEagerly)
            .build();
    var result =
        (SelectorList)
            eval("select({'a': 1, '//pkg:b': 2, '@other_repo//:file': 3})", semantics, ctx);
    var selectDict =
        ((SelectorValue) Iterables.getOnlyElement(result.getElements())).getDictionary();
    if (resolveSelectKeysEagerly) {
      assertThat(selectDict)
          .containsExactly(
              Label.parseCanonicalUnchecked("//other/pkg:a"),
              StarlarkInt.of(1),
              Label.parseCanonicalUnchecked("//pkg:b"),
              StarlarkInt.of(2),
              Label.parseCanonicalUnchecked("@@other_repo+//:file"),
              StarlarkInt.of(3))
          .inOrder();
    } else {
      assertThat(selectDict)
          .containsExactly(
              "a", StarlarkInt.of(1),
              "//pkg:b", StarlarkInt.of(2),
              "@other_repo//:file", StarlarkInt.of(3))
          .inOrder();
    }
  }
}
