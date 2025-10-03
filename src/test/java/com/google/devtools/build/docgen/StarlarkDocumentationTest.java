// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.docgen;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.skyframe.BzlLoadValue.keyForBuild;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Sets;
import com.google.devtools.build.docgen.StarlarkDocumentationProcessor.Category;
import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.docgen.annot.GlobalMethods;
import com.google.devtools.build.docgen.annot.GlobalMethods.Environment;
import com.google.devtools.build.docgen.annot.StarlarkConstructor;
import com.google.devtools.build.docgen.starlark.AnnotStarlarkConstructorMethodDoc;
import com.google.devtools.build.docgen.starlark.MemberDoc;
import com.google.devtools.build.docgen.starlark.ParamDoc;
import com.google.devtools.build.docgen.starlark.StarlarkDoc;
import com.google.devtools.build.docgen.starlark.StarlarkDocExpander;
import com.google.devtools.build.docgen.starlark.StarlarkDocPage;
import com.google.devtools.build.lib.analysis.starlark.StarlarkGlobalsImpl;
import com.google.devtools.build.lib.analysis.starlark.StarlarkRuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.skyframe.BzlLoadFunction;
import com.google.devtools.build.lib.starlark.util.BazelEvaluationTestCase;
import com.google.devtools.build.lib.starlarkdocextract.LabelRenderer;
import com.google.devtools.build.lib.starlarkdocextract.ModuleInfoExtractor;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.ModuleInfo;
import com.google.devtools.build.lib.util.Classpath;
import com.google.devtools.build.lib.util.Classpath.ClassPathException;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.Module;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.StarlarkValue;
import net.starlark.java.eval.Tuple;
import net.starlark.java.syntax.FileOptions;
import net.starlark.java.syntax.ParserInput;
import net.starlark.java.syntax.Program;
import net.starlark.java.syntax.StarlarkFile;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for Starlark documentation. */
@RunWith(JUnit4.class)
public class StarlarkDocumentationTest {

  private static final ImmutableList<String>
      DEPRECATED_OR_EXPERIMENTAL_UNDOCUMENTED_TOP_LEVEL_SYMBOLS = ImmutableList.of("Actions");

  private static final StarlarkDocExpander expander =
      new StarlarkDocExpander(null) {

        @Override
        public String expand(String docString) {
          return docString;
        }
      };

  @Test
  public void testStarlarkRuleClassBuiltInItemsAreDocumented() throws Exception {
    checkStarlarkTopLevelEnvItemsAreDocumented(StarlarkGlobalsImpl.INSTANCE.getFixedBzlToplevels());
  }

  private void checkStarlarkTopLevelEnvItemsAreDocumented(Map<String, Object> globals)
      throws Exception {
    ImmutableMap<Category, ImmutableList<StarlarkDocPage>> allPages =
        StarlarkDocumentationCollector.getAllDocPages(expander, ImmutableList.of());
    ImmutableSet<String> documentedItems =
        Stream.concat(
                allPages.get(Category.GLOBAL_FUNCTION).stream()
                    .flatMap(p -> p.getMembers().stream()),
                allPages.entrySet().stream()
                    .filter(e -> !e.getKey().equals(Category.GLOBAL_FUNCTION))
                    .flatMap(e -> e.getValue().stream()))
            .filter(m -> !m.getDocumentation().isEmpty())
            .map(StarlarkDoc::getName)
            .collect(ImmutableSet.toImmutableSet());

    assertThat(
            Sets.difference(
                Sets.difference(globals.keySet(), documentedItems),
                // These constants are currently undocumented.
                // If they need documentation, the easiest approach would be
                // to hard-code it in StarlarkDocumentationCollector.
                ImmutableSet.of("True", "False", "None")))
        .containsExactlyElementsIn(DEPRECATED_OR_EXPERIMENTAL_UNDOCUMENTED_TOP_LEVEL_SYMBOLS);
  }

  // TODO(bazel-team): come up with better Starlark specific tests.
  @Test
  public void testDirectJavaMethodsAreGenerated() throws Exception {
    assertThat(collect(StarlarkRuleContext.class)).isNotEmpty();
  }

  /** MockClassA */
  @StarlarkBuiltin(name = "MockClassA", category = DocCategory.BUILTIN, doc = "MockClassA")
  private static class MockClassA implements StarlarkValue {
    @StarlarkMethod(name = "get", doc = "MockClassA#get")
    public Integer get() {
      return 0;
    }
  }

  /** MockClassD */
  @StarlarkBuiltin(name = "MockClassD", category = DocCategory.BUILTIN, doc = "MockClassD")
  @SuppressWarnings("unused") // test code
  private static class MockClassD implements StarlarkValue {
    @StarlarkMethod(
        name = "test",
        doc = "MockClassD#test",
        parameters = {
          @Param(name = "a"),
          @Param(name = "b"),
          @Param(name = "c", named = true, positional = false),
          @Param(name = "d", named = true, positional = false, defaultValue = "1"),
        })
    public Integer test(int a, int b, int c, int d) {
      return 0;
    }
  }

  /** MockClassE */
  @StarlarkBuiltin(name = "MockClassE", category = DocCategory.BUILTIN, doc = "MockClassE")
  private static class MockClassE extends MockClassA {
    @Override
    public Integer get() {
      return 1;
    }
  }

  /** MockClassF */
  @StarlarkBuiltin(name = "MockClassF", category = DocCategory.BUILTIN, doc = "MockClassF")
  @SuppressWarnings("unused") // test code
  private static class MockClassF implements StarlarkValue {
    @StarlarkMethod(
        name = "test",
        doc = "MockClassF#test",
        parameters = {
          @Param(name = "a", named = false, positional = true),
          @Param(name = "b", named = true, positional = true),
          @Param(name = "c", named = true, positional = false),
          @Param(name = "d", named = true, positional = false, defaultValue = "1"),
        },
        extraPositionals = @Param(name = "myArgs"))
    public Integer test(int a, int b, int c, int d, Sequence<?> args) {
      return 0;
    }
  }

  /** MockClassG */
  @StarlarkBuiltin(name = "MockClassG", category = DocCategory.BUILTIN, doc = "MockClassG")
  @SuppressWarnings("unused") // test code
  private static class MockClassG implements StarlarkValue {
    @StarlarkMethod(
        name = "test",
        doc = "MockClassG#test",
        parameters = {
          @Param(name = "a", named = false, positional = true),
          @Param(name = "b", named = true, positional = true),
          @Param(name = "c", named = true, positional = false),
          @Param(name = "d", named = true, positional = false, defaultValue = "1"),
        },
        extraKeywords = @Param(name = "myKwargs"))
    public Integer test(int a, int b, int c, int d, Dict<?, ?> kwargs) {
      return 0;
    }
  }

  /** MockClassH */
  @StarlarkBuiltin(name = "MockClassH", category = DocCategory.BUILTIN, doc = "MockClassH")
  @SuppressWarnings("unused") // test code
  private static class MockClassH implements StarlarkValue {
    @StarlarkMethod(
        name = "test",
        doc = "MockClassH#test",
        parameters = {
          @Param(name = "a", named = false, positional = true),
          @Param(name = "b", named = true, positional = true),
          @Param(name = "c", named = true, positional = false),
          @Param(name = "d", named = true, positional = false, defaultValue = "1"),
        },
        extraPositionals = @Param(name = "myArgs"),
        extraKeywords = @Param(name = "myKwargs"))
    public Integer test(int a, int b, int c, int d, Sequence<?> args, Dict<?, ?> kwargs) {
      return 0;
    }
  }

  /** MockClassI */
  @StarlarkBuiltin(name = "MockClassI", category = DocCategory.BUILTIN, doc = "MockClassI")
  @SuppressWarnings("unused") // test code
  private static class MockClassI implements StarlarkValue {
    @StarlarkMethod(
        name = "test",
        doc = "MockClassI#test",
        parameters = {
          @Param(name = "a", named = false, positional = true),
          @Param(name = "b", named = true, positional = true),
          @Param(name = "c", named = true, positional = false),
          @Param(name = "d", named = true, positional = false, defaultValue = "1"),
          @Param(
              name = "e",
              named = true,
              positional = false,
              documented = false,
              defaultValue = "2"),
        },
        extraPositionals = @Param(name = "myArgs"))
    public Integer test(int a, int b, int c, int d, int e, Sequence<?> args) {
      return 0;
    }
  }

  /**
   * MockGlobalLibrary. While nothing directly depends on it, a test method in
   * StarlarkDocumentationTest checks all of the classes under a wide classpath and ensures this one
   * shows up.
   */
  @GlobalMethods(environment = Environment.BZL)
  @SuppressWarnings("unused")
  private static class MockGlobalLibrary {
    @StarlarkMethod(
        name = "MockGlobalCallable",
        doc = "GlobalCallable documentation",
        parameters = {
          @Param(name = "a", named = false, positional = true),
          @Param(name = "b", named = true, positional = true),
          @Param(name = "c", named = true, positional = false),
          @Param(name = "d", named = true, positional = false, defaultValue = "1"),
        },
        extraPositionals = @Param(name = "myArgs"),
        extraKeywords = @Param(name = "myKwargs"))
    public Integer test(int a, int b, int c, int d, Sequence<?> args, Dict<?, ?> kwargs) {
      return 0;
    }
  }

  /** MockClassWithContainerReturnValues */
  @StarlarkBuiltin(
      name = "MockClassWithContainerReturnValues",
      category = DocCategory.BUILTIN,
      doc = "MockClassWithContainerReturnValues")
  private static class MockClassWithContainerReturnValues implements StarlarkValue {

    @StarlarkMethod(name = "depset", doc = "depset")
    public Depset /*<Integer>*/ getNestedSet() {
      return null;
    }

    @StarlarkMethod(name = "tuple", doc = "tuple")
    public Tuple getTuple() {
      return null;
    }

    @StarlarkMethod(name = "immutable", doc = "immutable")
    public ImmutableList<Integer> getImmutableList() {
      return null;
    }

    @StarlarkMethod(name = "mutable", doc = "mutable")
    public StarlarkList<Integer> getMutableList() {
      return null;
    }

    @StarlarkMethod(name = "starlark", doc = "starlark")
    public Sequence<Integer> getStarlarkList() {
      return null;
    }
  }

  /** MockClassCommonNameOne */
  @StarlarkBuiltin(
      name = "MockClassCommonName",
      category = DocCategory.BUILTIN,
      doc = "MockClassCommonName")
  private static class MockClassCommonNameOne implements StarlarkValue {

    @StarlarkMethod(name = "one", doc = "one")
    public Integer one() {
      return 1;
    }
  }

  /** SubclassOfMockClassCommonNameOne */
  @StarlarkBuiltin(
      name = "MockClassCommonName",
      category = DocCategory.BUILTIN,
      doc = "MockClassCommonName")
  private static class SubclassOfMockClassCommonNameOne extends MockClassCommonNameOne {

    @StarlarkMethod(name = "two", doc = "two")
    public Integer two() {
      return 1;
    }
  }

  /** PointsToCommonNameOneWithSubclass */
  @StarlarkBuiltin(
      name = "PointsToCommonNameOneWithSubclass",
      category = DocCategory.BUILTIN,
      doc = "PointsToCommonNameOneWithSubclass")
  private static class PointsToCommonNameOneWithSubclass implements StarlarkValue {
    @StarlarkMethod(name = "one", doc = "one")
    public MockClassCommonNameOne getOne() {
      return null;
    }

    @StarlarkMethod(name = "one_subclass", doc = "one_subclass")
    public SubclassOfMockClassCommonNameOne getOneSubclass() {
      return null;
    }
  }

  /** MockClassCommonNameOneUndocumented */
  @StarlarkBuiltin(name = "MockClassCommonName", documented = false, doc = "")
  private static class MockClassCommonNameUndocumented implements StarlarkValue {

    @StarlarkMethod(name = "two", doc = "two")
    public Integer two() {
      return 1;
    }
  }

  /** PointsToCommonNameAndUndocumentedModule */
  @StarlarkBuiltin(
      name = "PointsToCommonNameAndUndocumentedModule",
      category = DocCategory.BUILTIN,
      doc = "PointsToCommonNameAndUndocumentedModule")
  private static class PointsToCommonNameAndUndocumentedModule implements StarlarkValue {
    @StarlarkMethod(name = "one", doc = "one")
    public MockClassCommonNameOne getOne() {
      return null;
    }

    @StarlarkMethod(name = "undocumented_module", doc = "undocumented_module")
    public MockClassCommonNameUndocumented getUndocumented() {
      return null;
    }
  }

  /** A module which has a selfCall method which constructs copies of MockClassA. */
  @StarlarkBuiltin(
      name = "MockClassWithSelfCallConstructor",
      category = DocCategory.BUILTIN,
      doc = "MockClassWithSelfCallConstructor")
  private static class MockClassWithSelfCallConstructor implements StarlarkValue {
    @StarlarkMethod(name = "one", doc = "one")
    public MockClassCommonNameOne getOne() {
      return null;
    }

    @StarlarkMethod(name = "makeMockClassA", selfCall = true, doc = "makeMockClassA")
    @StarlarkConstructor
    public MockClassA makeMockClassA() {
      return new MockClassA();
    }
  }

  @Test
  public void testStarlarkCallableParameters() throws Exception {
    ImmutableMap<Category, ImmutableList<StarlarkDocPage>> objects = collect(MockClassD.class);
    assertThat(objects.get(Category.BUILTIN)).hasSize(1);
    StarlarkDocPage moduleDoc = objects.get(Category.BUILTIN).get(0);
    assertThat(moduleDoc.getDocumentation()).isEqualTo("MockClassD");
    assertThat(moduleDoc.getMembers()).hasSize(1);
    MemberDoc methodDoc = moduleDoc.getMembers().getFirst();
    assertThat(methodDoc.getDocumentation()).isEqualTo("MockClassD#test");
    assertThat(methodDoc.getSignature())
        .isEqualTo(
            "<a class=\"anchor\" href=\"../core/int.html\">int</a> MockClassD.test(a, b, *, c,"
                + " d=1)");
    assertThat(methodDoc.getParams()).hasSize(4);
  }

  @Test
  public void testStarlarkCallableParametersAndArgs() throws Exception {
    ImmutableMap<Category, ImmutableList<StarlarkDocPage>> objects = collect(MockClassF.class);
    assertThat(objects.get(Category.BUILTIN)).hasSize(1);
    StarlarkDocPage moduleDoc = objects.get(Category.BUILTIN).get(0);
    assertThat(moduleDoc.getDocumentation()).isEqualTo("MockClassF");
    assertThat(moduleDoc.getMembers()).hasSize(1);
    MemberDoc methodDoc = moduleDoc.getMembers().getFirst();
    assertThat(methodDoc.getDocumentation()).isEqualTo("MockClassF#test");
    assertThat(methodDoc.getSignature())
        .isEqualTo(
            "<a class=\"anchor\" href=\"../core/int.html\">int</a> "
                + "MockClassF.test(a, b, *myArgs, c, d=1)");
    assertThat(methodDoc.getParams()).hasSize(5);
  }

  @Test
  public void testStarlarkCallableParametersAndKwargs() throws Exception {
    ImmutableMap<Category, ImmutableList<StarlarkDocPage>> objects = collect(MockClassG.class);
    assertThat(objects.get(Category.BUILTIN)).hasSize(1);
    StarlarkDocPage moduleDoc = objects.get(Category.BUILTIN).get(0);
    assertThat(moduleDoc.getDocumentation()).isEqualTo("MockClassG");
    assertThat(moduleDoc.getMembers()).hasSize(1);
    MemberDoc methodDoc = moduleDoc.getMembers().getFirst();
    assertThat(methodDoc.getDocumentation()).isEqualTo("MockClassG#test");
    assertThat(methodDoc.getSignature())
        .isEqualTo(
            "<a class=\"anchor\" href=\"../core/int.html\">int</a> "
                + "MockClassG.test(a, b, *, c, d=1, **myKwargs)");
    assertThat(methodDoc.getParams()).hasSize(5);
  }

  @Test
  public void testStarlarkCallableParametersAndArgsAndKwargs() throws Exception {
    ImmutableMap<Category, ImmutableList<StarlarkDocPage>> objects = collect(MockClassH.class);
    assertThat(objects.get(Category.BUILTIN)).hasSize(1);
    StarlarkDocPage moduleDoc = objects.get(Category.BUILTIN).get(0);
    assertThat(moduleDoc.getDocumentation()).isEqualTo("MockClassH");
    assertThat(moduleDoc.getMembers()).hasSize(1);
    MemberDoc methodDoc = moduleDoc.getMembers().getFirst();
    assertThat(methodDoc.getDocumentation()).isEqualTo("MockClassH#test");
    assertThat(methodDoc.getSignature())
        .isEqualTo(
            "<a class=\"anchor\" href=\"../core/int.html\">int</a> "
                + "MockClassH.test(a, b, *myArgs, c, d=1, **myKwargs)");
    assertThat(methodDoc.getParams()).hasSize(6);
  }

  @Test
  public void testStarlarkUndocumentedParameters() throws Exception {
    ImmutableMap<Category, ImmutableList<StarlarkDocPage>> objects = collect(MockClassI.class);
    assertThat(objects.get(Category.BUILTIN)).hasSize(1);
    StarlarkDocPage moduleDoc = objects.get(Category.BUILTIN).get(0);
    assertThat(moduleDoc.getDocumentation()).isEqualTo("MockClassI");
    assertThat(moduleDoc.getMembers()).hasSize(1);
    MemberDoc methodDoc = moduleDoc.getMembers().getFirst();
    assertThat(methodDoc.getDocumentation()).isEqualTo("MockClassI#test");
    assertThat(methodDoc.getSignature())
        .isEqualTo(
            "<a class=\"anchor\" href=\"../core/int.html\">int</a> "
                + "MockClassI.test(a, b, *myArgs, c, d=1)");
    assertThat(methodDoc.getParams()).hasSize(5);
  }

  @Test
  public void testStarlarkGlobalLibraryCallable() throws Exception {
    StarlarkDocPage topLevel =
        StarlarkDocumentationCollector.getAllDocPages(expander, ImmutableList.of())
            .get(Category.GLOBAL_FUNCTION)
            .stream()
            .filter(p -> p.getTitle().equals(Environment.BZL.getTitle()))
            .findAny()
            .get();

    boolean foundGlobalLibrary = false;
    for (MemberDoc methodDoc : topLevel.getMembers()) {
      if (methodDoc.getName().equals("MockGlobalCallable")) {
        assertThat(methodDoc.getDocumentation()).isEqualTo("GlobalCallable documentation");
        assertThat(methodDoc.getSignature())
            .isEqualTo(
                "<a class=\"anchor\" href=\"../core/int.html\">int</a> "
                    + "MockGlobalCallable(a, b, *myArgs, c, d=1, **myKwargs)");
        foundGlobalLibrary = true;
        break;
      }
    }
    assertThat(foundGlobalLibrary).isTrue();
  }

  @Test
  public void testStarlarkCallableOverriding() throws Exception {
    ImmutableMap<Category, ImmutableList<StarlarkDocPage>> objects =
        collect(MockClassA.class, MockClassE.class);
    StarlarkDocPage moduleDoc =
        objects.get(Category.BUILTIN).stream()
            .filter(p -> p.getTitle().equals("MockClassE"))
            .findAny()
            .get();
    assertThat(moduleDoc.getDocumentation()).isEqualTo("MockClassE");
    assertThat(moduleDoc.getMembers()).hasSize(1);
    MemberDoc methodDoc = moduleDoc.getMembers().getFirst();
    assertThat(methodDoc.getDocumentation()).isEqualTo("MockClassA#get");
    assertThat(methodDoc.getSignature())
        .isEqualTo("<a class=\"anchor\" href=\"../core/int.html\">int</a> MockClassE.get()");
  }

  @Test
  public void testStarlarkContainerReturnTypesWithoutAnnotations() throws Exception {
    ImmutableMap<Category, ImmutableList<StarlarkDocPage>> objects =
        collect(MockClassWithContainerReturnValues.class);
    assertThat(objects.get(Category.BUILTIN)).hasSize(1);
    StarlarkDocPage moduleDoc = objects.get(Category.BUILTIN).get(0);
    ImmutableList<? extends MemberDoc> methods = moduleDoc.getMembers();

    List<String> signatures =
        methods.stream().map(m -> m.getSignature()).collect(Collectors.toList());
    assertThat(signatures).hasSize(5);
    assertThat(signatures)
        .contains(
            "<a class=\"anchor\" href=\"../builtins/depset.html\">depset</a> "
                + "MockClassWithContainerReturnValues.depset()");
    assertThat(signatures)
        .contains(
            "<a class=\"anchor\" href=\"../core/tuple.html\">tuple</a> "
                + "MockClassWithContainerReturnValues.tuple()");
    assertThat(signatures)
        .contains(
            "<a class=\"anchor\" href=\"../core/list.html\">list</a> "
                + "MockClassWithContainerReturnValues.immutable()");
    assertThat(signatures)
        .contains(
            "<a class=\"anchor\" href=\"../core/list.html\">list</a> "
                + "MockClassWithContainerReturnValues.mutable()");
    assertThat(signatures)
        .contains(
            "<a class=\"anchor\" href=\"../core/list.html\">sequence</a> "
                + "MockClassWithContainerReturnValues.starlark()");
  }

  @Test
  public void testDocumentedModuleTakesPrecedence() throws Exception {
    ImmutableMap<Category, ImmutableList<StarlarkDocPage>> objects =
        collect(
            PointsToCommonNameAndUndocumentedModule.class,
            MockClassCommonNameOne.class,
            MockClassCommonNameUndocumented.class);
    ImmutableList<MemberDoc> methods =
        objects.get(Category.BUILTIN).stream()
            .filter(p -> p.getTitle().equals("MockClassCommonName"))
            .findAny()
            .get()
            .getMembers();
    List<String> methodNames = methods.stream().map(m -> m.getName()).collect(Collectors.toList());
    assertThat(methodNames).containsExactly("one");
  }

  @Test
  public void testDocumentModuleSubclass() throws Exception {
    ImmutableMap<Category, ImmutableList<StarlarkDocPage>> objects =
        collect(
            PointsToCommonNameOneWithSubclass.class,
            MockClassCommonNameOne.class,
            SubclassOfMockClassCommonNameOne.class);
    ImmutableList<MemberDoc> methods =
        objects.get(Category.BUILTIN).stream()
            .filter(p -> p.getTitle().equals("MockClassCommonName"))
            .findAny()
            .get()
            .getMembers();
    List<String> methodNames = methods.stream().map(m -> m.getName()).collect(Collectors.toList());
    assertThat(methodNames).containsExactly("one", "two");
  }

  @Test
  public void testDocumentSelfcallConstructor() throws Exception {
    ImmutableMap<Category, ImmutableList<StarlarkDocPage>> objects =
        collect(MockClassA.class, MockClassWithSelfCallConstructor.class);
    ImmutableList<MemberDoc> methods =
        objects.get(Category.BUILTIN).stream()
            .filter(p -> p.getTitle().equals("MockClassA"))
            .findAny()
            .get()
            .getMembers();
    MemberDoc firstMethod = methods.getFirst();
    assertThat(firstMethod).isInstanceOf(AnnotStarlarkConstructorMethodDoc.class);

    List<String> methodNames = methods.stream().map(m -> m.getName()).collect(Collectors.toList());
    assertThat(methodNames).containsExactly("MockClassA", "get");
  }

  @Test
  public void testStardocProtoStructBasicFunctionality() throws Exception {
    ModuleInfo moduleInfo =
        getModuleInfo(
            "//pkg:test.bzl",
            """
            def _bar(a, *args, b = 42, **kwargs):
                '''Blah blah blah'''
                pass
            #: Foo module.
            foo = struct(bar = _bar)
            """);
    ImmutableMap<Category, ImmutableList<StarlarkDocPage>> objects = collect(moduleInfo);
    assertThat(objects.get(Category.TOP_LEVEL_MODULE)).hasSize(1);
    StarlarkDocPage moduleDoc = objects.get(Category.TOP_LEVEL_MODULE).getFirst();
    assertThat(moduleDoc.getDocumentation()).isEqualTo("Foo module.");
    assertThat(moduleDoc.getLoadStatement()).isEqualTo("load(\"//pkg:test.bzl\", \"foo\")");
    assertThat(moduleDoc.getSourceFile()).isEqualTo("pkg/test.bzl");
    assertThat(moduleDoc.getConstructor()).isNull();
    assertThat(moduleDoc.getMembers()).hasSize(1);
    MemberDoc methodDoc = moduleDoc.getMembers().getFirst();
    assertThat(methodDoc.getDocumentation()).isEqualTo("Blah blah blah");
    assertThat(methodDoc.getSignature()).isEqualTo("unknown foo.bar(a, *args, b=42, **kwargs)");
  }

  @Test
  public void testStardocProtoFunctionParamDocs() throws Exception {
    ModuleInfo moduleInfo =
        getModuleInfo(
            "//pkg:test.bzl",
            """
            def _func(name, *extra_names, answer=42, vals={}, **kwargs):
                '''Blah blah blah

                Args:
                  name: (string): Entity name.
                  *extra_names: Extra names.
                  answer: (int | None) Expected answer.
                  vals: (dict[string, int]) Dict of values.
                '''
                pass
            #: Foo module.
            foo = struct(func = _func)
            """);
    ImmutableMap<Category, ImmutableList<StarlarkDocPage>> objects = collect(moduleInfo);
    StarlarkDocPage moduleDoc = objects.get(Category.TOP_LEVEL_MODULE).getFirst();
    // Note that getParams returns parameters in calling convention order, with the varargs
    // parameter in the position immediately before kwargs. (Same as in Stardoc proto output, and in
    // AnnotStarlarkMethodDoc::getParams.)
    assertThat(moduleDoc.getMembers().getFirst().getParams().stream().map(ParamDoc::getName))
        .containsExactly("name", "answer", "vals", "extra_names", "kwargs")
        .inOrder();
    assertThat(moduleDoc.getMembers().getFirst().getParams().stream().map(ParamDoc::getKind))
        .containsExactly(
            ParamDoc.Kind.ORDINARY,
            ParamDoc.Kind.KEYWORD_ONLY,
            ParamDoc.Kind.KEYWORD_ONLY,
            ParamDoc.Kind.VARARGS,
            ParamDoc.Kind.KWARGS)
        .inOrder();
    assertThat(
            moduleDoc.getMembers().getFirst().getParams().stream().map(ParamDoc::getDocumentation))
        .containsExactly("Entity name.", "Expected answer.", "Dict of values.", "Extra names.", "")
        .inOrder();
    assertThat(moduleDoc.getMembers().getFirst().getParams().stream().map(ParamDoc::getType))
        .containsExactly(
            "<code><a class=\"anchor\" href=\"../core/string.html\">string</a></code>",
            "<code><a class=\"anchor\" href=\"../core/int.html\">int</a> | None</code>",
            "<code><a class=\"anchor\" href=\"../core/dict.html\">dict</a>[<a class=\"anchor\""
                + " href=\"../core/string.html\">string</a>, <a class=\"anchor\""
                + " href=\"../core/int.html\">int</a>]</code>",
            "",
            "")
        .inOrder();
    assertThat(
            moduleDoc.getMembers().getFirst().getParams().stream().map(ParamDoc::getDefaultValue))
        .containsExactly("", "42", "{}", "", "")
        .inOrder();
  }

  @Test
  public void testStardocProtoFunctionReturnsDocs() throws Exception {
    ModuleInfo moduleInfo =
        getModuleInfo(
            "//pkg:test.bzl",
            """
            def _func(**kwargs):
                '''Blah blah blah

                Returns:
                  (set[string]) The answers. Not always accurate.
                '''
                pass
            #: Foo module.
            foo = struct(func = _func)
            """);
    ImmutableMap<Category, ImmutableList<StarlarkDocPage>> objects = collect(moduleInfo);
    StarlarkDocPage moduleDoc = objects.get(Category.TOP_LEVEL_MODULE).getFirst();
    assertThat(moduleDoc.getMembers().getFirst().getReturnType())
        .isEqualTo(
            "<code><a class=\"anchor\" href=\"../core/set.html\">set</a>[<a class=\"anchor\""
                + " href=\"../core/string.html\">string</a>]</code>");
    assertThat(moduleDoc.getMembers().getFirst().getReturnsStanza())
        .isEqualTo("The answers. Not always accurate.");
    // Unused for Stardoc proto-based documentation.
    assertThat(moduleDoc.getMembers().getFirst().getReturnTypeExtraMessage()).isEmpty();
  }

  @Test
  public void testStardocProtoFunctionDeprecatedDocs() throws Exception {
    ModuleInfo moduleInfo =
        getModuleInfo(
            "//pkg:test.bzl",
            """
            def _func(**kwargs):
                '''Blah blah blah

                Deprecated:
                  Don't use this function!
                '''
                pass
            #: Foo module.
            foo = struct(func = _func)
            """);
    ImmutableMap<Category, ImmutableList<StarlarkDocPage>> objects = collect(moduleInfo);
    StarlarkDocPage moduleDoc = objects.get(Category.TOP_LEVEL_MODULE).getFirst();
    assertThat(moduleDoc.getMembers().getFirst().getDeprecatedStanza())
        .isEqualTo("Don't use this function!");
  }

  @Test
  public void testStardocProtoStructMembersSorted() throws Exception {
    ModuleInfo moduleInfo =
        getModuleInfo(
            "//:test.bzl",
            """
            #: Foo module.
            foo = struct(c = lambda x: x, a = lambda x: x, b = lambda x: x)
            """);
    ImmutableMap<Category, ImmutableList<StarlarkDocPage>> objects = collect(moduleInfo);
    assertThat(objects.get(Category.TOP_LEVEL_MODULE)).hasSize(1);
    StarlarkDocPage moduleDoc = objects.get(Category.TOP_LEVEL_MODULE).getFirst();
    assertThat(moduleDoc.getMembers().stream().map(MemberDoc::getName))
        .containsExactly("a", "b", "c")
        .inOrder();
  }

  @Test
  public void testStardocProtoMultipleStructsInOneFile() throws Exception {
    ModuleInfo moduleInfo =
        getModuleInfo(
            "//:test.bzl",
            """
            #: Foo module.
            foo = struct(bar = lambda x: x)
            #: Baz module.
            baz = struct(qux = lambda x: x)
            """);
    ImmutableMap<Category, ImmutableList<StarlarkDocPage>> objects = collect(moduleInfo);
    assertThat(objects.get(Category.TOP_LEVEL_MODULE).stream().map(StarlarkDocPage::getName))
        .containsExactly("foo", "baz");
    assertThat(
            objects.get(Category.TOP_LEVEL_MODULE).stream().map(StarlarkDocPage::getDocumentation))
        .containsExactly("Foo module.", "Baz module.");
  }

  @Test
  public void testStardocProtoMultipleFiles() throws Exception {
    ModuleInfo fooModuleInfo =
        getModuleInfo(
            "//:foo.bzl",
            """
            #: Foo module.
            foo = struct(bar = lambda x: x)
            """);
    ModuleInfo bazModuleInfo =
        getModuleInfo(
            "//:baz.bzl",
            """
            #: Bar module.
            baz = struct(qux = lambda x: x)
            """);
    ImmutableMap<Category, ImmutableList<StarlarkDocPage>> objects =
        collect(fooModuleInfo, bazModuleInfo);
    assertThat(objects.get(Category.TOP_LEVEL_MODULE).stream().map(StarlarkDocPage::getName))
        .containsExactly("foo", "baz");
    assertThat(objects.get(Category.TOP_LEVEL_MODULE).stream().map(StarlarkDocPage::getSourceFile))
        .containsExactly("foo.bzl", "baz.bzl");
  }

  @Test
  public void testStardocProtoProviderBasicFunctionality() throws Exception {
    ModuleInfo moduleInfo =
        getModuleInfo(
            "//pkg:my_info.bzl",
            """
            MyInfo = provider(doc = "My info.", fields = ["a", "b"])
            """);
    ImmutableMap<Category, ImmutableList<StarlarkDocPage>> objects = collect(moduleInfo);
    assertThat(objects.get(Category.PROVIDER)).hasSize(1);
    StarlarkDocPage providerDoc = objects.get(Category.PROVIDER).getFirst();
    assertThat(providerDoc.getDocumentation()).isEqualTo("My info.");
    assertThat(providerDoc.getLoadStatement()).isEqualTo("load(\"//pkg:my_info.bzl\", \"MyInfo\")");
    assertThat(providerDoc.getSourceFile()).isEqualTo("pkg/my_info.bzl");
    assertThat(providerDoc.getConstructor()).isNull();
    assertThat(providerDoc.getMembers()).hasSize(2);
    assertThat(providerDoc.getMembers().stream().map(MemberDoc::getName))
        .containsExactly("a", "b")
        .inOrder();
    assertThat(providerDoc.getMembers().stream().map(MemberDoc::isCallable))
        .containsExactly(false, false)
        .inOrder();
    assertThat(providerDoc.getMembers().stream().map(MemberDoc::getReturnType))
        .containsExactly("unknown", "unknown")
        .inOrder();
    assertThat(providerDoc.getMembers().stream().map(MemberDoc::getDocumentation))
        .containsExactly("", "")
        .inOrder();
  }

  @Test
  public void testStardocProtoProviderFieldDocs() throws Exception {
    ModuleInfo moduleInfo =
        getModuleInfo(
            "//pkg:my_info.bzl",
            """
            MyInfo = provider(
                doc = "My info.",
                fields = {"b": "Field B.", "a": "(list[string] | None) Field A."},
            )
            """);
    ImmutableMap<Category, ImmutableList<StarlarkDocPage>> objects = collect(moduleInfo);
    assertThat(objects.get(Category.PROVIDER)).hasSize(1);
    StarlarkDocPage providerDoc = objects.get(Category.PROVIDER).getFirst();
    assertThat(providerDoc.getMembers()).hasSize(2);
    assertThat(providerDoc.getMembers().stream().map(MemberDoc::getName))
        .containsExactly("a", "b") // sorted!
        .inOrder();
    assertThat(providerDoc.getMembers().stream().map(MemberDoc::getReturnType))
        .containsExactly(
            "<code><a class=\"anchor\" href=\"../core/list.html\">list</a>[<a class=\"anchor\""
                + " href=\"../core/string.html\">string</a>] | None</code>",
            "unknown")
        .inOrder();
    assertThat(providerDoc.getMembers().stream().map(MemberDoc::getDocumentation))
        .containsExactly("Field A.", "Field B.")
        .inOrder();
  }

  @Test
  public void testStardocProtoProviderInit() throws Exception {
    ModuleInfo moduleInfo =
        getModuleInfo(
            "//pkg:my_info.bzl",
            """
            def _init_my_info(a, b=1, **kwargs):
                '''Initializes a MyInfo instance.'''
                return {"a": a + b}
            MyInfo, _new_my_info = provider(doc = "My info.", fields = ["a"], init = _init_my_info)
            """);
    ImmutableMap<Category, ImmutableList<StarlarkDocPage>> objects = collect(moduleInfo);
    assertThat(objects.get(Category.PROVIDER)).hasSize(1);
    StarlarkDocPage providerDoc = objects.get(Category.PROVIDER).getFirst();
    assertThat(providerDoc.getMembers().stream().map(MemberDoc::getName))
        .containsExactly("MyInfo", "a")
        .inOrder();
    MemberDoc constructor = providerDoc.getConstructor();
    assertThat(constructor.getName()).isEqualTo("MyInfo");
    assertThat(constructor.getDocumentation()).isEqualTo("Initializes a MyInfo instance.");
    assertThat(constructor.getSignature())
        .isEqualTo(
            "<code><a class=\"anchor\" href=\"../providers/MyInfo.html\">MyInfo</a></code>"
                + " MyInfo(a, b=1, **kwargs)");
    assertThat(constructor.getParams().stream().map(ParamDoc::getName))
        .containsExactly("a", "b", "kwargs")
        .inOrder();
  }

  private ImmutableMap<Category, ImmutableList<StarlarkDocPage>> collect(Class<?>... classObjects)
      throws IOException {
    return StarlarkDocumentationCollector.collectDocPages(
        expander, ImmutableList.copyOf(classObjects), ImmutableList.of());
  }

  private ImmutableMap<Category, ImmutableList<StarlarkDocPage>> collect(
      ModuleInfo... apiStardocProtos) throws IOException, ClassPathException {
    return StarlarkDocumentationCollector.collectDocPages(
        expander, getCoreStarlarkClasses(), ImmutableList.copyOf(apiStardocProtos));
  }

  private Set<Class<?>> getCoreStarlarkClasses() throws ClassPathException {
    return Classpath.findClasses("net/starlark/java");
  }

  /**
   * Parses and evaluates a .bzl file, extracts documentation from it into a Stardoc proto, and
   * writes the binary proto data to a new scratch file.
   */
  private ModuleInfo getModuleInfo(String bzlLabelString, String... lines) throws Exception {
    BazelEvaluationTestCase ev = new BazelEvaluationTestCase(bzlLabelString);
    Module moduleForCompilation = ev.newModule();
    Label bzlLabel = Label.parseCanonical(bzlLabelString);
    ev.setThreadOwner(keyForBuild(bzlLabel));
    ParserInput input = ParserInput.fromLines(lines);
    StarlarkFile file = StarlarkFile.parse(input, FileOptions.DEFAULT);
    Program program = Program.compileFile(file, moduleForCompilation);
    Module moduleForEvaluation = ev.newModule(program);
    BzlLoadFunction.execAndExport(
        program, bzlLabel, ev.getEventHandler(), moduleForEvaluation, ev.getStarlarkThread());
    ModuleInfoExtractor extractor = new ModuleInfoExtractor(name -> true, LabelRenderer.DEFAULT);
    return extractor.extractFrom(moduleForEvaluation);
  }
}
