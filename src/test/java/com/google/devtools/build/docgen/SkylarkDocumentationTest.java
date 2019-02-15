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
import static com.google.common.truth.Truth.assertWithMessage;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.docgen.skylark.SkylarkMethodDoc;
import com.google.devtools.build.docgen.skylark.SkylarkModuleDoc;
import com.google.devtools.build.lib.analysis.skylark.SkylarkRuleContext;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.skylark.util.SkylarkTestCase;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkGlobalLibrary;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.SkylarkDict;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkList.MutableList;
import com.google.devtools.build.lib.syntax.SkylarkList.Tuple;
import com.google.devtools.build.lib.syntax.SkylarkSemantics;
import com.google.devtools.build.lib.syntax.util.EvaluationTestCase;
import com.google.devtools.build.lib.util.Classpath;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for Skylark documentation.
 */
@RunWith(JUnit4.class)
public class SkylarkDocumentationTest extends SkylarkTestCase {

  private static final ImmutableList<String> DEPRECATED_UNDOCUMENTED_TOP_LEVEL_SYMBOLS =
      ImmutableList.of("Actions");

  @Before
  public final void createBuildFile() throws Exception {
    scratch.file("foo/BUILD",
        "genrule(name = 'foo',",
        "  cmd = 'dummy_cmd',",
        "  srcs = ['a.txt', 'b.img'],",
        "  tools = ['t.exe'],",
        "  outs = ['c.txt'])");
  }

  @Override
  protected EvaluationTestCase createEvaluationTestCase(SkylarkSemantics semantics) {
    return new EvaluationTestCase();
  }

  @Test
  public void testSkylarkRuleClassBuiltInItemsAreDocumented() throws Exception {
    checkSkylarkTopLevelEnvItemsAreDocumented(ev.getEnvironment());
  }

  @Test
  public void testSkylarkRuleImplementationBuiltInItemsAreDocumented() throws Exception {
    // TODO(bazel-team): fix documentation for built in java objects coming from modules.
    checkSkylarkTopLevelEnvItemsAreDocumented(ev.getEnvironment());
  }

  @SuppressWarnings("unchecked")
  private void checkSkylarkTopLevelEnvItemsAreDocumented(Environment env) throws Exception {
    Map<String, String> docMap = new HashMap<>();
    Map<String, SkylarkModuleDoc> modules =
        SkylarkDocumentationCollector.collectModules(
            Classpath.findClasses(SkylarkDocumentationProcessor.MODULES_PACKAGE_PREFIX));
    SkylarkModuleDoc topLevel =
        modules.remove(SkylarkDocumentationCollector.getTopLevelModule().name());
    for (SkylarkMethodDoc method : topLevel.getMethods()) {
      docMap.put(method.getName(), method.getDocumentation());
    }
    for (Map.Entry<String, SkylarkModuleDoc> entry : modules.entrySet()) {
      docMap.put(entry.getKey(), entry.getValue().getDocumentation());
    }

    List<String> undocumentedItems = new ArrayList<>();
    // All built in variables are registered in the Skylark global environment.
    for (String varname : env.getVariableNames()) {
      if (docMap.containsKey(varname)) {
        if (docMap.get(varname).isEmpty()) {
          undocumentedItems.add(varname);
        }
      } else {
        undocumentedItems.add(varname);
      }
    }
    assertWithMessage("Undocumented Skylark Environment items: " + undocumentedItems)
        .that(undocumentedItems)
        .containsExactlyElementsIn(DEPRECATED_UNDOCUMENTED_TOP_LEVEL_SYMBOLS);
  }

  // TODO(bazel-team): come up with better Skylark specific tests.
  @Test
  public void testDirectJavaMethodsAreGenerated() throws Exception {
    assertThat(collect(SkylarkRuleContext.class)).isNotEmpty();
  }

  /** MockClassA */
  @SkylarkModule(name = "MockClassA", doc = "MockClassA")
  private static class MockClassA {
    @SkylarkCallable(name = "get", doc = "MockClassA#get")
    public Integer get() {
      return 0;
    }
  }

  /** MockClassD */
  @SkylarkModule(name = "MockClassD", doc = "MockClassD")
  private static class MockClassD {
    @SkylarkCallable(
      name = "test",
      doc = "MockClassD#test",
      parameters = {
        @Param(name = "a"),
        @Param(name = "b"),
        @Param(name = "c", named = true, positional = false),
        @Param(name = "d", named = true, positional = false, defaultValue = "1"),
      }
    )
    public Integer test(int a, int b, int c, int d) {
      return 0;
    }
  }

  /** MockClassE */
  @SkylarkModule(name = "MockClassE", doc = "MockClassE")
  private static class MockClassE extends MockClassA {
    @Override
    public Integer get() {
      return 1;
    }
  }

  /** MockClassF */
  @SkylarkModule(name = "MockClassF", doc = "MockClassF")
  private static class MockClassF {
    @SkylarkCallable(
      name = "test",
      doc = "MockClassF#test",
      parameters = {
        @Param(name = "a", named = false, positional = true),
        @Param(name = "b", named = true, positional = true),
        @Param(name = "c", named = true, positional = false),
        @Param(name = "d", named = true, positional = false, defaultValue = "1"),
      },
      extraPositionals = @Param(name = "myArgs")
    )
    public Integer test(int a, int b, int c, int d, SkylarkList<?> args) {
      return 0;
    }
  }

  /** MockClassG */
  @SkylarkModule(name = "MockClassG", doc = "MockClassG")
  private static class MockClassG {
    @SkylarkCallable(
      name = "test",
      doc = "MockClassG#test",
      parameters = {
        @Param(name = "a", named = false, positional = true),
        @Param(name = "b", named = true, positional = true),
        @Param(name = "c", named = true, positional = false),
        @Param(name = "d", named = true, positional = false, defaultValue = "1"),
      },
      extraKeywords = @Param(name = "myKwargs")
    )
    public Integer test(int a, int b, int c, int d, SkylarkDict<?, ?> kwargs) {
      return 0;
    }
  }

  /** MockClassH */
  @SkylarkModule(name = "MockClassH", doc = "MockClassH")
  private static class MockClassH {
    @SkylarkCallable(
      name = "test",
      doc = "MockClassH#test",
      parameters = {
        @Param(name = "a", named = false, positional = true),
        @Param(name = "b", named = true, positional = true),
        @Param(name = "c", named = true, positional = false),
        @Param(name = "d", named = true, positional = false, defaultValue = "1"),
      },
      extraPositionals = @Param(name = "myArgs"),
      extraKeywords = @Param(name = "myKwargs")
    )
    public Integer test(int a, int b, int c, int d, SkylarkList<?> args, SkylarkDict<?, ?> kwargs) {
      return 0;
    }
  }

  /**
   * MockGlobalLibrary. While nothing directly depends on it, a test method in
   * SkylarkDocumentationTest checks all of the classes under a wide classpath and ensures this one
   * shows up.
   */
  @SkylarkGlobalLibrary
  private static class MockGlobalLibrary {
    @SkylarkCallable(
        name = "MockGlobalCallable",
        doc = "GlobalCallable documentation",
        parameters = {
            @Param(name = "a", named = false, positional = true),
            @Param(name = "b", named = true, positional = true),
            @Param(name = "c", named = true, positional = false),
            @Param(name = "d", named = true, positional = false, defaultValue = "1"),
        },
        extraPositionals = @Param(name = "myArgs"),
        extraKeywords = @Param(name = "myKwargs")
    )
    public Integer test(int a, int b, int c, int d, SkylarkList<?> args, SkylarkDict<?, ?> kwargs) {
      return 0;
    }
  }

  /** MockClassWithContainerReturnValues */
  @SkylarkModule(name = "MockClassWithContainerReturnValues",
      doc = "MockClassWithContainerReturnValues")
  private static class MockClassWithContainerReturnValues {

    @SkylarkCallable(name = "depset", doc = "depset")
    public NestedSet<Integer> getNestedSet() {
      return null;
    }

    @SkylarkCallable(name = "tuple", doc = "tuple")
    public Tuple<Integer> getTuple() {
      return null;
    }

    @SkylarkCallable(name = "immutable", doc = "immutable")
    public ImmutableList<Integer> getImmutableList() {
      return null;
    }

    @SkylarkCallable(name = "mutable", doc = "mutable")
    public MutableList<Integer> getMutableList() {
      return null;
    }

    @SkylarkCallable(name = "skylark", doc = "skylark")
    public SkylarkList<Integer> getSkylarkList() {
      return null;
    }
  }

  /** MockClassCommonNameOne */
  @SkylarkModule(name = "MockClassCommonName",
      doc = "MockClassCommonName")
  private static class MockClassCommonNameOne {

    @SkylarkCallable(name = "one", doc = "one")
    public Integer one() {
      return 1;
    }
  }

  /** SubclassOfMockClassCommonNameOne */
  @SkylarkModule(name = "MockClassCommonName",
      doc = "MockClassCommonName")
  private static class SubclassOfMockClassCommonNameOne extends MockClassCommonNameOne {

    @SkylarkCallable(name = "two", doc = "two")
    public Integer two() {
      return 1;
    }
  }

  /** PointsToCommonNameOneWithSubclass */
  @SkylarkModule(name = "PointsToCommonNameOneWithSubclass",
      doc = "PointsToCommonNameOneWithSubclass")
  private static class PointsToCommonNameOneWithSubclass {
    @SkylarkCallable(name = "one", doc = "one")
    public MockClassCommonNameOne getOne() {
      return null;
    }

    @SkylarkCallable(name = "one_subclass", doc = "one_subclass")
    public SubclassOfMockClassCommonNameOne getOneSubclass() {
      return null;
    }
  }

  /** MockClassCommonNameOneUndocumented */
  @SkylarkModule(name = "MockClassCommonName",
      documented = false,
      doc = "")
  private static class MockClassCommonNameUndocumented {

    @SkylarkCallable(name = "two", doc = "two")
    public Integer two() {
      return 1;
    }
  }

  /** PointsToCommonNameAndUndocumentedModule */
  @SkylarkModule(name = "PointsToCommonNameAndUndocumentedModule",
      doc = "PointsToCommonNameAndUndocumentedModule")
  private static class PointsToCommonNameAndUndocumentedModule {
    @SkylarkCallable(name = "one", doc = "one")
    public MockClassCommonNameOne getOne() {
      return null;
    }

    @SkylarkCallable(name = "undocumented_module", doc = "undocumented_module")
    public MockClassCommonNameUndocumented getUndocumented() {
      return null;
    }
  }

  @Test
  public void testSkylarkCallableParameters() throws Exception {
    Map<String, SkylarkModuleDoc> objects = collect(MockClassD.class);
    SkylarkModuleDoc moduleDoc = objects.get("MockClassD");
    assertThat(moduleDoc.getDocumentation()).isEqualTo("MockClassD");
    assertThat(moduleDoc.getMethods()).hasSize(1);
    SkylarkMethodDoc methodDoc = moduleDoc.getMethods().iterator().next();
    assertThat(methodDoc.getDocumentation()).isEqualTo("MockClassD#test");
    assertThat(methodDoc.getSignature())
        .isEqualTo(
            "<a class=\"anchor\" href=\"int.html\">int</a> MockClassD.test(a, b, *, c, d=1)");
    assertThat(methodDoc.getParams()).hasSize(4);
  }

  @Test
  public void testSkylarkCallableParametersAndArgs() throws Exception {
    Map<String, SkylarkModuleDoc> objects = collect(MockClassF.class);
    SkylarkModuleDoc moduleDoc = objects.get("MockClassF");
    assertThat(moduleDoc.getDocumentation()).isEqualTo("MockClassF");
    assertThat(moduleDoc.getMethods()).hasSize(1);
    SkylarkMethodDoc methodDoc = moduleDoc.getMethods().iterator().next();
    assertThat(methodDoc.getDocumentation()).isEqualTo("MockClassF#test");
    assertThat(methodDoc.getSignature())
        .isEqualTo(
            "<a class=\"anchor\" href=\"int.html\">int</a> "
                + "MockClassF.test(a, b, *, c, d=1, *myArgs)");
    assertThat(methodDoc.getParams()).hasSize(5);
  }

  @Test
  public void testSkylarkCallableParametersAndKwargs() throws Exception {
    Map<String, SkylarkModuleDoc> objects = collect(MockClassG.class);
    SkylarkModuleDoc moduleDoc = objects.get("MockClassG");
    assertThat(moduleDoc.getDocumentation()).isEqualTo("MockClassG");
    assertThat(moduleDoc.getMethods()).hasSize(1);
    SkylarkMethodDoc methodDoc = moduleDoc.getMethods().iterator().next();
    assertThat(methodDoc.getDocumentation()).isEqualTo("MockClassG#test");
    assertThat(methodDoc.getSignature())
        .isEqualTo(
            "<a class=\"anchor\" href=\"int.html\">int</a> "
                + "MockClassG.test(a, b, *, c, d=1, **myKwargs)");
    assertThat(methodDoc.getParams()).hasSize(5);
  }

  @Test
  public void testSkylarkCallableParametersAndArgsAndKwargs() throws Exception {
    Map<String, SkylarkModuleDoc> objects = collect(MockClassH.class);
    SkylarkModuleDoc moduleDoc = objects.get("MockClassH");
    assertThat(moduleDoc.getDocumentation()).isEqualTo("MockClassH");
    assertThat(moduleDoc.getMethods()).hasSize(1);
    SkylarkMethodDoc methodDoc = moduleDoc.getMethods().iterator().next();
    assertThat(methodDoc.getDocumentation()).isEqualTo("MockClassH#test");
    assertThat(methodDoc.getSignature())
        .isEqualTo(
            "<a class=\"anchor\" href=\"int.html\">int</a> "
                + "MockClassH.test(a, b, *, c, d=1, *myArgs, **myKwargs)");
    assertThat(methodDoc.getParams()).hasSize(6);
  }

  @Test
  public void testSkylarkGlobalLibraryCallable() throws Exception {
    Map<String, SkylarkModuleDoc> modules =
        SkylarkDocumentationCollector.collectModules(
            Classpath.findClasses(SkylarkDocumentationProcessor.MODULES_PACKAGE_PREFIX));
    SkylarkModuleDoc topLevel =
        modules.remove(SkylarkDocumentationCollector.getTopLevelModule().name());

    boolean foundGlobalLibrary = false;
    for (SkylarkMethodDoc methodDoc : topLevel.getMethods()) {
      if (methodDoc.getName().equals("MockGlobalCallable")) {
        assertThat(methodDoc.getDocumentation()).isEqualTo("GlobalCallable documentation");
        assertThat(methodDoc.getSignature())
            .isEqualTo(
                "<a class=\"anchor\" href=\"int.html\">int</a> "
                    + "MockGlobalCallable(a, b, *, c, d=1, *myArgs, **myKwargs)");
        foundGlobalLibrary = true;
        break;
      }
    }
    assertThat(foundGlobalLibrary).isTrue();
  }


  @Test
  public void testSkylarkCallableOverriding() throws Exception {
    Map<String, SkylarkModuleDoc> objects =
        collect(ImmutableList.of(MockClassA.class, MockClassE.class));
    SkylarkModuleDoc moduleDoc = objects.get("MockClassE");
    assertThat(moduleDoc.getDocumentation()).isEqualTo("MockClassE");
    assertThat(moduleDoc.getMethods()).hasSize(1);
    SkylarkMethodDoc methodDoc = moduleDoc.getMethods().iterator().next();
    assertThat(methodDoc.getDocumentation()).isEqualTo("MockClassA#get");
    assertThat(methodDoc.getSignature())
        .isEqualTo("<a class=\"anchor\" href=\"int.html\">int</a> MockClassE.get()");
  }

  @Test
  public void testSkylarkContainerReturnTypesWithoutAnnotations() throws Exception {
    Map<String, SkylarkModuleDoc> objects = collect(MockClassWithContainerReturnValues.class);
    assertThat(objects).containsKey("MockClassWithContainerReturnValues");
    Collection<SkylarkMethodDoc> methods =
        objects.get("MockClassWithContainerReturnValues").getMethods();

    List<String> signatures =
        methods.stream().map(m -> m.getSignature()).collect(Collectors.toList());
    assertThat(signatures).hasSize(5);
    assertThat(signatures)
        .contains(
            "<a class=\"anchor\" href=\"depset.html\">depset</a> "
                + "MockClassWithContainerReturnValues.depset()");
    assertThat(signatures)
    .contains(
        "<a class=\"anchor\" href=\"list.html\">tuple</a> "
            + "MockClassWithContainerReturnValues.tuple()");
    assertThat(signatures)
        .contains(
            "<a class=\"anchor\" href=\"list.html\">list</a> "
                + "MockClassWithContainerReturnValues.immutable()");
    assertThat(signatures)
        .contains(
            "<a class=\"anchor\" href=\"list.html\">list</a> "
                + "MockClassWithContainerReturnValues.mutable()");
    assertThat(signatures)
        .contains(
            "<a class=\"anchor\" href=\"list.html\">sequence</a> "
                + "MockClassWithContainerReturnValues.skylark()");
  }

  @Test
  public void testDocumentedModuleTakesPrecedence() throws Exception {
    Map<String, SkylarkModuleDoc> objects =
        collect(
            ImmutableList.of(
                PointsToCommonNameAndUndocumentedModule.class,
                MockClassCommonNameOne.class,
                MockClassCommonNameUndocumented.class));
    Collection<SkylarkMethodDoc> methods =
        objects.get("MockClassCommonName").getMethods();
    List<String> methodNames =
        methods.stream().map(m -> m.getName()).collect(Collectors.toList());
    assertThat(methodNames).containsExactly("one");
  }

  @Test
  public void testDocumentModuleSubclass() {
    Map<String, SkylarkModuleDoc> objects =
        collect(
            ImmutableList.of(
                PointsToCommonNameOneWithSubclass.class,
                MockClassCommonNameOne.class,
                SubclassOfMockClassCommonNameOne.class));
    Collection<SkylarkMethodDoc> methods =
        objects.get("MockClassCommonName").getMethods();
    List<String> methodNames =
        methods.stream().map(m -> m.getName()).collect(Collectors.toList());
    assertThat(methodNames).containsExactly("one", "two");
  }

  private Map<String, SkylarkModuleDoc> collect(Iterable<Class<?>> classObjects) {
    return SkylarkDocumentationCollector.collectModules(classObjects);
  }

  private Map<String, SkylarkModuleDoc> collect(Class<?> classObject) {
    return collect(ImmutableList.of(classObject));
  }
}
