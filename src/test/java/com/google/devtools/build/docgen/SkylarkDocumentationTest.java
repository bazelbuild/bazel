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

import com.google.common.base.Function;
import com.google.common.collect.Iterables;
import com.google.devtools.build.docgen.skylark.SkylarkBuiltinMethodDoc;
import com.google.devtools.build.docgen.skylark.SkylarkJavaMethodDoc;
import com.google.devtools.build.docgen.skylark.SkylarkMethodDoc;
import com.google.devtools.build.docgen.skylark.SkylarkModuleDoc;
import com.google.devtools.build.lib.rules.SkylarkRuleContext;
import com.google.devtools.build.lib.skylark.util.SkylarkTestCase;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.util.EvaluationTestCase;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.TreeMap;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for Skylark documentation.
 */
@RunWith(JUnit4.class)
public class SkylarkDocumentationTest extends SkylarkTestCase {

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
  protected EvaluationTestCase createEvaluationTestCase() {
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
  private void checkSkylarkTopLevelEnvItemsAreDocumented(Environment env) {
    Map<String, String> docMap = new HashMap<>();
    Map<String, SkylarkModuleDoc> modules = SkylarkDocumentationCollector.collectModules();
    SkylarkModuleDoc topLevel =
        modules.remove(SkylarkDocumentationCollector.getTopLevelModule().name());
    for (Entry<String, SkylarkBuiltinMethodDoc> entry : topLevel.getBuiltinMethods().entrySet()) {
      docMap.put(entry.getKey(), entry.getValue().getDocumentation());
    }
    for (Entry<String, SkylarkModuleDoc> entry : modules.entrySet()) {
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
        .that(undocumentedItems).isEmpty();
  }

  // TODO(bazel-team): come up with better Skylark specific tests.
  @Test
  public void testDirectJavaMethodsAreGenerated() throws Exception {
    assertThat(collect(SkylarkRuleContext.class)).isNotEmpty();
  }

  /** MockClassA */
  @SkylarkModule(name = "MockClassA", doc = "MockClassA")
  private static class MockClassA {
    @SkylarkCallable(doc = "MockClassA#get")
    public Integer get() {
      return 0;
    }
  }

  /** MockClassB */
  @SkylarkModule(name = "MockClassB", doc = "MockClassB")
  private static class MockClassB {
    @SkylarkCallable(doc = "MockClassB#get")
    public MockClassA get() {
      return new MockClassA();
    }
  }

  /** MockClassC */
  @SkylarkModule(name = "MockClassC", doc = "MockClassC")
  private static class MockClassC extends MockClassA {
    @SkylarkCallable(doc = "MockClassC#get2")
    public Integer get2() {
      return 0;
    }
  }

  /** MockClassD */
  @SkylarkModule(name = "MockClassD", doc = "MockClassD")
  private static class MockClassD {
    @SkylarkCallable(
      doc = "MockClassD#test",
      mandatoryPositionals = 1,
      parameters = {
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

  @Test
  public void testSkylarkJavaInterfaceExplorerOnSimpleClass() throws Exception {
    Map<String, SkylarkModuleDoc> objects = collect(MockClassA.class);
    assertThat(extractMethods(Iterables.getOnlyElement(objects.values())
        .getJavaMethods())).containsExactly(MockClassA.class.getMethod("get"));
  }

  @Test
  public void testSkylarkJavaInterfaceExplorerFindsClassFromReturnValue() throws Exception {
    Map<String, SkylarkModuleDoc> objects = collect(MockClassB.class);
    assertThat(extractMethods(
        objects.get("MockClassA").getJavaMethods())).containsExactly(
        MockClassA.class.getMethod("get"));
  }

  @Test
  public void testSkylarkJavaInterfaceExplorerFindsAllMethodsOnSubClass() throws Exception {
    Map<String, SkylarkModuleDoc> objects = collect(MockClassC.class);
    assertThat(extractMethods(Iterables.getOnlyElement(objects.values())
        .getJavaMethods())).containsExactly(
        MockClassA.class.getMethod("get"), MockClassC.class.getMethod("get2"));
  }

  @Test
  public void testSkylarkCallableParameters() throws Exception {
    Map<String, SkylarkModuleDoc> objects = collect(MockClassD.class);
    assertThat(objects).hasSize(1);
    assertThat(objects).containsKey("MockClassD");
    SkylarkModuleDoc moduleDoc = objects.get("MockClassD");
    assertThat(moduleDoc.getDocumentation()).isEqualTo("MockClassD");
    assertThat(moduleDoc.getMethods()).hasSize(1);
    SkylarkMethodDoc methodDoc = moduleDoc.getMethods().iterator().next();
    assertThat(methodDoc.getDocumentation()).isEqualTo("MockClassD#test");
    assertThat(methodDoc.getSignature())
        .isEqualTo(
            "<a class=\"anchor\" href=\"int.html\">int</a> "
                + "MockClassD.test(arg0:<a class=\"anchor\" href=\"int.html\">int</a>, "
                + "b, *, c, d=1)");
    assertThat(methodDoc.getParams()).hasSize(3);
  }

  @Test
  public void testSkylarkCallableOverriding() throws Exception {
    Map<String, SkylarkModuleDoc> objects = collect(MockClassE.class);
    assertThat(objects).hasSize(1);
    assertThat(objects).containsKey("MockClassE");
    SkylarkModuleDoc moduleDoc = objects.get("MockClassE");
    assertThat(moduleDoc.getDocumentation()).isEqualTo("MockClassE");
    assertThat(moduleDoc.getMethods()).hasSize(1);
    SkylarkMethodDoc methodDoc = moduleDoc.getMethods().iterator().next();
    assertThat(methodDoc.getDocumentation()).isEqualTo("MockClassA#get");
    assertThat(methodDoc.getSignature())
        .isEqualTo("<a class=\"anchor\" href=\"int.html\">int</a> MockClassE.get()");
  }

  private Iterable<Method> extractMethods(Collection<SkylarkJavaMethodDoc> methods) {
    return Iterables.transform(methods, new Function<SkylarkJavaMethodDoc, Method>() {
      @Override
      public Method apply(SkylarkJavaMethodDoc input) {
        return input.getMethod();
      }
    });
  }

  private Map<String, SkylarkModuleDoc> collect(Class<?> classObject) {
    Map<String, SkylarkModuleDoc> modules = new TreeMap<>();
    SkylarkDocumentationCollector.collectJavaObjects(
        classObject.getAnnotation(SkylarkModule.class), classObject, modules);
    return modules;
  }
}
