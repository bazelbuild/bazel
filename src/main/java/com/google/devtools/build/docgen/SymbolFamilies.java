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
package com.google.devtools.build.docgen;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Lists;
import com.google.devtools.build.docgen.starlark.StarlarkBuiltinDoc;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.util.Classpath.ClassPathException;
import com.google.devtools.build.skydoc.fakebuildapi.FakeApi;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.List;
import java.util.Map;
import net.starlark.java.eval.Starlark;

/**
 * A helper class that collects Starlark Api symbols including top level modules, native rules and
 * builtin types.
 */
public class SymbolFamilies {
  private final ImmutableList<RuleDocumentation> nativeRules;
  private final ImmutableMap<String, StarlarkBuiltinDoc> types;

  // Mappings between Starlark names and Starlark entities generated from the fakebuildapi.
  private final ImmutableMap<String, Object> globals;
  private final ImmutableMap<String, Object> bzlGlobals;

  public SymbolFamilies(
      String productName, String provider, List<String> inputDirs, String denyList)
      throws NoSuchMethodException, ClassPathException, InvocationTargetException,
          IllegalAccessException, BuildEncyclopediaDocException, ClassNotFoundException,
          IOException {
    this.nativeRules =
        ImmutableList.copyOf(collectNativeRules(productName, provider, inputDirs, denyList));
    this.globals = Starlark.UNIVERSE;

    ImmutableMap.Builder<String, Object> env = ImmutableMap.builder();
    FakeApi.addPredeclared(
        env,
        /*rules=*/ Lists.newArrayList(),
        /*providers=*/ Lists.newArrayList(),
        /*aspects=*/ Lists.newArrayList());
    this.bzlGlobals = env.build();

    this.types = StarlarkDocumentationCollector.getAllModules();
  }

  /*
   * Returns a list of native rules.
   */
  public List<RuleDocumentation> getNativeRules() {
    return nativeRules;
  }

  /*
   * Returns a mapping between Starlark names and Starkark entities that are available both in BZL
   * and BUILD files.
   */
  public Map<String, Object> getGlobals() {
    return globals;
  }

  /*
   * Returns a mapping between Starlark names and Starkark entities that are available only in BZL
   * files.
   */
  public Map<String, Object> getBzlGlobals() {
    return bzlGlobals;
  }

  // Returns a mapping between type names and module/type documentation.
  public Map<String, StarlarkBuiltinDoc> getTypes() {
    return types;
  }

  /*
   * Collects a list of native rules that are available in BUILD files as top level functions
   * and in BZL files as methods of the native package.
   */
  private List<RuleDocumentation> collectNativeRules(
      String productName, String provider, List<String> inputDirs, String denyList)
      throws NoSuchMethodException, InvocationTargetException, IllegalAccessException,
          BuildEncyclopediaDocException, ClassNotFoundException, IOException {
    ProtoFileBuildEncyclopediaProcessor processor =
        new ProtoFileBuildEncyclopediaProcessor(productName, createRuleClassProvider(provider));
    processor.generateDocumentation(inputDirs, "", denyList);
    return processor.getNativeRules();
  }

  private ConfiguredRuleClassProvider createRuleClassProvider(String classProvider)
      throws NoSuchMethodException, InvocationTargetException, IllegalAccessException,
          ClassNotFoundException {
    Class<?> providerClass = Class.forName(classProvider);
    Method createMethod = providerClass.getMethod("create");
    return (ConfiguredRuleClassProvider) createMethod.invoke(null);
  }
}
