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
import com.google.devtools.build.docgen.StarlarkDocumentationProcessor.Category;
import com.google.devtools.build.docgen.starlark.StarlarkDocExpander;
import com.google.devtools.build.docgen.starlark.StarlarkDocPage;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.packages.StarlarkNativeModule;
import com.google.devtools.build.lib.util.Classpath.ClassPathException;
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
  private final ImmutableMap<Category, ImmutableList<StarlarkDocPage>> allDocPages;

  // Mappings between Starlark names and Starlark entities generated from the fakebuildapi.
  private final ImmutableMap<String, Object> globals;
  private final ImmutableMap<String, Object> bzlGlobals;

  public SymbolFamilies(
      StarlarkDocExpander expander,
      SourceUrlMapper urlMapper,
      String provider,
      List<String> inputJavaDirs,
      List<String> buildEncyclopediaStardocProtos,
      String ruleDenyList,
      List<String> apiStardocProtos)
      throws NoSuchMethodException,
          ClassPathException,
          InvocationTargetException,
          IllegalAccessException,
          BuildEncyclopediaDocException,
          ClassNotFoundException,
          IOException {
    ConfiguredRuleClassProvider configuredRuleClassProvider = createRuleClassProvider(provider);
    this.nativeRules =
        ImmutableList.copyOf(
            collectNativeRules(
                expander.ruleExpander,
                urlMapper,
                configuredRuleClassProvider,
                inputJavaDirs,
                buildEncyclopediaStardocProtos,
                ruleDenyList));
    this.globals = Starlark.UNIVERSE;
    this.bzlGlobals = collectBzlGlobals(configuredRuleClassProvider);
    this.allDocPages =
        StarlarkDocumentationCollector.getAllDocPages(
            expander, ImmutableList.copyOf(apiStardocProtos));
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
  public ImmutableMap<Category, ImmutableList<StarlarkDocPage>> getAllDocPages() {
    return allDocPages;
  }

  /** Collects symbols predefined in BZL files. */
  private ImmutableMap<String, Object> collectBzlGlobals(ConfiguredRuleClassProvider provider) {
    // StarlarkNativeModule is treated specially because we want to inherit the documentation
    // carried in its annotations, whereas the real "native" object is just a bare struct.
    ImmutableMap.Builder<String, Object> env = ImmutableMap.builder();
    env.put("native", new StarlarkNativeModule());
    for (Map.Entry<String, Object> entry :
        provider.getBazelStarlarkEnvironment().getUninjectedBuildBzlEnv().entrySet()) {
      if (entry.getKey().equals("native")) {
        continue;
      }
      env.put(entry);
    }
    return env.buildOrThrow();
  }

  /*
   * Collects a list of native rules that are available in BUILD files as top level functions
   * and in BZL files as methods of the native package.
   */
  private List<RuleDocumentation> collectNativeRules(
      RuleLinkExpander linkExpander,
      SourceUrlMapper urlMapper,
      ConfiguredRuleClassProvider provider,
      List<String> inputJavaDirs,
      List<String> buildEncyclopediaStardocProtos,
      String denyList)
      throws BuildEncyclopediaDocException, IOException {
    ProtoFileBuildEncyclopediaProcessor processor =
        new ProtoFileBuildEncyclopediaProcessor(linkExpander, urlMapper, provider);
    processor.generateDocumentation(inputJavaDirs, buildEncyclopediaStardocProtos, "", denyList);
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
