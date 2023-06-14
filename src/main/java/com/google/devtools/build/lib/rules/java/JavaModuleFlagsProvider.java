// Copyright 2022 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.java;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.devtools.build.lib.packages.Type.STRING_LIST;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Streams;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.rules.java.JavaInfo.JavaInfoInternalProvider;
import com.google.devtools.build.lib.starlarkbuildapi.java.JavaModuleFlagsProviderApi;
import java.util.Collection;
import java.util.List;
import java.util.stream.Stream;

/**
 * Provides information about {@code --add-exports=} and {@code --add-opens=} flags for Java
 * targets.
 */
final class JavaModuleFlagsProvider
    implements JavaInfoInternalProvider, JavaModuleFlagsProviderApi {

  private final NestedSet<String> addExports;
  private final NestedSet<String> addOpens;

  public NestedSet<String> addExports() {
    return addExports;
  }

  public NestedSet<String> addOpens() {
    return addOpens;
  }

  @Override
  public Depset /*String*/ getAddExports() {
    return Depset.of(String.class, addExports);
  }

  @Override
  public Depset /*String*/ getAddOpens() {
    return Depset.of(String.class, addOpens);
  }

  public JavaModuleFlagsProvider(NestedSet<String> addExports, NestedSet<String> addOpens) {
    this.addExports = addExports;
    this.addOpens = addOpens;
  }

  public static final JavaModuleFlagsProvider EMPTY =
      new JavaModuleFlagsProvider(
          NestedSetBuilder.emptySet(Order.STABLE_ORDER),
          NestedSetBuilder.emptySet(Order.STABLE_ORDER));

  public static JavaModuleFlagsProvider merge(Collection<JavaModuleFlagsProvider> providers) {
    return create(ImmutableList.of(), ImmutableList.of(), providers.stream());
  }

  public static JavaModuleFlagsProvider create(
      List<String> addExports, List<String> addOpens, Stream<JavaModuleFlagsProvider> transitive) {
    NestedSetBuilder<String> addExportsBuilder = NestedSetBuilder.stableOrder();
    NestedSetBuilder<String> addOpensBuilder = NestedSetBuilder.stableOrder();
    addExportsBuilder.addAll(addExports);
    addOpensBuilder.addAll(addOpens);
    transitive.forEach(
        provider -> {
          addExportsBuilder.addTransitive(provider.addExports());
          addOpensBuilder.addTransitive(provider.addOpens());
        });
    if (addExportsBuilder.isEmpty() && addOpensBuilder.isEmpty()) {
      return EMPTY;
    }
    return new JavaModuleFlagsProvider(addExportsBuilder.build(), addOpensBuilder.build());
  }

  public static JavaModuleFlagsProvider create(
      RuleContext ruleContext, Stream<JavaModuleFlagsProvider> transitive) {
    AttributeMap attributes = ruleContext.attributes();
    return create(
        attributes.getOrDefault("add_exports", STRING_LIST, ImmutableList.of()),
        attributes.getOrDefault("add_opens", STRING_LIST, ImmutableList.of()),
        transitive);
  }

  public static ImmutableList<String> toFlags(List<String> addExports, List<String> addOpens) {
    return Streams.concat(
            addExports.stream().map(x -> String.format("--add-exports=%s=ALL-UNNAMED", x)),
            addOpens.stream().map(x -> String.format("--add-opens=%s=ALL-UNNAMED", x)))
        .collect(toImmutableList());
  }

  public ImmutableList<String> toFlags() {
    return toFlags(addExports().toList(), addOpens().toList());
  }
}
