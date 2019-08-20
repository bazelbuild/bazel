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

package com.google.devtools.build.skydoc.fakebuildapi;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.skylarkbuildapi.ProviderApi;
import com.google.devtools.build.lib.skylarkbuildapi.SkylarkAttrApi;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrinter;
import com.google.devtools.build.lib.skylarkinterface.StarlarkContext;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.SkylarkDict;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.AttributeType;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

/**
 * Fake implementation of {@link SkylarkAttrApi}.
 */
public class FakeSkylarkAttrApi implements SkylarkAttrApi {

  @Override
  public Descriptor intAttribute(
      Integer defaultInt,
      String doc,
      Boolean mandatory,
      SkylarkList<?> values,
      FuncallExpression ast,
      Environment env,
      StarlarkContext context)
      throws EvalException {
    return new FakeDescriptor(AttributeType.INT, doc, mandatory, ImmutableList.of());
  }

  @Override
  public Descriptor stringAttribute(
      String defaultString,
      String doc,
      Boolean mandatory,
      SkylarkList<?> values,
      FuncallExpression ast,
      Environment env,
      StarlarkContext context)
      throws EvalException {
    return new FakeDescriptor(AttributeType.STRING, doc, mandatory, ImmutableList.of());
  }

  @Override
  public Descriptor labelAttribute(
      Object defaultO,
      String doc,
      Boolean executable,
      Object allowFiles,
      Object allowSingleFile,
      Boolean mandatory,
      SkylarkList<?> providers,
      Object allowRules,
      Boolean singleFile,
      Object cfg,
      SkylarkList<?> aspects,
      FuncallExpression ast,
      Environment env,
      StarlarkContext context)
      throws EvalException {
    List<List<String>> allNameGroups = new ArrayList<>();
    if (providers != null) {
      allNameGroups = allProviderNameGroups(providers, env);
    }
    return new FakeDescriptor(AttributeType.LABEL, doc, mandatory, allNameGroups);
  }

  @Override
  public Descriptor stringListAttribute(
      Boolean mandatory,
      Boolean nonEmpty,
      Boolean allowEmpty,
      SkylarkList<?> defaultList,
      String doc,
      FuncallExpression ast,
      Environment env,
      StarlarkContext context)
      throws EvalException {
    return new FakeDescriptor(AttributeType.STRING_LIST, doc, mandatory, ImmutableList.of());
  }

  @Override
  public Descriptor intListAttribute(
      Boolean mandatory,
      Boolean nonEmpty,
      Boolean allowEmpty,
      SkylarkList<?> defaultList,
      String doc,
      FuncallExpression ast,
      Environment env,
      StarlarkContext context)
      throws EvalException {
    return new FakeDescriptor(AttributeType.INT_LIST, doc, mandatory, ImmutableList.of());
  }

  @Override
  public Descriptor labelListAttribute(
      Boolean allowEmpty,
      Object defaultList,
      String doc,
      Object allowFiles,
      Object allowRules,
      SkylarkList<?> providers,
      SkylarkList<?> flags,
      Boolean mandatory,
      Boolean nonEmpty,
      Object cfg,
      SkylarkList<?> aspects,
      FuncallExpression ast,
      Environment env,
      StarlarkContext context)
      throws EvalException {
    List<List<String>> allNameGroups = new ArrayList<>();
    if (providers != null) {
      allNameGroups = allProviderNameGroups(providers, env);
    }
    return new FakeDescriptor(AttributeType.LABEL_LIST, doc, mandatory, allNameGroups);
  }

  @Override
  public Descriptor labelKeyedStringDictAttribute(
      Boolean allowEmpty,
      Object defaultList,
      String doc,
      Object allowFiles,
      Object allowRules,
      SkylarkList<?> providers,
      SkylarkList<?> flags,
      Boolean mandatory,
      Boolean nonEmpty,
      Object cfg,
      SkylarkList<?> aspects,
      FuncallExpression ast,
      Environment env,
      StarlarkContext context)
      throws EvalException {
    List<List<String>> allNameGroups = new ArrayList<>();
    if (providers != null) {
      allNameGroups = allProviderNameGroups(providers, env);
    }
    return new FakeDescriptor(AttributeType.LABEL_STRING_DICT, doc, mandatory, allNameGroups);
  }

  @Override
  public Descriptor boolAttribute(
      Boolean defaultO,
      String doc,
      Boolean mandatory,
      FuncallExpression ast,
      Environment env,
      StarlarkContext context)
      throws EvalException {
    return new FakeDescriptor(AttributeType.BOOLEAN, doc, mandatory, ImmutableList.of());
  }

  @Override
  public Descriptor outputAttribute(
      Object defaultO,
      String doc,
      Boolean mandatory,
      FuncallExpression ast,
      Environment env,
      StarlarkContext context)
      throws EvalException {
    return new FakeDescriptor(AttributeType.OUTPUT, doc, mandatory, ImmutableList.of());
  }

  @Override
  public Descriptor outputListAttribute(
      Boolean allowEmpty,
      Object defaultList,
      String doc,
      Boolean mandatory,
      Boolean nonEmpty,
      FuncallExpression ast,
      Environment env,
      StarlarkContext context)
      throws EvalException {
    return new FakeDescriptor(AttributeType.OUTPUT_LIST, doc, mandatory, ImmutableList.of());
  }

  @Override
  public Descriptor stringDictAttribute(
      Boolean allowEmpty,
      SkylarkDict<?, ?> defaultO,
      String doc,
      Boolean mandatory,
      Boolean nonEmpty,
      FuncallExpression ast,
      Environment env,
      StarlarkContext context)
      throws EvalException {
    return new FakeDescriptor(AttributeType.STRING_DICT, doc, mandatory, ImmutableList.of());
  }

  @Override
  public Descriptor stringListDictAttribute(
      Boolean allowEmpty,
      SkylarkDict<?, ?> defaultO,
      String doc,
      Boolean mandatory,
      Boolean nonEmpty,
      FuncallExpression ast,
      Environment env,
      StarlarkContext context)
      throws EvalException {
    return new FakeDescriptor(AttributeType.STRING_LIST_DICT, doc, mandatory, ImmutableList.of());
  }

  @Override
  public Descriptor licenseAttribute(
      Object defaultO,
      String doc,
      Boolean mandatory,
      FuncallExpression ast,
      Environment env,
      StarlarkContext context)
      throws EvalException {
    return new FakeDescriptor(AttributeType.STRING_LIST, doc, mandatory, ImmutableList.of());
  }

  @Override
  public void repr(SkylarkPrinter printer) {}

  /**
   * Returns a list of provider name groups, given the value of a Starlark attribute's "providers"
   * argument.
   *
   * <p>{@code providers} can either be a list of providers or a list of lists of providers, where
   * each provider is represented by a ProviderApi or by a String. In the case of a single-level
   * list, the whole list is considered a single group, while in the case of a double-level list,
   * each of the inner lists is a separate group.
   */
  private static List<List<String>> allProviderNameGroups(
      SkylarkList<?> providers, Environment env) {

    List<List<String>> allNameGroups = new ArrayList<>();
    for (Object object : providers) {
      List<String> providerNameGroup;
      if (object instanceof SkylarkList) {
        SkylarkList<?> group = (SkylarkList<?>) object;
        providerNameGroup = parseProviderGroup(group, env);
        allNameGroups.add(providerNameGroup);
      } else {
        providerNameGroup = parseProviderGroup(providers, env);
        allNameGroups.add(providerNameGroup);
        break;
      }
    }
    return allNameGroups;
  }

  /**
   * Returns the names of the providers in the given group.
   *
   * <p>Each item in the group may be either a {@link ProviderApi} or a {@code String} (representing
   * a legacy provider).
   */
  private static List<String> parseProviderGroup(SkylarkList<?> group, Environment env) {
    List<String> providerNameGroup = new ArrayList<>();
    for (Object object : group) {
      if (object instanceof ProviderApi) {
        ProviderApi provider = (ProviderApi) object;
        String providerName = providerName(provider, env);
        providerNameGroup.add(providerName);
      } else if (object instanceof String) {
        String legacyProvider = (String) object;
        providerNameGroup.add(legacyProvider);
      }
    }
    return providerNameGroup;
  }

  /**
   * Returns the name of {@code provider}.
   *
   * <p>{@code env} contains a {@code Map<String, Object>} where the values are built-in objects or
   * objects defined in the file and the keys are the names of these objects. If a {@code provider}
   * is in the map, the name of the provider is set as the key of this object in {@code bindings}.
   * If it is not in the map, the provider may be part of a module in the map and the name will be
   * set to "Unknown Provider".
   */
  private static String providerName(ProviderApi provider, Environment env) {
    Map<String, Object> bindings = env.getGlobals().getTransitiveBindings();
    if (bindings.containsValue(provider)) {
      for (Entry<String, Object> envEntry : bindings.entrySet()) {
        if (provider.equals(envEntry.getValue())) {
          return envEntry.getKey();
        }
      }
    }
    return "Unknown Provider";
  }
}
