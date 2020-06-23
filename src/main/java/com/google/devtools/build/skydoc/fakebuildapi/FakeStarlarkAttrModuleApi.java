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
import com.google.devtools.build.lib.skylarkbuildapi.StarlarkAttrModuleApi;
import com.google.devtools.build.lib.skylarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.syntax.Dict;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Module;
import com.google.devtools.build.lib.syntax.Printer;
import com.google.devtools.build.lib.syntax.Sequence;
import com.google.devtools.build.lib.syntax.StarlarkThread;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.AttributeType;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

/**
 * Fake implementation of {@link StarlarkAttrModuleApi}.
 */
public class FakeStarlarkAttrModuleApi implements StarlarkAttrModuleApi {

  @Override
  public Descriptor intAttribute(
      Integer defaultInt,
      String doc,
      Boolean mandatory,
      Sequence<?> values,
      StarlarkThread thread)
      throws EvalException {
    return new FakeDescriptor(AttributeType.INT, doc, mandatory, ImmutableList.of(), defaultInt);
  }

  @Override
  public Descriptor stringAttribute(
      String defaultString,
      String doc,
      Boolean mandatory,
      Sequence<?> values,
      StarlarkThread thread)
      throws EvalException {
    return new FakeDescriptor(
        AttributeType.STRING,
        doc,
        mandatory,
        ImmutableList.of(),
        defaultString != null ? "\"" + defaultString + "\"" : null);
  }

  @Override
  public Descriptor labelAttribute(
      Object defaultO,
      String doc,
      Boolean executable,
      Object allowFiles,
      Object allowSingleFile,
      Boolean mandatory,
      Sequence<?> providers,
      Object allowRules,
      Object cfg,
      Sequence<?> aspects,
      StarlarkThread thread)
      throws EvalException {
    List<List<String>> allNameGroups = new ArrayList<>();
    if (providers != null) {
      allNameGroups = allProviderNameGroups(providers, thread);
    }
    return new FakeDescriptor(AttributeType.LABEL, doc, mandatory, allNameGroups, defaultO);
  }

  @Override
  public Descriptor stringListAttribute(
      Boolean mandatory,
      Boolean allowEmpty,
      Sequence<?> defaultList,
      String doc,
      StarlarkThread thread)
      throws EvalException {
    return new FakeDescriptor(
        AttributeType.STRING_LIST, doc, mandatory, ImmutableList.of(), defaultList);
  }

  @Override
  public Descriptor intListAttribute(
      Boolean mandatory,
      Boolean allowEmpty,
      Sequence<?> defaultList,
      String doc,
      StarlarkThread thread)
      throws EvalException {
    return new FakeDescriptor(
        AttributeType.INT_LIST, doc, mandatory, ImmutableList.of(), defaultList);
  }

  @Override
  public Descriptor labelListAttribute(
      Boolean allowEmpty,
      Object defaultList,
      String doc,
      Object allowFiles,
      Object allowRules,
      Sequence<?> providers,
      Sequence<?> flags,
      Boolean mandatory,
      Object cfg,
      Sequence<?> aspects,
      StarlarkThread thread)
      throws EvalException {
    List<List<String>> allNameGroups = new ArrayList<>();
    if (providers != null) {
      allNameGroups = allProviderNameGroups(providers, thread);
    }
    return new FakeDescriptor(AttributeType.LABEL_LIST, doc, mandatory, allNameGroups, defaultList);
  }

  @Override
  public Descriptor labelKeyedStringDictAttribute(
      Boolean allowEmpty,
      Object defaultList,
      String doc,
      Object allowFiles,
      Object allowRules,
      Sequence<?> providers,
      Sequence<?> flags,
      Boolean mandatory,
      Object cfg,
      Sequence<?> aspects,
      StarlarkThread thread)
      throws EvalException {
    List<List<String>> allNameGroups = new ArrayList<>();
    if (providers != null) {
      allNameGroups = allProviderNameGroups(providers, thread);
    }
    return new FakeDescriptor(
        AttributeType.LABEL_STRING_DICT, doc, mandatory, allNameGroups, defaultList);
  }

  @Override
  public Descriptor boolAttribute(
      Boolean defaultO, String doc, Boolean mandatory, StarlarkThread thread) throws EvalException {
    return new FakeDescriptor(
        AttributeType.BOOLEAN,
        doc,
        mandatory,
        ImmutableList.of(),
        Boolean.TRUE.equals(defaultO) ? "True" : "False");
  }

  @Override
  public Descriptor outputAttribute(String doc, Boolean mandatory, StarlarkThread thread)
      throws EvalException {
    return new FakeDescriptor(AttributeType.OUTPUT, doc, mandatory, ImmutableList.of(), "");
  }

  @Override
  public Descriptor outputListAttribute(
      Boolean allowEmpty,
      String doc,
      Boolean mandatory,
      StarlarkThread thread)
      throws EvalException {
    return new FakeDescriptor(AttributeType.OUTPUT_LIST, doc, mandatory, ImmutableList.of(), "");
  }

  @Override
  public Descriptor stringDictAttribute(
      Boolean allowEmpty,
      Dict<?, ?> defaultO,
      String doc,
      Boolean mandatory,
      StarlarkThread thread)
      throws EvalException {
    return new FakeDescriptor(
        AttributeType.STRING_DICT, doc, mandatory, ImmutableList.of(), defaultO);
  }

  @Override
  public Descriptor stringListDictAttribute(
      Boolean allowEmpty,
      Dict<?, ?> defaultO,
      String doc,
      Boolean mandatory,
      StarlarkThread thread)
      throws EvalException {
    return new FakeDescriptor(
        AttributeType.STRING_LIST_DICT, doc, mandatory, ImmutableList.of(), defaultO);
  }

  @Override
  public Descriptor licenseAttribute(
      Object defaultO, String doc, Boolean mandatory, StarlarkThread thread) throws EvalException {
    return new FakeDescriptor(
        AttributeType.STRING_LIST, doc, mandatory, ImmutableList.of(), defaultO);
  }

  @Override
  public void repr(Printer printer) {}

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
      Sequence<?> providers, StarlarkThread thread) {

    List<List<String>> allNameGroups = new ArrayList<>();
    for (Object object : providers) {
      List<String> providerNameGroup;
      if (object instanceof Sequence) {
        Sequence<?> group = (Sequence<?>) object;
        providerNameGroup = parseProviderGroup(group, thread);
        allNameGroups.add(providerNameGroup);
      } else {
        providerNameGroup = parseProviderGroup(providers, thread);
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
  private static List<String> parseProviderGroup(Sequence<?> group, StarlarkThread thread) {
    List<String> providerNameGroup = new ArrayList<>();
    for (Object object : group) {
      if (object instanceof ProviderApi) {
        ProviderApi provider = (ProviderApi) object;
        String providerName = providerName(provider, thread);
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
   * <p>{@code thread} contains a {@code Map<String, Object>} where the values are built-in objects
   * or objects defined in the file and the keys are the names of these objects. If a {@code
   * provider} is in the map, the name of the provider is set as the key of this object in {@code
   * bindings}. If it is not in the map, the provider may be part of a module in the map and the
   * name will be set to "Unknown Provider".
   */
  private static String providerName(ProviderApi provider, StarlarkThread thread) {
    Map<String, Object> bindings =
        Module.ofInnermostEnclosingStarlarkFunction(thread).getTransitiveBindings();
    for (Entry<String, Object> envEntry : bindings.entrySet()) {
      if (provider.equals(envEntry.getValue())) {
        return envEntry.getKey();
      }
    }
    return "Unknown Provider";
  }
}
