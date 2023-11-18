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
import com.google.devtools.build.lib.starlarkbuildapi.StarlarkAttrModuleApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.ProviderApi;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.AttributeType;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Module;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkInt;
import net.starlark.java.eval.StarlarkThread;

/** Fake implementation of {@link StarlarkAttrModuleApi}. */
public class FakeStarlarkAttrModuleApi implements StarlarkAttrModuleApi {

  @Override
  public Descriptor intAttribute(
      StarlarkInt defaultInt,
      Object doc,
      Boolean mandatory,
      Sequence<?> values,
      StarlarkThread thread)
      throws EvalException {
    return new FakeDescriptor(
        AttributeType.INT, toTrimmedString(doc), mandatory, ImmutableList.of(), defaultInt);
  }

  @Override
  public Descriptor stringAttribute(
      Object defaultString,
      Object doc,
      Boolean mandatory,
      Sequence<?> values,
      StarlarkThread thread)
      throws EvalException {
    return new FakeDescriptor(
        AttributeType.STRING,
        toTrimmedString(doc),
        mandatory,
        ImmutableList.of(),
        defaultString != null ? "\"" + defaultString + "\"" : null);
  }

  @Override
  public Descriptor labelAttribute(
      Object defaultO,
      Object doc,
      Boolean executable,
      Object allowFiles,
      Object allowSingleFile,
      Boolean mandatory,
      Sequence<?> providers,
      Object allowRules,
      Object cfg,
      Sequence<?> aspects,
      Object flags,
      StarlarkThread thread)
      throws EvalException {
    List<List<String>> allNameGroups = new ArrayList<>();
    if (providers != null) {
      allNameGroups = allProviderNameGroups(providers, thread);
    }
    return new FakeDescriptor(
        AttributeType.LABEL, toTrimmedString(doc), mandatory, allNameGroups, defaultO);
  }

  @Override
  public Descriptor stringListAttribute(
      Boolean mandatory, Boolean allowEmpty, Object defaultList, Object doc, StarlarkThread thread)
      throws EvalException {
    return new FakeDescriptor(
        AttributeType.STRING_LIST,
        toTrimmedString(doc),
        mandatory,
        ImmutableList.of(),
        defaultList);
  }

  @Override
  public Descriptor intListAttribute(
      Boolean mandatory,
      Boolean allowEmpty,
      Sequence<?> defaultList,
      Object doc,
      StarlarkThread thread)
      throws EvalException {
    return new FakeDescriptor(
        AttributeType.INT_LIST, toTrimmedString(doc), mandatory, ImmutableList.of(), defaultList);
  }

  @Override
  public Descriptor labelListAttribute(
      Boolean allowEmpty,
      Object defaultList,
      Object doc,
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
        AttributeType.LABEL_LIST, toTrimmedString(doc), mandatory, allNameGroups, defaultList);
  }

  @Override
  public Descriptor labelKeyedStringDictAttribute(
      Boolean allowEmpty,
      Object defaultList,
      Object doc,
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
        AttributeType.LABEL_STRING_DICT,
        toTrimmedString(doc),
        mandatory,
        allNameGroups,
        defaultList);
  }

  @Override
  public Descriptor boolAttribute(
      Boolean defaultO, Object doc, Boolean mandatory, StarlarkThread thread) throws EvalException {
    return new FakeDescriptor(
        AttributeType.BOOLEAN,
        toTrimmedString(doc),
        mandatory,
        ImmutableList.of(),
        Boolean.TRUE.equals(defaultO) ? "True" : "False");
  }

  @Override
  public Descriptor outputAttribute(Object doc, Boolean mandatory, StarlarkThread thread)
      throws EvalException {
    return new FakeDescriptor(
        AttributeType.OUTPUT, toTrimmedString(doc), mandatory, ImmutableList.of(), "");
  }

  @Override
  public Descriptor outputListAttribute(
      Boolean allowEmpty, Object doc, Boolean mandatory, StarlarkThread thread)
      throws EvalException {
    return new FakeDescriptor(
        AttributeType.OUTPUT_LIST, toTrimmedString(doc), mandatory, ImmutableList.of(), "");
  }

  @Override
  public Descriptor stringDictAttribute(
      Boolean allowEmpty, Dict<?, ?> defaultO, Object doc, Boolean mandatory, StarlarkThread thread)
      throws EvalException {
    return new FakeDescriptor(
        AttributeType.STRING_DICT, toTrimmedString(doc), mandatory, ImmutableList.of(), defaultO);
  }

  @Override
  public Descriptor stringListDictAttribute(
      Boolean allowEmpty, Dict<?, ?> defaultO, Object doc, Boolean mandatory, StarlarkThread thread)
      throws EvalException {
    return new FakeDescriptor(
        AttributeType.STRING_LIST_DICT,
        toTrimmedString(doc),
        mandatory,
        ImmutableList.of(),
        defaultO);
  }

  @Override
  public Descriptor licenseAttribute(
      Object defaultO, Object doc, Boolean mandatory, StarlarkThread thread) throws EvalException {
    return new FakeDescriptor(
        AttributeType.STRING_LIST, toTrimmedString(doc), mandatory, ImmutableList.of(), defaultO);
  }

  private static Optional<String> toTrimmedString(Object doc) {
    return Starlark.toJavaOptional(doc, String.class).map(Starlark::trimDocString);
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

  // Returns the name of the provider using fragile heuristics.
  private static String providerName(ProviderApi provider, StarlarkThread thread) {
    Module bzl = Module.ofInnermostEnclosingStarlarkFunction(thread);

    // Generic fake provider? (e.g. Starlark-defined, or trivial fake)
    // Return name set at construction, or by "export" operation, if any.
    if (provider instanceof FakeProviderApi) {
      return ((FakeProviderApi) provider).getName(); // may be "Unexported Provider"
    }

    // Specialized fake provider? (e.g. DefaultInfo)
    // Return name under which FakeApi.addPredeclared added it to environment.
    // (This only works for top-level names such as DefaultInfo, but not for
    // nested ones such as cc_common.XyzInfo, but that has always been broken;
    // it is not part of the regression that is b/175703093.)
    for (Map.Entry<String, Object> e : bzl.getPredeclaredBindings().entrySet()) {
      if (provider.equals(e.getValue())) {
        return e.getKey();
      }
    }

    return "Unknown Provider";
  }
}
