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
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkbuildapi.SkylarkAspectApi;
import com.google.devtools.build.lib.skylarkbuildapi.SkylarkRuleFunctionsApi;
import com.google.devtools.build.lib.skylarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.syntax.BaseFunction;
import com.google.devtools.build.lib.syntax.Dict;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.FunctionSignature;
import com.google.devtools.build.lib.syntax.Sequence;
import com.google.devtools.build.lib.syntax.SkylarkType;
import com.google.devtools.build.lib.syntax.Starlark;
import com.google.devtools.build.lib.syntax.StarlarkFunction;
import com.google.devtools.build.lib.syntax.StarlarkThread;
import com.google.devtools.build.skydoc.rendering.AspectInfoWrapper;
import com.google.devtools.build.skydoc.rendering.ProviderInfoWrapper;
import com.google.devtools.build.skydoc.rendering.RuleInfoWrapper;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.AspectInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.AttributeInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.AttributeType;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.ProviderFieldInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.ProviderInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.RuleInfo;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Fake implementation of {@link SkylarkRuleFunctionsApi}.
 *
 * <p>This fake hooks into the global {@code rule()} function, adding descriptors of calls of that
 * function to a {@link List} given in the class constructor.</p>
 */
public class FakeSkylarkRuleFunctionsApi implements SkylarkRuleFunctionsApi<FileApi> {

  private static final FakeDescriptor IMPLICIT_NAME_ATTRIBUTE_DESCRIPTOR =
      new FakeDescriptor(
          AttributeType.NAME, "A unique name for this target.", true, ImmutableList.of(), "");
  private final List<RuleInfoWrapper> ruleInfoList;

  private final List<ProviderInfoWrapper> providerInfoList;

  private final List<AspectInfoWrapper> aspectInfoList;

  /**
   * Constructor.
   *
   * @param ruleInfoList the list of {@link RuleInfo} objects to which rule() invocation information
   *     will be added
   * @param providerInfoList the list of {@link ProviderInfo} objects to which provider() invocation
   *     information will be added
   * @param aspectInfoList the list of {@link AspectInfo} objects to which aspect() invocation
   *     information will be added
   */
  public FakeSkylarkRuleFunctionsApi(
      List<RuleInfoWrapper> ruleInfoList,
      List<ProviderInfoWrapper> providerInfoList,
      List<AspectInfoWrapper> aspectInfoList) {
    this.ruleInfoList = ruleInfoList;
    this.providerInfoList = providerInfoList;
    this.aspectInfoList = aspectInfoList;
  }

  @Override
  public ProviderApi provider(String doc, Object fields, Location location) throws EvalException {
    FakeProviderApi fakeProvider = new FakeProviderApi();
    // Field documentation will be output preserving the order in which the fields are listed.
    ImmutableList.Builder<ProviderFieldInfo> providerFieldInfos = ImmutableList.builder();
    if (fields instanceof Sequence) {
      @SuppressWarnings("unchecked")
      Sequence<String> fieldNames =
          (Sequence<String>)
              SkylarkType.cast(
                  fields,
                  Sequence.class,
                  String.class,
                  location,
                  "Expected list of strings or dictionary of string -> string for 'fields'");
      for (String fieldName : fieldNames) {
        providerFieldInfos.add(asProviderFieldInfo(fieldName, "(Undocumented)"));
      }
    } else if (fields instanceof Dict) {
      Map<String, String> dict = SkylarkType.castMap(
          fields,
          String.class, String.class,
          "Expected list of strings or dictionary of string -> string for 'fields'");
      for (Map.Entry<String, String> fieldEntry : dict.entrySet()) {
        providerFieldInfos.add(asProviderFieldInfo(fieldEntry.getKey(), fieldEntry.getValue()));
      }
    } else {
      // fields is NONE, so there is no field information to add.
    }
    providerInfoList.add(forProviderInfo(fakeProvider, doc, providerFieldInfos.build()));
    return fakeProvider;
  }

  /** Constructor for ProviderFieldInfo. */
  public ProviderFieldInfo asProviderFieldInfo(String name, String docString) {
    return ProviderFieldInfo.newBuilder().setName(name).setDocString(docString).build();
  }

  /** Constructor for ProviderInfoWrapper. */
  public ProviderInfoWrapper forProviderInfo(
      BaseFunction identifier, String docString, Collection<ProviderFieldInfo> fieldInfos) {
    return new ProviderInfoWrapper(identifier, docString, fieldInfos);
  }

  @Override
  public BaseFunction rule(
      StarlarkFunction implementation,
      Boolean test,
      Object attrs,
      Object implicitOutputs,
      Boolean executable,
      Boolean outputToGenfiles,
      Sequence<?> fragments,
      Sequence<?> hostFragments,
      Boolean skylarkTestable,
      Sequence<?> toolchains,
      String doc,
      Sequence<?> providesArg,
      Sequence<?> execCompatibleWith,
      Object analysisTest,
      Object buildSetting,
      Object cfg,
      FuncallExpression ast,
      StarlarkThread funcallThread)
      throws EvalException {
    ImmutableMap.Builder<String, FakeDescriptor> attrsMapBuilder = ImmutableMap.builder();
    if (attrs != null && attrs != Starlark.NONE) {
      Dict<?, ?> attrsDict = (Dict<?, ?>) attrs;
      attrsMapBuilder.putAll(attrsDict.getContents(String.class, FakeDescriptor.class, "attrs"));
    }

    attrsMapBuilder.put("name", IMPLICIT_NAME_ATTRIBUTE_DESCRIPTOR);
    List<AttributeInfo> attrInfos =
        attrsMapBuilder.build().entrySet().stream()
            .filter(entry -> !entry.getKey().startsWith("_"))
            .map(entry -> entry.getValue().asAttributeInfo(entry.getKey()))
            .collect(Collectors.toList());
    attrInfos.sort(new AttributeNameComparator());

    RuleDefinitionIdentifier functionIdentifier = new RuleDefinitionIdentifier();

    // Only the Builder is passed to RuleInfoWrapper as the rule name is not yet available.
    RuleInfo.Builder ruleInfo = RuleInfo.newBuilder().setDocString(doc).addAllAttribute(attrInfos);

    ruleInfoList.add(new RuleInfoWrapper(functionIdentifier, ast.getLocation(), ruleInfo));

    return functionIdentifier;
  }

  @Override
  public Label label(
      String labelString, Boolean relativeToCallerRepository, Location loc, StarlarkThread thread)
      throws EvalException {
    try {
      return Label.parseAbsolute(
          labelString,
          /* defaultToMain= */ false,
          /* repositoryMapping= */ ImmutableMap.of());
    } catch (LabelSyntaxException e) {
      throw new EvalException(loc, "Illegal absolute label syntax: " + labelString);
    }
  }

  @Override
  public SkylarkAspectApi aspect(
      StarlarkFunction implementation,
      Sequence<?> attributeAspects,
      Object attrs,
      Sequence<?> requiredAspectProvidersArg,
      Sequence<?> providesArg,
      Sequence<?> fragments,
      Sequence<?> hostFragments,
      Sequence<?> toolchains,
      String doc,
      Boolean applyToFiles,
      FuncallExpression ast,
      StarlarkThread funcallThread)
      throws EvalException {
    FakeSkylarkAspect fakeAspect = new FakeSkylarkAspect();
    ImmutableMap.Builder<String, FakeDescriptor> attrsMapBuilder = ImmutableMap.builder();
    if (attrs != null && attrs != Starlark.NONE) {
      Dict<?, ?> attrsDict = (Dict<?, ?>) attrs;
      attrsMapBuilder.putAll(attrsDict.getContents(String.class, FakeDescriptor.class, "attrs"));
    }

    attrsMapBuilder.put("name", IMPLICIT_NAME_ATTRIBUTE_DESCRIPTOR);
    List<AttributeInfo> attrInfos =
        attrsMapBuilder.build().entrySet().stream()
            .filter(entry -> !entry.getKey().startsWith("_"))
            .map(entry -> entry.getValue().asAttributeInfo(entry.getKey()))
            .collect(Collectors.toList());
    attrInfos.sort(new AttributeNameComparator());

    List<String> aspectAttrs = new ArrayList<>();
    if (attributeAspects != null) {
      aspectAttrs = attributeAspects.getContents(String.class, "aspectAttrs");
    }

    aspectAttrs =
        aspectAttrs.stream().filter(entry -> !entry.startsWith("_")).collect(Collectors.toList());

    // Only the Builder is passed to AspectInfoWrapper as the aspect name is not yet available.
    AspectInfo.Builder aspectInfo =
        AspectInfo.newBuilder()
            .setDocString(doc)
            .addAllAttribute(attrInfos)
            .addAllAspectAttribute(aspectAttrs);

    aspectInfoList.add(new AspectInfoWrapper(fakeAspect, ast.getLocation(), aspectInfo));

    return fakeAspect;
  }

  /**
   * A fake {@link BaseFunction} implementation which serves as an identifier for a rule definition.
   * A skylark invocation of 'rule()' should spawn a unique instance of this class and return it.
   * Thus, skylark code such as 'foo = rule()' will result in 'foo' being assigned to a unique
   * identifier, which can later be matched to a registered rule() invocation saved by the fake
   * build API implementation.
   */
  private static class RuleDefinitionIdentifier extends BaseFunction {

    private static int idCounter = 0;
    private final String name = "RuleDefinitionIdentifier" + idCounter++;

    @Override
    public FunctionSignature getSignature() {
      return FunctionSignature.KWARGS;
    }

    @Override
    public String getName() {
      return name;
    }
  }

  /**
   * A comparator for {@link AttributeInfo} objects which sorts by attribute name alphabetically,
   * except that any attribute named "name" is placed first.
   */
  public static class AttributeNameComparator implements Comparator<AttributeInfo> {

    @Override
    public int compare(AttributeInfo o1, AttributeInfo o2) {
      if (o1.getName().equals("name")) {
        return o2.getName().equals("name") ? 0 : -1;
      } else if (o2.getName().equals("name")) {
        return 1;
      } else {
        return o1.getName().compareTo(o2.getName());
      }
    }
  }
}
