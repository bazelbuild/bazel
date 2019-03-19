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
import com.google.devtools.build.lib.skylarkbuildapi.FileTypeApi;
import com.google.devtools.build.lib.skylarkbuildapi.ProviderApi;
import com.google.devtools.build.lib.skylarkbuildapi.SkylarkAspectApi;
import com.google.devtools.build.lib.skylarkbuildapi.SkylarkRuleFunctionsApi;
import com.google.devtools.build.lib.skylarkinterface.StarlarkContext;
import com.google.devtools.build.lib.syntax.BaseFunction;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.Runtime;
import com.google.devtools.build.lib.syntax.SkylarkDict;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkType;
import com.google.devtools.build.skydoc.rendering.AttributeInfo;
import com.google.devtools.build.skydoc.rendering.AttributeInfo.Type;
import com.google.devtools.build.skydoc.rendering.ProviderFieldInfo;
import com.google.devtools.build.skydoc.rendering.ProviderInfo;
import com.google.devtools.build.skydoc.rendering.RuleInfo;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import javax.annotation.Nullable;

/**
 * Fake implementation of {@link SkylarkRuleFunctionsApi}.
 *
 * <p>This fake hooks into the global {@code rule()} function, adding descriptors of calls of that
 * function to a {@link List} given in the class constructor.</p>
 */
public class FakeSkylarkRuleFunctionsApi implements SkylarkRuleFunctionsApi<FileApi> {

  private static final FakeDescriptor IMPLICIT_NAME_ATTRIBUTE_DESCRIPTOR =
      new FakeDescriptor(Type.NAME, "A unique name for this target.", true);
  private final List<RuleInfo> ruleInfoList;
  private final List<ProviderInfo> providerInfoList;

  /**
   * Constructor.
   *
   * @param ruleInfoList the list of {@link RuleInfo} objects to which rule() invocation information
   *     will be added
   * @param providerInfoList the list of {@link ProviderInfo} objects to which provider()
   *     invocation information will be added
   */
  public FakeSkylarkRuleFunctionsApi(List<RuleInfo> ruleInfoList,
      List<ProviderInfo> providerInfoList) {
    this.ruleInfoList = ruleInfoList;
    this.providerInfoList = providerInfoList;
  }

  @Override
  public ProviderApi provider(String doc, Object fields, Location location) throws EvalException {
    FakeProviderApi fakeProvider = new FakeProviderApi();
    // Field documentation will be output preserving the order in which the fields are listed.
    ImmutableList.Builder<ProviderFieldInfo> providerFieldInfos = ImmutableList.builder();
    if (fields instanceof SkylarkList) {
      @SuppressWarnings("unchecked")
      SkylarkList<String> fieldNames = (SkylarkList<String>)
          SkylarkType.cast(
              fields,
              SkylarkList.class, String.class, location,
              "Expected list of strings or dictionary of string -> string for 'fields'");
      for (String fieldName : fieldNames) {
        providerFieldInfos.add(new ProviderFieldInfo(fieldName));
      }
    } else if (fields instanceof SkylarkDict) {
      Map<String, String> dict = SkylarkType.castMap(
          fields,
          String.class, String.class,
          "Expected list of strings or dictionary of string -> string for 'fields'");
      for (Map.Entry<String, String> fieldEntry : dict.entrySet()) {
        providerFieldInfos.add(new ProviderFieldInfo(fieldEntry.getKey(), fieldEntry.getValue()));
      }
    } else {
      // fields is NONE, so there is no field information to add.
    }
    providerInfoList.add(new ProviderInfo(fakeProvider, doc, providerFieldInfos.build()));
    return fakeProvider;
  }

  @Override
  public BaseFunction rule(
      BaseFunction implementation,
      Boolean test,
      Object attrs,
      Object implicitOutputs,
      Boolean executable,
      Boolean outputToGenfiles,
      SkylarkList<?> fragments,
      SkylarkList<?> hostFragments,
      Boolean skylarkTestable,
      SkylarkList<?> toolchains,
      String doc,
      SkylarkList<?> providesArg,
      Boolean executionPlatformConstraintsAllowed,
      SkylarkList<?> execCompatibleWith,
      Object analysisTest,
      Object buildSetting,
      Object cfg,
      FuncallExpression ast,
      Environment funcallEnv,
      StarlarkContext context)
      throws EvalException {
    List<AttributeInfo> attrInfos;
    ImmutableMap.Builder<String, FakeDescriptor> attrsMapBuilder = ImmutableMap.builder();
    if (attrs != null && attrs != Runtime.NONE) {
      SkylarkDict<?, ?> attrsDict = (SkylarkDict<?, ?>) attrs;
      attrsMapBuilder.putAll(attrsDict.getContents(String.class, FakeDescriptor.class, "attrs"));
    }

    attrsMapBuilder.put("name", IMPLICIT_NAME_ATTRIBUTE_DESCRIPTOR);
    attrInfos = attrsMapBuilder.build().entrySet().stream()
        .filter(entry -> !entry.getKey().startsWith("_"))
        .map(entry -> new AttributeInfo(
            entry.getKey(),
            entry.getValue().getDocString(),
            entry.getValue().getType(),
            entry.getValue().isMandatory()))
        .collect(Collectors.toList());
    attrInfos.sort(new AttributeNameComparator());

    RuleDefinitionIdentifier functionIdentifier = new RuleDefinitionIdentifier();

    ruleInfoList.add(new RuleInfo(functionIdentifier, ast.getLocation(), doc, attrInfos));
    return functionIdentifier;
  }

  @Override
  public Label label(
      String labelString,
      Boolean relativeToCallerRepository,
      Location loc,
      Environment env,
      StarlarkContext context)
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
  public FileTypeApi<FileApi> fileType(SkylarkList<?> types, Location loc, Environment env)
      throws EvalException {
    return null;
  }

  @Override
  public SkylarkAspectApi aspect(BaseFunction implementation, SkylarkList<?> attributeAspects,
      Object attrs, SkylarkList<?> requiredAspectProvidersArg, SkylarkList<?> providesArg,
      SkylarkList<?> fragments, SkylarkList<?> hostFragments, SkylarkList<?> toolchains, String doc,
      FuncallExpression ast, Environment funcallEnv) throws EvalException {
    return new FakeSkylarkAspect();
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

    public RuleDefinitionIdentifier() {
      super("RuleDefinitionIdentifier" + idCounter++);
    }

    @Override
    public boolean equals(@Nullable Object other) {
      // Use exact object matching.
      return this == other;
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
