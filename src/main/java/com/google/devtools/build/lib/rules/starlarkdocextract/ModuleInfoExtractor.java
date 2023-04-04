// Copyright 2023 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.starlarkdocextract;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.starlark.StarlarkRuleClassFunctions.StarlarkRuleFunction;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.StarlarkDefinedAspect;
import com.google.devtools.build.lib.packages.StarlarkProvider;
import com.google.devtools.build.lib.packages.StarlarkProviderIdentifier;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.skydoc.rendering.DocstringParseException;
import com.google.devtools.build.skydoc.rendering.FunctionUtil;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.AspectInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.AttributeInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.AttributeType;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.ModuleInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.ProviderFieldInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.ProviderInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.ProviderNameGroup;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.RuleInfo;
import java.util.Map;
import java.util.Optional;
import java.util.function.Predicate;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Module;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.StarlarkFunction;
import net.starlark.java.eval.Structure;

/** API documentation extractor for a compiled, loaded Starlark module. */
final class ModuleInfoExtractor {
  @VisibleForTesting
  static final AttributeInfo IMPLICIT_NAME_ATTRIBUTE_INFO =
      AttributeInfo.newBuilder()
          .setName("name")
          .setType(AttributeType.NAME)
          .setMandatory(true)
          .setDocString("A unique name for this target.")
          .build();

  // TODO(b/276733504): do we want to add an implicit repo_mapping attribute for repo rules, as
  // FakeRepositoryModule currently does?

  /**
   * Extracts structured documentation for the exported symbols (meaning symbols whose first
   * character is alphabetic) of a given module.
   *
   * @param isWantedName a filter applied to symbols; only those symbols for which the filter
   *     returns true will be documented
   */
  public static ModuleInfo extractFrom(Module module, Predicate<String> isWantedName)
      throws ExtractionException {
    ModuleInfo.Builder builder = ModuleInfo.newBuilder();
    Optional.ofNullable(module.getDocumentation()).ifPresent(builder::setModuleDocstring);
    for (var entry : module.getGlobals().entrySet()) {
      String topLevelSymbol = entry.getKey();
      if (isExportableName(topLevelSymbol) && isWantedName.test(topLevelSymbol)) {
        addInfo(builder, topLevelSymbol, entry.getValue());
      }
    }
    return builder.build();
  }

  private static boolean isExportableName(String name) {
    return name.length() > 0 && Character.isAlphabetic(name.charAt(0));
  }

  /** An exception indicating that the module's API documentation could not be extracted. */
  public static class ExtractionException extends Exception {
    public ExtractionException(String message) {
      super(message);
    }

    public ExtractionException(Throwable cause) {
      super(cause);
    }

    public ExtractionException(String message, Throwable cause) {
      super(message, cause);
    }
  }

  /**
   * @param builder proto builder to which to append documentation
   * @param name the name under which the value is exported by the module; for example, "foo.bar"
   *     for field bar of exported struct foo
   * @param value documentable Starlark value
   */
  private static void addInfo(ModuleInfo.Builder builder, String name, Object value)
      throws ExtractionException {
    // Note that may be exported under a different name than its getName() value.
    if (value instanceof StarlarkRuleFunction) {
      addRuleInfo(builder, name, (StarlarkRuleFunction) value);
    } else if (value instanceof StarlarkProvider) {
      addProviderInfo(builder, name, (StarlarkProvider) value);
    } else if (value instanceof StarlarkFunction) {
      try {
        builder.addFuncInfo(FunctionUtil.fromNameAndFunction(name, (StarlarkFunction) value));
      } catch (DocstringParseException e) {
        throw new ExtractionException(e);
      }
    } else if (value instanceof StarlarkDefinedAspect) {
      addAspectInfo(builder, name, (StarlarkDefinedAspect) value);
    } else if (value instanceof Structure) {
      addStructureInfo(builder, name, (Structure) value);
    }
    // else the value is a constant (string, list etc.), and we currently don't have a convention
    // for associating a doc string with one - so we don't emit documentation for it.
    // TODO(b/276733504): should we recurse into dicts to search for documentable values?
  }

  private static void addStructureInfo(ModuleInfo.Builder builder, String name, Structure structure)
      throws ExtractionException {
    for (String fieldName : structure.getFieldNames()) {
      if (isExportableName(fieldName)) {
        try {
          Object fieldValue = structure.getValue(fieldName);
          if (fieldValue != null) {
            addInfo(builder, String.format("%s.%s", name, fieldName), fieldValue);
          }
        } catch (EvalException e) {
          throw new ExtractionException(
              String.format("in struct %s field %s: failed to read value", name, fieldName), e);
        }
      }
    }
  }

  private static AttributeType getAttributeType(Attribute attribute, String where)
      throws ExtractionException {
    Type<?> type = attribute.getType();
    if (type.equals(Type.INTEGER)) {
      return AttributeType.INT;
    } else if (type.equals(BuildType.LABEL)) {
      return AttributeType.LABEL;
    } else if (type.equals(Type.STRING)) {
      if (attribute.getPublicName().equals("name")) {
        return AttributeType.NAME;
      } else {
        return AttributeType.STRING;
      }
    } else if (type.equals(Type.STRING_LIST)) {
      return AttributeType.STRING_LIST;
    } else if (type.equals(Type.INTEGER_LIST)) {
      return AttributeType.INT_LIST;
    } else if (type.equals(BuildType.LABEL_LIST)) {
      return AttributeType.LABEL_LIST;
    } else if (type.equals(Type.BOOLEAN)) {
      return AttributeType.BOOLEAN;
    } else if (type.equals(BuildType.LABEL_KEYED_STRING_DICT)) {
      return AttributeType.LABEL_STRING_DICT;
    } else if (type.equals(Type.STRING_DICT)) {
      return AttributeType.STRING_DICT;
    } else if (type.equals(Type.STRING_LIST_DICT)) {
      return AttributeType.STRING_LIST_DICT;
    } else if (type.equals(BuildType.OUTPUT)) {
      return AttributeType.OUTPUT;
    } else if (type.equals(BuildType.OUTPUT_LIST)) {
      return AttributeType.OUTPUT_LIST;
    }

    throw new ExtractionException(
        String.format(
            "in %s attribute %s: unsupported type %s",
            where, attribute.getPublicName(), type.getClass().getSimpleName()));
  }

  private static AttributeInfo buildAttributeInfo(Attribute attribute, String where)
      throws ExtractionException {
    AttributeInfo.Builder builder = AttributeInfo.newBuilder();
    builder.setName(attribute.getPublicName());
    Optional.ofNullable(attribute.getDoc()).ifPresent(builder::setDocString);
    builder.setType(getAttributeType(attribute, where));
    builder.setMandatory(attribute.isMandatory());
    for (ImmutableSet<StarlarkProviderIdentifier> providerGroup :
        attribute.getRequiredProviders().getStarlarkProviders()) {
      ProviderNameGroup.Builder providerNameGroupBuilder = ProviderNameGroup.newBuilder();
      for (StarlarkProviderIdentifier provider : providerGroup) {
        // TODO(b/276733504): if this module exports a provider under a different name or in a
        // namespace, document it under that exported name rather than the provider's key name.
        providerNameGroupBuilder.addProviderName(provider.toString());
      }
      builder.addProviderNameGroup(providerNameGroupBuilder.build());
    }

    if (!attribute.isMandatory()) {
      // TODO(b/276733504): deeply canonicalize label objects to strings
      Object defaultValue = attribute.getDefaultValueUnchecked();
      builder.setDefaultValue(
          new Printer().repr(Attribute.valueToStarlark(defaultValue)).toString());
    }
    return builder.build();
  }

  private static void addRuleInfo(
      ModuleInfo.Builder moduleInfoBuilder, String exportedName, StarlarkRuleFunction ruleFunction)
      throws ExtractionException {
    RuleInfo.Builder ruleInfoBuilder = RuleInfo.newBuilder();
    // Allow rules to be exported under a different name (e.g. in a struct).
    ruleInfoBuilder.setRuleName(exportedName);
    ruleFunction.getDocumentation().ifPresent(ruleInfoBuilder::setDocString);
    RuleClass ruleClass = ruleFunction.getRuleClass();
    ruleInfoBuilder.addAttribute(IMPLICIT_NAME_ATTRIBUTE_INFO); // name comes first
    for (Attribute attribute : ruleClass.getAttributes()) {
      if (attribute.starlarkDefined()
          && attribute.isDocumented()
          && isExportableName(attribute.getPublicName())) {
        ruleInfoBuilder.addAttribute(buildAttributeInfo(attribute, "rule " + exportedName));
      }
    }
    moduleInfoBuilder.addRuleInfo(ruleInfoBuilder);
  }

  private static void addProviderInfo(
      ModuleInfo.Builder moduleInfoBuilder, String exportedName, StarlarkProvider provider) {
    ProviderInfo.Builder providerInfoBuilder = ProviderInfo.newBuilder();
    // Allow providers to be exported under a different name (e.g. in a struct).
    providerInfoBuilder.setProviderName(exportedName);
    provider.getDocumentation().ifPresent(providerInfoBuilder::setDocString);
    ImmutableMap<String, Optional<String>> schemaWithDocumentation =
        provider.getSchemaWithDocumentation();
    if (schemaWithDocumentation != null) {
      for (Map.Entry<String, Optional<String>> entry : schemaWithDocumentation.entrySet()) {
        if (isExportableName(entry.getKey())) {
          ProviderFieldInfo.Builder fieldInfoBuilder = ProviderFieldInfo.newBuilder();
          fieldInfoBuilder.setName(entry.getKey());
          entry.getValue().ifPresent(fieldInfoBuilder::setDocString);
          providerInfoBuilder.addFieldInfo(fieldInfoBuilder.build());
        }
      }
    }
    moduleInfoBuilder.addProviderInfo(providerInfoBuilder);
  }

  private static void addAspectInfo(
      ModuleInfo.Builder moduleInfoBuilder, String exportedName, StarlarkDefinedAspect aspect)
      throws ExtractionException {
    AspectInfo.Builder aspectInfoBuilder = AspectInfo.newBuilder();
    // Allow aspects to be exported under a different name (e.g. in a struct).
    aspectInfoBuilder.setAspectName(exportedName);
    aspect.getDocumentation().ifPresent(aspectInfoBuilder::setDocString);
    aspectInfoBuilder.addAllAspectAttribute(aspect.getAttributeAspects());
    aspectInfoBuilder.addAttribute(IMPLICIT_NAME_ATTRIBUTE_INFO); // name comes first
    for (Attribute attribute : aspect.getAttributes()) {
      if (isExportableName(attribute.getPublicName())) {
        aspectInfoBuilder.addAttribute(buildAttributeInfo(attribute, "aspect " + exportedName));
      }
    }
    moduleInfoBuilder.addAspectInfo(aspectInfoBuilder);
  }

  private ModuleInfoExtractor() {}
}
