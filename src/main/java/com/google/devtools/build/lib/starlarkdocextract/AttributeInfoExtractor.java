// Copyright 2024 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.starlarkdocextract;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.StarlarkProviderIdentifier;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.packages.Types;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.AttributeInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.AttributeType;
import java.util.Optional;
import java.util.function.Consumer;
import net.starlark.java.eval.Starlark.InvalidStarlarkValueException;

/** Starlark API documentation extractor for a rule, macro, or aspect attribute. */
@VisibleForTesting
public final class AttributeInfoExtractor {

  @VisibleForTesting
  public static final AttributeInfo IMPLICIT_NAME_ATTRIBUTE_INFO =
      AttributeInfo.newBuilder()
          .setName("name")
          .setType(AttributeType.NAME)
          .setMandatory(true)
          .setDocString("A unique name for this target.")
          .build();

  @VisibleForTesting
  public static final AttributeInfo IMPLICIT_MACRO_NAME_ATTRIBUTE_INFO =
      AttributeInfo.newBuilder()
          .setName("name")
          .setType(AttributeType.NAME)
          .setMandatory(true)
          .setDocString(
              "A unique name for this macro instance. Normally, this is also the name for the"
                  + " macro's main or only target. The names of any other targets that this macro"
                  + " might create will be this name with a string suffix.")
          .build();

  @VisibleForTesting public static final String UNREPRESENTABLE_VALUE = "<unrepresentable value>";

  static AttributeInfo buildAttributeInfo(ExtractorContext context, Attribute attribute) {
    AttributeInfo.Builder builder =
        AttributeInfo.newBuilder()
            .setName(attribute.getPublicName())
            .setType(getAttributeType(context, attribute.getType(), attribute.getPublicName()))
            .setMandatory(attribute.isMandatory());
    Optional.ofNullable(attribute.getDoc()).ifPresent(builder::setDocString);
    if (!attribute.isConfigurable()) {
      builder.setNonconfigurable(true);
    }
    for (ImmutableSet<StarlarkProviderIdentifier> providerGroup :
        attribute.getRequiredProviders().getStarlarkProviders()) {
      // TODO(b/290788853): it is meaningless to require a provider on an attribute of a
      // repository rule or of a module extension tag.
      builder.addProviderNameGroup(
          ProviderNameGroupExtractor.buildProviderNameGroup(context, providerGroup));
    }

    if (!attribute.isMandatory()) {
      try {
        Object defaultValue = Attribute.valueToStarlark(attribute.getDefaultValueUnchecked());
        builder.setDefaultValue(context.labelRenderer().reprWithoutLabelConstructor(defaultValue));
      } catch (InvalidStarlarkValueException e) {
        builder.setDefaultValue(UNREPRESENTABLE_VALUE);
      }
    }
    return builder.build();
  }

  static void addDocumentableAttributes(
      ExtractorContext context, Iterable<Attribute> attributes, Consumer<AttributeInfo> builder) {
    for (Attribute attribute : attributes) {
      if (attribute.getName().equals(IMPLICIT_NAME_ATTRIBUTE_INFO.getName())) {
        // We inject our own IMPLICIT_NAME_ATTRIBUTE_INFO with better documentation.
        continue;
      }
      if ((attribute.starlarkDefined() || context.extractNonStarlarkAttrs())
          && attribute.isDocumented()
          && ExtractorContext.isPublicName(attribute.getPublicName())) {
        builder.accept(buildAttributeInfo(context, attribute));
      }
    }
  }

  static AttributeType getAttributeType(
      ExtractorContext context, Type<?> type, String attributePublicName) {
    if (type.equals(Type.INTEGER)) {
      return AttributeType.INT;
    } else if (type.equals(BuildType.LABEL)
        || type.equals(BuildType.NODEP_LABEL)
        || type.equals(BuildType.GENQUERY_SCOPE_TYPE)
        || type.equals(BuildType.DORMANT_LABEL)) {
      return AttributeType.LABEL;
    } else if (type.equals(Type.STRING) || type.equals(Type.STRING_NO_INTERN)) {
      if (attributePublicName.equals("name")) {
        return AttributeType.NAME;
      } else {
        return AttributeType.STRING;
      }
    } else if (type.equals(Types.STRING_LIST)) {
      return AttributeType.STRING_LIST;
    } else if (type.equals(Types.INTEGER_LIST)) {
      return AttributeType.INT_LIST;
    } else if (type.equals(BuildType.LABEL_LIST)
        || type.equals(BuildType.NODEP_LABEL_LIST)
        || type.equals(BuildType.GENQUERY_SCOPE_TYPE_LIST)
        || type.equals(BuildType.DORMANT_LABEL_LIST)) {
      return AttributeType.LABEL_LIST;
    } else if (type.equals(Type.BOOLEAN)) {
      return AttributeType.BOOLEAN;
    } else if (type.equals(BuildType.LABEL_KEYED_STRING_DICT)) {
      return AttributeType.LABEL_STRING_DICT;
    } else if (type.equals(Types.STRING_DICT)) {
      return AttributeType.STRING_DICT;
    } else if (type.equals(Types.STRING_LIST_DICT)) {
      return AttributeType.STRING_LIST_DICT;
    } else if (type.equals(BuildType.LABEL_DICT_UNARY)) {
      return AttributeType.LABEL_DICT_UNARY;
    } else if (type.equals(BuildType.OUTPUT)) {
      return AttributeType.OUTPUT;
    } else if (type.equals(BuildType.OUTPUT_LIST)) {
      return AttributeType.OUTPUT_LIST;
    } else if (type.equals(BuildType.LICENSE) || type.equals(BuildType.DISTRIBUTIONS)) {
      // TODO(https://github.com/bazelbuild/bazel/issues/6420): deprecated, disabled in Bazel by
      // default, broken and with almost no remaining users, so we don't have an AttributeType for
      // it. Until this type is removed, following the example of legacy Stardoc, pretend it's a
      // list of strings.
      return AttributeType.STRING_LIST;
    } else if (type.equals(BuildType.TRISTATE)) {
      // Given that the native TRISTATE type is not exposed to Starlark attr API, let's treat it as
      // an integer.
      return AttributeType.INT;
    }

    return AttributeType.UNKNOWN;
  }

  private AttributeInfoExtractor() {}
}
