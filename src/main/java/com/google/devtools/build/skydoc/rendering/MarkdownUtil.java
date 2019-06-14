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

package com.google.devtools.build.skydoc.rendering;

import com.google.common.base.Joiner;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.AttributeInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.AttributeType;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Contains a number of utility methods for markdown rendering.
 */
public final class MarkdownUtil {

  /**
   * Return a string representing the rule summary for the given rule with the given name.
   *
   * For example: 'my_rule(foo, bar)'.
   * The summary will contain hyperlinks for each attribute.
   */
  @SuppressWarnings("unused") // Used by markdown template.
  public String ruleSummary(String ruleName, RuleInfo ruleInfo) {
    List<String> attributeNames = ruleInfo.getAttributes().stream()
        .map(attr -> attr.getName())
        .collect(Collectors.toList());
    return summary(ruleName, attributeNames);
  }

  /**
   * Return a string representing the summary for the given provider with the given name.
   *
   * For example: 'MyInfo(foo, bar)'.
   * The summary will contain hyperlinks for each field.
   */
  @SuppressWarnings("unused") // Used by markdown template.
  public String providerSummary(String providerName, ProviderInfo providerInfo) {
    List<String> fieldNames = providerInfo.getFields().stream()
        .map(field -> field.getName())
        .collect(Collectors.toList());
    return summary(providerName, fieldNames);
  }

  /**
   * Return a string representing the summary for the given user-defined function.
   *
   * For example: 'my_func(foo, bar)'.
   * The summary will contain hyperlinks for each parameter.
   */
  @SuppressWarnings("unused") // Used by markdown template.
  public String funcSummary(UserDefinedFunctionInfo funcInfo) {
    List<String> paramNames = funcInfo.getParameters().stream()
        .map(param -> param.getName())
        .collect(Collectors.toList());
    return summary(funcInfo.getName(), paramNames);
  }

  private String summary(String functionName, List<String> paramNames) {
    List<String> paramLinks = paramNames.stream()
        .map(param ->
             String.format("<a href=\"#%s-%s\">%s</a>",
                 functionName, param, param))
        .collect(Collectors.toList());
    return String.format("%s(%s)", functionName, Joiner.on(", ").join(paramLinks));
  }

  /**
   * Returns a string describing the given attribute's type. The description consists of a hyperlink
   * if there is a relevant hyperlink to Bazel documentation available.
   */
  public String attributeTypeString(AttributeInfo attrInfo) {
    String typeLink;
    switch (attrInfo.getType()) {
      case LABEL:
      case LABEL_LIST:
      case OUTPUT:
        typeLink = "https://bazel.build/docs/build-ref.html#labels";
        break;
      case NAME:
        typeLink = "https://bazel.build/docs/build-ref.html#name";
        break;
      case STRING_DICT:
      case STRING_LIST_DICT:
      case LABEL_STRING_DICT:
        typeLink = "https://bazel.build/docs/skylark/lib/dict.html";
        break;
      default:
        typeLink = null;
        break;
    }
    if (typeLink == null) {
      return attributeTypeDescription(attrInfo.getType());
    } else {
      return String.format(
          "<a href=\"%s\">%s</a>", typeLink, attributeTypeDescription(attrInfo.getType()));
    }
  }

  public String mandatoryString(AttributeInfo attrInfo) {
    return attrInfo.getMandatory() ? "required" : "optional";
  }

  private String attributeTypeDescription(AttributeType attributeType) {
    switch (attributeType) {
      case NAME:
        return "Name";
      case INT:
        return "Integer";
      case LABEL:
        return "Label";
      case STRING:
        return "String";
      case STRING_LIST:
        return "List of strings";
      case INT_LIST:
        return "List of integers";
      case LABEL_LIST:
        return "List of labels";
      case BOOLEAN:
        return "Boolean";
      case LABEL_STRING_DICT:
        return "Dictionary: Label -> String";
      case STRING_DICT:
        return "Dictionary: String -> String";
      case STRING_LIST_DICT:
        return "Dictionary: String -> List of strings";
      case OUTPUT:
        return "Label";
      case OUTPUT_LIST:
        return "List of labels";
      case UNKNOWN:
      case UNRECOGNIZED:
        throw new IllegalArgumentException("Unhandled type " + attributeType);
    }
    throw new IllegalArgumentException("Unhandled type " + attributeType);
  }
}
