// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.common.options;

import com.google.common.collect.ImmutableList;
import javax.annotation.Nullable;

/**
 * The representation of a parsed option instance.
 *
 * <p>An option instance is distinct from the final value of an option, as multiple instances
 * provide values may be overridden or combined in some way.
 */
public final class ParsedOptionDescription {

  private final OptionDefinition optionDefinition;
  private final String commandLineForm;
  @Nullable private final String unconvertedValue;
  private final OptionInstanceOrigin origin;

  public ParsedOptionDescription(
      OptionDefinition optionDefinition,
      String commandLineForm,
      @Nullable String unconvertedValue,
      OptionInstanceOrigin origin) {
    this.optionDefinition = optionDefinition;
    this.commandLineForm = commandLineForm;
    this.unconvertedValue = unconvertedValue;
    this.origin = origin;
  }

  public OptionDefinition getOptionDefinition() {
    return optionDefinition;
  }

  public String getCommandLineForm() {
    return commandLineForm;
  }

  public boolean isBooleanOption() {
    return optionDefinition.getType().equals(boolean.class);
  }

  private OptionDocumentationCategory documentationCategory() {
    return optionDefinition.getDocumentationCategory();
  }

  private ImmutableList<OptionMetadataTag> metadataTags() {
    return ImmutableList.copyOf(optionDefinition.getOptionMetadataTags());
  }

  public boolean isDocumented() {
    return documentationCategory() != OptionDocumentationCategory.UNDOCUMENTED && !isHidden();
  }

  public boolean isHidden() {
    ImmutableList<OptionMetadataTag> tags = metadataTags();
    return tags.contains(OptionMetadataTag.HIDDEN) || tags.contains(OptionMetadataTag.INTERNAL);
  }

  public String getUnconvertedValue() {
    return unconvertedValue;
  }

  OptionPriority getPriority() {
    return origin.getPriority();
  }

  public String getSource() {
    return origin.getSource();
  }

  OptionDefinition getImplicitDependent() {
    return origin.getImplicitDependent();
  }

  OptionDefinition getExpandedFrom() {
    return origin.getExpandedFrom();
  }

  public boolean isExplicit() {
    return origin.getExpandedFrom() == null && origin.getImplicitDependent() == null;
  }

  public Object getConvertedValue() throws OptionsParsingException {
    Converter<?> converter = optionDefinition.getConverter();
    try {
      return converter.convert(unconvertedValue);
    } catch (OptionsParsingException e) {
      // The converter doesn't know the option name, so we supply it here by re-throwing:
      throw new OptionsParsingException(
          String.format("While parsing option %s: %s", commandLineForm, e.getMessage()), e);
    }
  }

  @Override
  public String toString() {
    StringBuilder result = new StringBuilder();
    result.append("option '").append(optionDefinition.getOptionName()).append("' ");
    result.append("set to '").append(unconvertedValue).append("' ");
    result.append("with priority ").append(origin.getPriority());
    if (origin.getSource() != null) {
      result.append(" and source '").append(origin.getSource()).append("'");
    }
    return result.toString();
  }

}
