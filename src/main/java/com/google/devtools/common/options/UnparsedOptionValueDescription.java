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
 * The name and unparsed value of an option with additional metadata describing its priority,
 * source, whether it was set via an implicit dependency, and if so, by which other option.
 *
 * <p>Note that the unparsed value and the source parameters can both be null.
 */
public final class UnparsedOptionValueDescription {

  private final OptionDefinition optionDefinition;
  @Nullable private final String unparsedValue;
  private final OptionPriority priority;
  @Nullable private final String source;
  private final boolean explicit;

  public UnparsedOptionValueDescription(
      OptionDefinition optionDefinition,
      @Nullable String unparsedValue,
      OptionPriority priority,
      @Nullable String source,
      boolean explicit) {
    this.optionDefinition = optionDefinition;
    this.unparsedValue = unparsedValue;
    this.priority = priority;
    this.source = source;
    this.explicit = explicit;
  }

  public String getName() {
    return optionDefinition.getOptionName();
  }

  OptionDefinition getOptionDefinition() {
    return optionDefinition;
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

  boolean isExpansion() {
    return optionDefinition.isExpansionOption();
  }

  boolean isImplicitRequirement() {
    return optionDefinition.getImplicitRequirements().length > 0;
  }

  public String getUnparsedValue() {
    return unparsedValue;
  }

  OptionPriority getPriority() {
    return priority;
  }

  public String getSource() {
    return source;
  }

  public boolean isExplicit() {
    return explicit;
  }

  @Override
  public String toString() {
    StringBuilder result = new StringBuilder();
    result.append("option '").append(optionDefinition.getOptionName()).append("' ");
    result.append("set to '").append(unparsedValue).append("' ");
    result.append("with priority ").append(priority);
    if (source != null) {
      result.append(" and source '").append(source).append("'");
    }
    return result.toString();
  }
}

