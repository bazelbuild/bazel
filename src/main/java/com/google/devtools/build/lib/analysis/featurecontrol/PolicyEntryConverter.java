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

package com.google.devtools.build.lib.analysis.featurecontrol;

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.OptionsParsingException;

/**
 * A converter which creates a PolicyEntry from a flag value.
 *
 * @deprecated This is deprecated because the dependency on the package group used to hold the
 *     whitelist is not accessible through blaze query. Use {@link Whitelist}.
 */
@Deprecated
public final class PolicyEntryConverter implements Converter<PolicyEntry> {
  @Override
  public PolicyEntry convert(String input) throws OptionsParsingException {
    int divider = input.indexOf('=');
    if (divider == -1) {
      throw new OptionsParsingException(
          "value must be of the form feature=label; missing =");
    }
    String feature = input.substring(0, divider);
    if (feature.isEmpty()) {
      throw new OptionsParsingException(
          "value must be of the form feature=label; feature cannot be empty");
    }
    String label = input.substring(divider + 1);
    if (label.isEmpty()) {
      throw new OptionsParsingException(
          "value must be of the form feature=label; label cannot be empty");
    }
    try {
      return PolicyEntry.create(feature, Label.parseAbsolute(label));
    } catch (LabelSyntaxException ex) {
      throw new OptionsParsingException(
          "value must be of the form feature=label; " + ex.getMessage());
    }
  }

  @Override
  public String getTypeDescription() {
    return "a feature=label pair";
  }
}
