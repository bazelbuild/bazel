// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.apple;

import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.OptionsParsingException;

/**
 * Converter for options representing {@link DottedVersion} values.
 */
public class DottedVersionConverter implements Converter<DottedVersion.Option> {

  @Override
  public DottedVersion.Option convert(String input) throws OptionsParsingException {
    try {
      return DottedVersion.option(DottedVersion.fromString(input));
    } catch (DottedVersion.InvalidDottedVersionException e) {
      throw new OptionsParsingException(e.getMessage());
    }
  }

  @Override
  public String getTypeDescription() {
    return "a dotted version (for example '2.3' or '3.3alpha2.4')";
  }
}
