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

package com.google.devtools.build.benchmark;

import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.OptionsParsingException;

/**
 * A converter class that convert an input string to a {@link VersionFilter} object.
 */
public class VersionFilterConverter implements Converter<VersionFilter> {

  public VersionFilterConverter() {
    super();
  }

  @Override
  public VersionFilter convert(String input) throws OptionsParsingException {
    if (input.isEmpty()) {
      return null;
    }

    String[] parts = input.split("\\.\\.");
    if (parts.length != 2) {
      throw new OptionsParsingException("Error parsing version_filter option: no '..' found.");
    }
    if (parts[0].isEmpty()) {
      throw new OptionsParsingException(
          "Error parsing version_filter option: start version not found");
    }
    if (parts[1].isEmpty()) {
      throw new OptionsParsingException(
          "Error parsing version_filter option: end version not found");
    }

    return VersionFilter.create(parts[0], parts[1]);
  }

  @Override
  public String getTypeDescription() {
    return "A version filter in format: <start version>..<end version>";
  }

}