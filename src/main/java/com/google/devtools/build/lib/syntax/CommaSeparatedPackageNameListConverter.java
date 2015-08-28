// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.syntax;

import com.google.common.base.Splitter;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.OptionsParsingException;

import java.util.List;

/**
 * A converter from strings containing comma-separated names of packages to lists of strings.
 */
public class CommaSeparatedPackageNameListConverter
    implements Converter<List<PackageIdentifier>> {

  private static final Splitter SPACE_SPLITTER = Splitter.on(',');

  @Override
  public List<PackageIdentifier> convert(String input) throws OptionsParsingException {
    if (Strings.isNullOrEmpty(input)) {
      return ImmutableList.of();
    }
    ImmutableList.Builder<PackageIdentifier> list = ImmutableList.builder();
    for (String s : SPACE_SPLITTER.split(input)) {
      try {
        list.add(PackageIdentifier.parse(s));
      } catch (TargetParsingException e) {
        throw new OptionsParsingException(e.getMessage());
      }
    }
    return list.build();
  }

  @Override
  public String getTypeDescription() {
    return "comma-separated list of package names";
  }

}
