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
package com.google.devtools.build.lib.exec;

import com.google.common.collect.Maps;
import com.google.devtools.build.lib.util.RegexFilter;
import com.google.devtools.build.lib.util.RegexFilter.RegexFilterConverter;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.Map;

/** A converter for options of the form RegexFilter=String. */
public class RegexFilterAssignmentConverter implements Converter<Map.Entry<RegexFilter, String>> {

  @Override
  public Map.Entry<RegexFilter, String> convert(String input) throws OptionsParsingException {
    int pos = input.indexOf('=');
    if (pos <= 0) {
      throw new OptionsParsingException("Must be in the form of a 'regex=value' assignment");
    }
    String value = input.substring(pos + 1);
    RegexFilter filter = new RegexFilterConverter().convert(input.substring(0, pos));
    return Maps.immutableEntry(filter, value);
  }

  @Override
  public String getTypeDescription() {
    return "a '<RegexFilter>=value' assignment";
  }
}
