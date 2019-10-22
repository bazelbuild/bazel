// Copyright 2019 The Bazel Authors. All rights reserved.
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
//

package com.google.devtools.build.lib.bazel.rules.ninja.parser;

import com.google.common.collect.ImmutableSortedSet;
import com.google.devtools.build.lib.bazel.rules.ninja.file.GenericParsingException;
import com.google.devtools.build.lib.util.Pair;
import java.util.List;

public class NinjaVariableParser implements NinjaDeclarationParser<Pair<String, String>> {
  public static final NinjaVariableParser INSTANCE = new NinjaVariableParser();

  @Override
  public ImmutableSortedSet<NinjaKeyword> getKeywords() {
    return ImmutableSortedSet.of();
  }

  @Override
  public Pair<String, String> parse(List<String> lines) throws GenericParsingException {
    if (lines.size() != 1) {
      throw new GenericParsingException("Wrong variable format: " + String.join("\n", lines));
    }
    String s = lines.get(0);
    int index = s.indexOf('=');
    if (index < 0 || index == 0) {
      throw new GenericParsingException(String.format("Unknown token: '%s'", s));
    }
    return Pair.of(s.substring(0, index), s.substring(index));
  }
}
