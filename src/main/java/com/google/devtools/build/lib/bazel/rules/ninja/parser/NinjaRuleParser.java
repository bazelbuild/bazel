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

import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.ImmutableSortedSet;
import com.google.devtools.build.lib.bazel.rules.ninja.file.GenericParsingException;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaRule.ParameterName;
import java.util.List;

public class NinjaRuleParser implements NinjaDeclarationParser<NinjaRule> {
  public static final NinjaRuleParser INSTANCE = new NinjaRuleParser();

  @Override
  public ImmutableSortedSet<NinjaKeyword> getKeywords() {
    return ImmutableSortedSet.of(NinjaKeyword.rule);
  }

  @Override
  public NinjaRule parse(List<String> lines) throws GenericParsingException {
      ImmutableSortedMap.Builder<NinjaRule.ParameterName, String> parametersBuilder =
          ImmutableSortedMap.naturalOrder();
      boolean inPool = false;

      String name = readRuleName(lines.get(0), lines);
      parametersBuilder.put(ParameterName.name, name);
      for (int i = 1; i < lines.size(); i++) {
        String line = lines.get(i);
        if (line.startsWith("rule ")) {
          throw new GenericParsingException(
              String.format("Expected only one rule definition: '%s'", String.join("\n", lines)));
        } else if (line.startsWith("pool ")) {
          // skip the pool statement; if there was some rule before, it will be processed when the
          // definition of the next rule comes, or after iteration
          inPool = true;
        } else if (line.startsWith(" ") || line.startsWith("\t")) {
          if (inPool) {
            continue;
          }
          int idx = line.indexOf("=");
          if (idx >= 0) {
            String key = line.substring(0, idx).trim();
            String value = line.substring(idx + 1).trim();
            ParameterName parameterName = ParameterName.nullOrValue(key);
            if (parameterName == null) {
              throw new GenericParsingException(
                  String.format("Unknown rule parameter: '%s' in rule '%s'", key, name));
            }
            if (parameterName.isDefinedByTarget()) {
              throw new GenericParsingException(
                  String.format("Parameter '%s' should not be defined in rule '%s'", key, name));
            }
            parametersBuilder.put(parameterName, value);
          } else {
            throw new GenericParsingException(
                String.format("Can not parse rule parameter: '%s' in rule '%s'", line, name));
          }
        } else {
          throw new GenericParsingException(
              String.format("Unknown top-level keyword in rules section: '%s'", line));
        }
      }

      return new NinjaRule(checkAndBuildParameters(name, parametersBuilder));
    }

  private static String readRuleName(String line, List<String> lines) throws GenericParsingException {
    if (!line.startsWith("rule ")) {
      throw new GenericParsingException(
          String.format("Expected to find rule definition: '%s'", String.join("\n", lines)));
    }
    String[] parts = line.split(" ");
    if (parts.length != 2) {
      throw new GenericParsingException(String.format("Wrong rule name: '%s'", line));
    }
    return parts[1];
  }

  private static ImmutableSortedMap<ParameterName, String> checkAndBuildParameters(String name,
      ImmutableSortedMap.Builder<ParameterName, String> parametersBuilder)
      throws GenericParsingException {
    ImmutableSortedMap<ParameterName, String> parameters = parametersBuilder.build();
    if (!parameters.containsKey(ParameterName.command)) {
      throw new GenericParsingException(
          String.format("Rule %s should have command, rule text: '%s'", name, parameters));
    }
    return parameters;
  }
}
