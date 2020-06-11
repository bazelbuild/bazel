// Copyright 2020 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.buildjar.javac;

import static com.google.common.base.Preconditions.checkArgument;

import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Logic for handling non-standard javac flag {@code -Werror:}, which allows failing the compilation
 * for individual xlint warnings.
 */
public class WerrorCustomOption {

  private static final String WERROR = "-Werror:";

  private final ImmutableMap<String, Boolean> werrors;

  public WerrorCustomOption(ImmutableMap<String, Boolean> werrors) {
    this.werrors = werrors;
  }

  /** Returns true if the given lint category should be promoted to an error. */
  public boolean isEnabled(String lintCategory) {
    boolean all = werrors.containsKey("all");
    return werrors.getOrDefault(lintCategory, all);
  }

  static WerrorCustomOption create(String arg) {
    return new WerrorCustomOption.Builder(/* warningsAsErrorsDefault= */ ImmutableList.of())
        .process(arg)
        .build();
  }

  /** A builder for {@link WerrorCustomOption}s. */
  static class Builder {

    private final ImmutableList<String> warningsAsErrorsDefault;

    private final Map<String, Boolean> werrors = new LinkedHashMap<>();

    Builder(ImmutableList<String> warningsAsErrorsDefault) {
      this.warningsAsErrorsDefault = warningsAsErrorsDefault;
      // initialize list of werrors with the ones we want on by default
      for (String errorWarning : warningsAsErrorsDefault) {
        werrors.put(errorWarning, true);
      }
    }

    Builder all() {
      werrors.clear();
      werrors.put("all", true);
      return this;
    }

    Builder process(String flag) {
      checkArgument(flag.startsWith(WERROR), flag);
      for (String arg : Splitter.on(',').split(flag.substring(WERROR.length()))) {
        // Warnings with a '+' or '-' have an implicit '+'.
        if (arg.equals("+all") || arg.equals("all")) {
          werrors.clear();
          werrors.put("all", true);
        } else if (arg.equals("-all") || arg.equals("none")) {
          werrors.clear();
          werrors.put("none", true);
          for (String errorWarning : warningsAsErrorsDefault) {
            werrors.put(errorWarning, true);
          }
        } else if (arg.startsWith("-")) {
          String warning = arg.substring(1);
          if (!warningsAsErrorsDefault.contains(warning)) {
            werrors.put(warning, false);
          }
        } else {
          // '+' or raw warning category (implicit '+')
          String warning = arg.startsWith("+") ? arg.substring(1) : arg;
          werrors.put(warning, true);
        }
      }
      return this;
    }

    WerrorCustomOption build() {
      return new WerrorCustomOption(ImmutableMap.copyOf(werrors));
    }
  }

  /** Returns a normalized {@code -Werror:} flag. */
  @Override
  public String toString() {
    if (this.werrors.isEmpty()) {
      return "";
    }
    Map<String, Boolean> werrors = new LinkedHashMap<>(this.werrors);
    StringBuilder sb = new StringBuilder("-Werror:");
    if (werrors.containsKey("all")) {
      boolean b = werrors.remove("all");
      sb.append(b ? "" : "-").append("all,");
    }
    for (String warning : werrors.keySet()) {
      boolean b = werrors.get(warning);
      sb.append(b ? "" : "-").append(warning).append(",");
    }
    // delete trailing ","
    sb.deleteCharAt(sb.length() - 1);
    return sb.toString();
  }
}
