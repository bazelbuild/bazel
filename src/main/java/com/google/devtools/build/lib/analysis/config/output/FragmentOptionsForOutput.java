// Copyright 2024 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis.config.output;

import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import java.util.Objects;
import java.util.Set;
import java.util.SortedMap;

/**
 * Data structure defining a {@link FragmentOptions} for creating user output.
 *
 * <p>See {@link FragmentForOutput} and {@link ConfigurationForOutput} for further details.
 */
public class FragmentOptionsForOutput {
  private final String name;
  private final SortedMap<String, String> options;

  public FragmentOptionsForOutput(String name, SortedMap<String, String> options) {
    this.name = name;
    this.options = options;
  }

  public String getName() {
    return name;
  }

  public SortedMap<String, String> getOptions() {
    return options;
  }

  public Set<String> optionNames() {
    return this.options.keySet();
  }

  public String getOption(String optionName) {
    return this.options.get(optionName);
  }

  @Override
  public boolean equals(Object o) {
    if (o instanceof FragmentOptionsForOutput) {
      FragmentOptionsForOutput other = (FragmentOptionsForOutput) o;
      return other.name.equals(name) && other.options.equals(options);
    }
    return false;
  }

  @Override
  public int hashCode() {
    return Objects.hash(name, options);
  }
}
