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

import com.google.devtools.build.lib.analysis.config.Fragment;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import java.util.List;
import java.util.Objects;

/**
 * Data structure defining a {@link Fragment} for the purpose of creating user output.
 *
 * <p>{@link Fragment} is a Java object representation of a domain-specific "piece" of configuration
 * (like "C++-related configuration"). It depends on one or more {@link FragmentOptions}, which are
 * the <code>--flag=value</code> pairs that key configurations.
 *
 * <p>See {@link FragmentOptionsForOutput} and {@link ConfigurationForOutput} for further details.
 */
public class FragmentForOutput {
  private final String name;
  // We store the name of the associated FragmentOptions instead of FragmentOptionsForOutput
  // objects because multiple fragments may use the same FragmentOptions and we don't want to list
  // it multiple times.
  private final List<String> fragmentOptions;

  public FragmentForOutput(String name, List<String> fragmentOptions) {
    this.name = name;
    this.fragmentOptions = fragmentOptions;
  }

  public String getName() {
    return name;
  }

  /** The names of the FragmentOptions, sorted. */
  public List<String> getFragmentOptions() {
    return fragmentOptions;
  }

  @Override
  public boolean equals(Object o) {
    if (o instanceof FragmentForOutput) {
      FragmentForOutput other = (FragmentForOutput) o;
      return other.name.equals(name) && other.fragmentOptions.equals(fragmentOptions);
    }
    return false;
  }

  @Override
  public int hashCode() {
    return Objects.hash(name, fragmentOptions);
  }
}
