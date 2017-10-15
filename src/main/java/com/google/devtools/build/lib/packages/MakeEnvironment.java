// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.packages;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Environment for varref variables (formerly called "Makefile
 * variables").
 *
 * <p><code>update</code> emulates a very restricted subset of the behaviour of
 * GNU Make's environment. In particular, does not attempt to simulate Make's
 * complex range of assigment operators.
 */
@Immutable @ThreadSafe
public class MakeEnvironment {
  /**
   *  The platform set regexp that matches all platforms.  Canonical.
   */
  public static final String MATCH_ANY = ".*";

  // A platform-specific binding of a value for a given variable.
  static class Binding {
    private final String value;
    private final String platformSetRegexp;

    Binding(String value, String platformSetRegexp) {
      this.value = value;
      this.platformSetRegexp = platformSetRegexp;
    }

    @Override
    public String toString() {
      return value + " (" + platformSetRegexp + ")";
    }

    String getValue() {
      return value;
    }

    String getPlatformSetRegexp() {
      return platformSetRegexp;
    }
  }

  // Maps each variable name to the [short] list of platform-specific bindings
  // for it. The first matching binding is definitive.
  private final ImmutableMap<String, ImmutableList<Binding>> env;

  private MakeEnvironment(ImmutableMap<String, ImmutableList<Binding>> env) {
    this.env = env;
  }

  /**
   * @return the "Make" value from the environment whose name is "varname", or
   *   null iff the variable is not defined in the environment.
   */
  public String lookup(String varname, String platform) {
    List<Binding> bindings = env.get(varname);
    if (bindings == null) {
      return null;
    }
    // First, look for a matching non-default binding.
    // (The order in 'bindings' is the reverse of the order of vardefs in the BUILD file, so
    // the first match in this for loop selects the last matching definition in the BUILD file.)
    for (Binding binding : bindings) {
      if (!binding.platformSetRegexp.equals(MATCH_ANY) &&
          platform.matches(binding.platformSetRegexp)) {
        return binding.value;
      }
    }
    // If we didn't find a matching non-default binding,
    // try using the last default binding.
    for (Binding binding : bindings) {
      if (binding.platformSetRegexp.equals(MATCH_ANY)) {
        return binding.value;
      }
    }
    return null;
  }

  Map<String, ImmutableList<Binding>> getBindings() {
    return env;
  }

  /**
   * Interface for creating a MakeEnvironment, settings its environment values,
   * and exposing it in immutable state.
   */
  public static class Builder {
    private final Map<String, LinkedList<Binding>> env = new HashMap<>();
    private Map<String, String> platformSets = ImmutableMap.<String, String>of("any", MATCH_ANY);

    /**
     * Performs an update of Makefile variable 'var' to value 'value' for all
     * platforms belonging to the specified 'platformSetRegexp'. Corresponds to
     * vardef. We explicitly do not support the various complex nuances of
     * Make's assignment operator.
     *
     * <p>The most recent binding for a particular variable takes precedence, even if
     * a more specific binding came earlier.
     *
     * @param varname the name of the Makefile variable;
     * @param value the string value to assign;
     * @param platformSetRegexp a set of platforms for which this variable definition
     *        should take effect.  This is expressed as a regexp over gplatform
     *        strings.
     */
    public void update(String varname, String value, String platformSetRegexp) {
      if (varname == null || value == null || platformSetRegexp == null) {
        throw new NullPointerException();
      }
      LinkedList<Binding> bindings = env.computeIfAbsent(varname, k -> new LinkedList<>());
      // push new bindings onto head of list (=> most recent binding is
      // definitive):
      bindings.addFirst(new Binding(value, platformSetRegexp));
    }

    /**
     * Sets the nickname to regexp mapping for <tt>vardef</tt>.
     */
    public void setPlatformSetRegexps(Map<String, String> sets) {
      this.platformSets = sets;
    }

    @Nullable
    public String getPlatformSetRegexp(String nickname) {
      return this.platformSets.get(nickname);
    }

    /**
     * Returns a new MakeEnvironment with environment settings corresponding
     * to the cumulative results of this builder's {@link #update} calls.
     */
    public MakeEnvironment build() {
      Map<String, ImmutableList<Binding>> newMap = new HashMap<>();
      for (Map.Entry<String, LinkedList<Binding>> entry : env.entrySet()) {
        newMap.put(entry.getKey(), ImmutableList.copyOf(entry.getValue()));
      }
      return new MakeEnvironment(ImmutableMap.copyOf(newMap));
    }
  }

  @Override
  public int hashCode() {
    throw new UnsupportedOperationException();
  }

  @Override
  public boolean equals(Object that) {
    throw new UnsupportedOperationException();
  }

  @Override
  public String toString() {
    return "MakeEnvironment=" + env;
  }
}
