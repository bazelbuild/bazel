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
package com.google.devtools.build.lib.packages;

import static com.google.common.collect.Iterables.getOnlyElement;

import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.Multimap;

import java.util.Objects;

/**
 * Objects of this class contain values of some attributes of rules. Used for passing this
 * information to the aspects.
 */
public final class AspectParameters {
  private final ImmutableMultimap<String, String> attributes;

  private AspectParameters(Multimap<String, String> attributes) {
    this.attributes = ImmutableMultimap.copyOf(attributes);
  }

  public static final AspectParameters EMPTY = new AspectParameters.Builder().build();

  /**
   * A builder for @{link {@link AspectParameters} class.
   */
  public static class Builder {
    private final Multimap<String, String> attributes = ArrayListMultimap.create();

    /**
     * Adds a new pair of attribute-value.
     */
    public Builder addAttribute(String name, String value) {
      attributes.put(name, value);
      return this;
    }

    /**
     * Creates a new instance of {@link AspectParameters} class.
     */
    public AspectParameters build() {
      return new AspectParameters(attributes);
    }
  }

  /**
   * Returns collection of values for specified key, or null if key is missing.
   */
  public ImmutableCollection<String> getAttribute(String key) {
    return attributes.get(key);
  }

  /**
   * Similar to {@link #getAttribute}}, but asserts that there's only one value for the provided
   * key.
   * Uses Guava's {@link Iterables#getOnlyElement}, which may throw exceptions if there isn't
   * exactly one element.
   */
  public String getOnlyValueOfAttribute(String key) {
    return getOnlyElement(getAttribute(key));
  }

  public boolean isEmpty() {
    return this.equals(AspectParameters.EMPTY);
  }

  @Override
  public boolean equals(Object other) {
    if (this == other) {
      return true;
    }
    if (!(other instanceof AspectParameters)) {
      return false;
    }
    AspectParameters that = (AspectParameters) other;
    return Objects.equals(this.attributes, that.attributes);
  }

  @Override
  public int hashCode() {
    return Objects.hashCode(attributes);
  }

  @Override
  public String toString() {
    return attributes.toString();
  }
}
