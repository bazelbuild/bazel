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

import com.google.common.base.Preconditions;

/**
 * {@link AttributeMap} implementation that triggers an {@link IllegalStateException} if called
 * on any attribute that supports configurable values, as determined by
 * {@link Attribute#isConfigurable()}.
 *
 * <p>This is particularly useful for logic that doesn't have access to configurations - it
 * protects against undefined behavior in response to unexpected configuration-dependent inputs.
 */
public class NonconfigurableAttributeMapper extends AbstractAttributeMapper {
  private NonconfigurableAttributeMapper(Rule rule) {
    super(rule);
  }

  /**
   * Example usage:
   *
   * <pre>
   *   Label fooLabel = NonconfigurableAttributeMapper.of(rule).get("foo", Type.LABEL);
   * </pre>
   */
  public static NonconfigurableAttributeMapper of (Rule rule) {
    return new NonconfigurableAttributeMapper(rule);
  }

  @Override
  public <T> T get(String attributeName, com.google.devtools.build.lib.packages.Type<T> type) {
    T attr = super.get(attributeName, type);
    Preconditions.checkState(!getAttributeDefinition(attributeName).isConfigurable(),
        "Attribute '%s' is potentially configurable - not allowed here", attributeName);
    return attr;
  }
}
