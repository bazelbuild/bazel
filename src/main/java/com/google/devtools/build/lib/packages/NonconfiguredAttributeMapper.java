// Copyright 2021 The Bazel Authors. All rights reserved.
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
import javax.annotation.Nullable;

/**
 * {@link AttributeMap} implementation that triggers an {@link IllegalStateException} if called on
 * any attribute that is configured (has a select statement).
 *
 * <p>This is particularly useful for logic that doesn't have access to configurations - it protects
 * against undefined behavior in response to unexpected configuration-dependent inputs.
 *
 * <p>This is different from {@link NonconfigurableAttributeMapper} as it does not require the
 * attribute be declared as nonconfigurable. Instead, the attributes must not currently be
 * configured (that is, have select statements). This distinction is crucial when considering
 * Starlark-defined rules as they cannot declare attributes as nonconfigurable.
 */
public class NonconfiguredAttributeMapper extends AbstractAttributeMapper {
  private NonconfiguredAttributeMapper(Rule rule) {
    super(rule);
  }

  /**
   * Example usage:
   *
   * <pre>
   *   Label fooLabel = UnconfiguredAttributeMapper.of(rule).get("foo", Type.LABEL);
   * </pre>
   */
  public static NonconfiguredAttributeMapper of(Rule rule) {
    return new NonconfiguredAttributeMapper(rule);
  }

  @Nullable
  @Override
  public <T> T get(String attributeName, com.google.devtools.build.lib.packages.Type<T> type) {
    Attribute attribute = getAttributeDefinition(attributeName);
    if (attribute == null) {
      return null;
    }
    if (attribute.isConfigurable()) {
      Preconditions.checkState(
          getSelectorList(attributeName, type) == null,
          "Attribute '%s' is configured - not allowed here",
          attributeName);
    }
    return super.get(attributeName, type);
  }
}
