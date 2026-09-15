// Copyright 2025 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.bazel.repository;

import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.bazel.bzlmod.AttributeValues;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import javax.annotation.Nullable;
import net.starlark.java.eval.Structure;
import net.starlark.java.spelling.SpellChecker;

/**
 * A fully-loaded repo definition, ready to be fetched. This class doubles as a Starlark value that
 * provides its own attribute struct.
 */
@AutoCodec
public record RepoDefinition(
    RepoRule repoRule, AttributeValues attrValues, String name, @Nullable String originalName)
    implements Structure {

  @Override
  public boolean isImmutable() {
    return true;
  }

  @Nullable
  @Override
  public Object getValue(String field) {
    if (field.equals("name")) {
      // Special case: `rctx.attr.name` can be used in place of `rctx.name`.
      return name;
    }
    @Nullable Object value = attrValues.attributes().get(field);
    if (value != null) {
      return value;
    }
    @Nullable Integer index = repoRule.attributeIndices().get(field);
    if (index == null) {
      return null;
    }
    return Attribute.valueToStarlark(repoRule.attributes().get(index).getDefaultValueUnchecked());
  }

  @Override
  public ImmutableSet<String> getFieldNames() {
    return Sets.union(repoRule.attributeIndices().keySet(), ImmutableSet.of("name"))
        .immutableCopy();
  }

  @Override
  public String getErrorMessageForUnknownField(String field) {
    return "unknown attribute " + field + SpellChecker.didYouMean(field, getFieldNames());
  }
}
