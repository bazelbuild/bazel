// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.testutil;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.packages.AbstractAttributeMapper;
import com.google.devtools.build.lib.syntax.Type;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Faked implementation of {@link AbstractAttributeMapper} for use in testing.
 */
public class FakeAttributeMapper extends AbstractAttributeMapper {
  private final Map<String, FakeAttributeMapperEntry<?>> attrs;

  private FakeAttributeMapper(Map<String, FakeAttributeMapperEntry<?>> attrs) {
    super(null, null, null, null);
    this.attrs = ImmutableMap.copyOf(attrs);
  }

  @Override
  @Nullable
  public <T> T get(String attributeName, Type<T> type) {
    FakeAttributeMapperEntry<?> entry = attrs.get(attributeName);
    if (entry == null) {
      // Not specified in attributes or defaults
      assertWithMessage("Attribute " + attributeName + " not in attributes!").fail();
      return null;
    }

    return entry.validateAndGet(type);
  }

  @Override
  public boolean isAttributeValueExplicitlySpecified(String attributeName) {
    return attrs.containsKey(attributeName);
  }

  public static FakeAttributeMapper empty() {
    return builder().build();
  }

  public static Builder builder() {
    return new Builder();
  }

  /**
   * Builder to construct a {@link FakeAttributeMapper}. If no attributes are needed, use {@link
   * #empty()} instead.
   */
  public static class Builder {
    private final ImmutableMap.Builder<String, FakeAttributeMapperEntry<?>> mapBuilder =
        ImmutableMap.builder();

    private Builder() { }

    public Builder withStringList(String attribute, List<String> value) {
      mapBuilder.put(attribute, FakeAttributeMapperEntry.forStringList(value));
      return this;
    }

    public FakeAttributeMapper build() {
      return new FakeAttributeMapper(mapBuilder.build());
    }
  }

  private static class FakeAttributeMapperEntry<T> {
    private final Type<T> type;
    private final T value;

    private FakeAttributeMapperEntry(Type<T> type, T value) {
      this.type = type;
      this.value = value;
    }

    private static FakeAttributeMapperEntry<List<String>> forStringList(List<String> list) {
      return new FakeAttributeMapperEntry<>(Type.STRING_LIST, list);
    }

    private <U> U validateAndGet(Type<U> otherType) {
      assertThat(type).isSameInstanceAs(otherType);
      return otherType.cast(value);
    }
  }
}
