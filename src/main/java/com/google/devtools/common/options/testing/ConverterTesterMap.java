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

package com.google.devtools.common.options.testing;

import com.google.common.collect.ForwardingMap;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.common.options.Converter;
import java.util.Map;

/**
 * An immutable mapping from {@link Converter} classes to {@link ConverterTester}s which test them.
 *
 * <p>Note that the ConverterTesters within are NOT immutable.
 */
public final class ConverterTesterMap
    extends ForwardingMap<Class<? extends Converter<?>>, ConverterTester> {

  private final ImmutableMap<Class<? extends Converter<?>>, ConverterTester> delegate;

  private ConverterTesterMap(
      ImmutableMap<Class<? extends Converter<?>>, ConverterTester> delegate) {
    this.delegate = delegate;
  }

  @Override
  protected Map<Class<? extends Converter<?>>, ConverterTester> delegate() {
    return delegate;
  }

  /** A builder to construct new {@link ConverterTesterMap}s. */
  public static final class Builder {
    private final ImmutableMap.Builder<Class<? extends Converter<?>>, ConverterTester> delegate;

    public Builder() {
      this.delegate = ImmutableMap.builder();
    }

    /**
     * Adds a new ConverterTester, mapping it to the class of converter it tests. Only one tester
     * for each class is permitted; duplicates will cause {@link #build} to fail.
     */
    public Builder add(ConverterTester item) {
      delegate.put(item.getConverterClass(), item);
      return this;
    }

    /**
     * Adds the entries from the given {@link ConverterTesterMap}. Only one tester for each class is
     * permitted; duplicates will cause {@link #build} to fail.
     */
    public Builder addAll(ConverterTesterMap map) {
      // this is safe because we know the other map was constructed the same way this one was
      delegate.putAll(map);
      return this;
    }

    /**
     * Adds all of the ConverterTesters from the given iterable. Only one tester for each class is
     * permitted; duplicates will cause {@link #build} to fail.
     */
    public Builder addAll(Iterable<ConverterTester> items) {
      items.forEach(this::add);
      return this;
    }

    public ConverterTesterMap build() {
      return new ConverterTesterMap(delegate.build());
    }
  }
}
