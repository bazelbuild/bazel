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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.cmdline.Label;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import javax.annotation.Nullable;

/**
 * An {@link AttributeMap} that delegates all behavior to another {@link AttributeMap}. Useful
 * for custom mappers that just want to override specific scenarios.
 */
public class DelegatingAttributeMapper implements AttributeMap {
  private final AttributeMap delegate;

  public DelegatingAttributeMapper(AttributeMap delegate) {
    this.delegate = Preconditions.checkNotNull(delegate);
  }

  @Override
  public String getName() {
    return delegate.getName();
  }

  @Override
  public Label getLabel() {
    return delegate.getLabel();
  }

  @Override
  public <T> T get(String attributeName, Type<T> type) {
    return delegate.get(attributeName, type);
  }

  @Override
  public boolean isConfigurable(String attributeName) {
    return delegate.isConfigurable(attributeName);
  }

  @Override
  public Iterable<String> getAttributeNames() {
    return delegate.getAttributeNames();
  }

  @Nullable
  @Override
  public Type<?> getAttributeType(String attrName) {
    return delegate.getAttributeType(attrName);
  }

  @Nullable
  @Override
  public Attribute getAttributeDefinition(String attrName) {
    return delegate.getAttributeDefinition(attrName);
  }

  @Override
  public boolean isAttributeValueExplicitlySpecified(String attributeName) {
    return delegate.isAttributeValueExplicitlySpecified(attributeName);
  }

  @Override
  public void visitAllLabels(BiConsumer<Attribute, Label> consumer) {
    delegate.visitAllLabels(consumer);
  }

  @Override
  public void visitLabels(String attributeName, Consumer<Label> consumer) {
    delegate.visitLabels(attributeName, consumer);
  }

  @Override
  public void visitLabels(DependencyFilter filter, BiConsumer<Attribute, Label> consumer) {
    delegate.visitLabels(filter, consumer);
  }

  @Override
  public PackageArgs getPackageArgs() {
    return delegate.getPackageArgs();
  }

  @Override
  public boolean has(String attrName) {
    return delegate.has(attrName);
  }

  @Override
  public <T> boolean has(String attrName, Type<T> type) {
    return delegate.has(attrName, type);
  }
}
