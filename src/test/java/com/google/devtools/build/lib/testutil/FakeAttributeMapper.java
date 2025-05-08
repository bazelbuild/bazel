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

import static com.google.common.truth.Truth.assertWithMessage;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.DependencyFilter;
import com.google.devtools.build.lib.packages.PackageArgs;
import com.google.devtools.build.lib.packages.Type;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import javax.annotation.Nullable;

/** Faked implementation of {@link AttributeMap} for use in testing. */
public class FakeAttributeMapper implements AttributeMap {

  public static FakeAttributeMapper empty() {
    return new FakeAttributeMapper();
  }

  @Override
  public Label getLabel() {
    return Label.parseCanonicalUnchecked("//fake:rule");
  }

  @Override
  public boolean has(String attrName) {
    return false;
  }

  @Override
  public <T> boolean has(String attrName, Type<T> type) {
    return false;
  }

  @Override
  @Nullable
  public <T> T get(String attributeName, Type<T> type) {
      // Not specified in attributes or defaults
      assertWithMessage("Attribute " + attributeName + " not in attributes!").fail();
      return null;
  }

  @Override
  public boolean isConfigurable(String attributeName) {
    return false;
  }

  @Override
  public Iterable<String> getAttributeNames() {
    return ImmutableSet.of();
  }

  @Nullable
  @Override
  public Type<?> getAttributeType(String attrName) {
    return null;
  }

  @Nullable
  @Override
  public Attribute getAttributeDefinition(String attrName) {
    return null;
  }

  @Override
  public boolean isAttributeValueExplicitlySpecified(String attributeName) {
    return false;
  }

  @Override
  public void visitAllLabels(BiConsumer<Attribute, Label> consumer) {}

  @Override
  public void visitLabels(String attributeName, Consumer<Label> consumer) {}

  @Override
  public void visitLabels(DependencyFilter filter, BiConsumer<Attribute, Label> consumer) {}

  @Override
  public PackageArgs getPackageArgs() {
    return PackageArgs.DEFAULT;
  }
}
