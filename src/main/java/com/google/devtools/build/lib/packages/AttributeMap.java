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
import com.google.devtools.build.lib.cmdline.Label;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import javax.annotation.Nullable;

/**
 * The interface for accessing a {@link Rule}'s attributes.
 *
 * <p>Since what an attribute lookup should return can be ambiguous (e.g. for configurable
 * attributes, should we return a configuration-resolved value or the original, unresolved
 * selection expression?), different implementations can apply different policies for how to
 * fulfill these methods. Calling code can then use the appropriate implementation for whatever
 * its particular needs are.
 */
public interface AttributeMap {
  /**
   * Returns the name of the rule; this is equivalent to {@code getLabel().getName()}.
   */
  String getName();

  /**
   * Returns the label of the rule.
   */
  Label getLabel();

  /**
   * Returns the name of the rule class.
   */
  String getRuleClassName();

  /**
   * Returns true if an attribute with the given name exists.
   */
  boolean has(String attrName);

  /**
   * Returns true if an attribute with the given name exists with the given type.
   *
   * <p>Don't use this version unless you really care about the type.
   */
  <T> boolean has(String attrName, Type<T> type);

  /**
   * Returns the value of the named rule attribute, which must be of the given type. This may
   * be null (for example, for an attribute with no default value that isn't explicitly set in
   * the rule - see {@link Type#getDefaultValue}).
   *
   * <p>If the rule doesn't have this attribute with the specified type, throws an
   * {@link IllegalArgumentException}.
   */
  @Nullable
  <T> T get(String attributeName, Type<T> type);

  /**
   * Returns the value of the named rule attribute if it exists, otherwise the given default value.
   * This may be null (for example, for an attribute with no default value that isn't explicitly set
   * in the rule - see {@link Type#getDefaultValue}).
   */
  default <T> T getOrDefault(String attributeName, Type<T> type, T defaultValue) {
    if (has(attributeName)) {
      return get(attributeName, type);
    }
    return defaultValue;
  }

  /**
   * Returns true if the given attribute is configurable for this rule instance, false
   * if it isn't configurable or doesn't exist.
   */
  boolean isConfigurable(String attributeName);

  /**
   * Returns the names of all attributes covered by this map.
   */
  Iterable<String> getAttributeNames();

  /**
   * Returns the type of the given attribute, if it exists. Otherwise returns null.
   */
  @Nullable
  Type<?> getAttributeType(String attrName);

  /**
   * Returns the attribute definition whose name is {@code attrName}, or null
   * if not found.
   */
  @Nullable Attribute getAttributeDefinition(String attrName);

  /**
   * Returns true iff the specified attribute is explicitly set in the target's definition (as
   * opposed to being omitted and taking on its default value from the rule definition).
   *
   * <p>Note that this returns true in the case where the attribute is explicitly set to the same
   * value as its default. Therefore, this method breaks encapsulation in the sense that it
   * describes *how* a target is defined rather than just *what* its attribute values are.
   *
   * <p>CAUTION: It is a good idea to avoid relying on this method if possible. It's confusing to
   * users that setting an attribute to (for example) an empty list is different from not setting it
   * at all. It also breaks some use cases, such as programmatically copying a target definition via
   * {@code native.existing_rules}. Specifically, the Starlark code doing the copying will observe
   * the attribute on the existing target whether or not it was set explicitly, and then set that
   * value explicitly on the new target. This can cause the two targets to behave differently, and
   * can be a difficult bug to track down. (See #7071, b/122596733).
   */
  boolean isAttributeValueExplicitlySpecified(String attributeName);

  /**
   * Invokes a consumer for labels of <em>every</em> attribute that contains labels in its value
   * (either by being a label or being a collection that includes labels).
   *
   * <p>If it is not necessary to visit labels of every attribute, prefer {@link
   * #visitLabels(Attribute, Consumer)} or {@link #visitLabels(DependencyFilter, BiConsumer)} for
   * better performance.
   */
  void visitAllLabels(BiConsumer<Attribute, Label> consumer);

  /** Same as {@link #visitAllLabels} but for a single attribute. */
  void visitLabels(Attribute attribute, Consumer<Label> consumer);

  /** Same as {@link #visitAllLabels} but for attributes matching a {@link DependencyFilter}. */
  void visitLabels(DependencyFilter filter, BiConsumer<Attribute, Label> consumer);

  // TODO(bazel-team): These methods are here to support computed defaults that inherit
  // package-level default values. Instead, we should auto-inherit and remove the computed
  // defaults. If we really need to give access to package-level defaults, we should come up with
  // a more generic interface.
  String getPackageDefaultHdrsCheck();

  Boolean getPackageDefaultTestOnly();

  String getPackageDefaultDeprecation();

  ImmutableList<String> getPackageDefaultCopts();
}
