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
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.syntax.Type;
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
   * Returns true iff the value of the specified attribute is explicitly set in the BUILD file (as
   * opposed to its default value). This also returns true if the value from the BUILD file is the
   * same as the default value.
   *
   * <p>It is probably a good idea to avoid this method in default value and implicit outputs
   * computation, because it is confusing that setting an attribute to an empty list (for example)
   * is different from not setting it at all.
   */
  boolean isAttributeValueExplicitlySpecified(String attributeName);

  /** Returns the {@link Location} at which the attribute was defined. */
  Location getAttributeLocation(String attrName);

  /** An interface which accepts {@link Attribute}s, used by {@link #visitLabels}. */
  interface AcceptsLabelAttribute {
    /**
     * Accept a (Label, Attribute) pair describing a dependency edge.
     *
     * @param label the target node of the (Rule, Label) edge. The source node should already be
     *     known.
     * @param attribute the attribute.
     */
    void acceptLabelAttribute(Label label, Attribute attribute) throws InterruptedException;
  }

  /**
   * For all attributes that contain labels in their values (either by *being* a label or being a
   * collection that includes labels), visits every label and notifies the specified observer at
   * each visit.
   */
  void visitLabels(AcceptsLabelAttribute observer) throws InterruptedException;

  // TODO(bazel-team): These methods are here to support computed defaults that inherit
  // package-level default values. Instead, we should auto-inherit and remove the computed
  // defaults. If we really need to give access to package-level defaults, we should come up with
  // a more generic interface.
  String getPackageDefaultHdrsCheck();

  Boolean getPackageDefaultTestOnly();

  String getPackageDefaultDeprecation();

  ImmutableList<String> getPackageDefaultCopts();
}
