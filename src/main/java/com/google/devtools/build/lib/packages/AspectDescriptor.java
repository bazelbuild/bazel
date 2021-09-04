// Copyright 2016 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.protobuf.TextFormat;
import java.util.Map;
import java.util.Objects;
import javax.annotation.Nullable;

/**
 * A wrapper for {@link AspectClass}, {@link AspectParameters}, {@code inheritedRequiredProviders}
 * and {@code inheritedAttributeAspects}
 *
 * <p>Used for dependency resolution.
 */
@Immutable
public final class AspectDescriptor {
  private final AspectClass aspectClass;
  private final AspectParameters aspectParameters;

  /**
   * Inherited required providers to enable aspects required by other aspects to be propagated along
   * with their main aspect until they can be applied. Null if the aspect does not inherit required
   * providers.
   */
  @Nullable private final RequiredProviders inheritedRequiredProviders;

  /**
   * Inherited required providers to enable aspects required by other aspects to be propagated along
   * with their main aspect based on its propagation attributes.
   */
  @Nullable private final ImmutableSet<String> inheritedAttributeAspects;

  /**
   * False if the inherited propagation information have no effect, i.e. {@code
   * inheritedRequiredProviders} accepts None and {@code inheritedAttributeAspects} is empty,
   * whether these values are actually inherited or not.
   */
  private final boolean inheritsPropagationInfo;

  public AspectDescriptor(
      AspectClass aspectClass,
      AspectParameters aspectParameters,
      RequiredProviders inheritedRequiredProviders,
      ImmutableSet<String> inheritedAttributeAspects) {
    this.aspectClass = aspectClass;
    this.aspectParameters = aspectParameters;
    if (isEmptyInheritedRequiredProviders(inheritedRequiredProviders)
        && isEmptyInheritedAttributeAspects(inheritedAttributeAspects)) {
      this.inheritsPropagationInfo = false;
      this.inheritedRequiredProviders = null;
      this.inheritedAttributeAspects = null;
    } else {
      this.inheritsPropagationInfo = true;
      this.inheritedRequiredProviders = inheritedRequiredProviders;
      this.inheritedAttributeAspects = inheritedAttributeAspects;
    }
  }

  public AspectDescriptor(AspectClass aspectClass, AspectParameters aspectParameters) {
    this(
        aspectClass,
        aspectParameters,
        /*inheritedRequiredProviders=*/ null,
        /*inheritedAttributeAspects=*/ null);
  }

  public AspectDescriptor(AspectClass aspectClass) {
    this(
        aspectClass,
        AspectParameters.EMPTY,
        /*inheritedRequiredProviders=*/ null,
        /*inheritedAttributeAspects=*/ null);
  }

  public AspectClass getAspectClass() {
    return aspectClass;
  }

  public AspectParameters getParameters() {
    return aspectParameters;
  }

  @Nullable
  public RequiredProviders getInheritedRequiredProviders() {
    return inheritedRequiredProviders;
  }

  @Nullable
  public ImmutableSet<String> getInheritedAttributeAspects() {
    if (!inheritsPropagationInfo) {
      return ImmutableSet.of(); // because returnning null means propagate through all attr aspects
    }
    return inheritedAttributeAspects;
  }

  public boolean satisfiesInheritedRequiredProviders(AdvertisedProviderSet advertisedProviders) {
    if (!inheritsPropagationInfo) {
      return false;
    }

    return inheritedRequiredProviders.isSatisfiedBy(advertisedProviders);
  }

  public boolean inheritedPropagateAlong(String attributeName) {
    if (!inheritsPropagationInfo) {
      return false;
    }

    if (inheritedAttributeAspects != null) {
      return inheritedAttributeAspects.contains(attributeName);
    }
    return true;
  }

  @Override
  public int hashCode() {
    return Objects.hash(
        aspectClass, aspectParameters, inheritedRequiredProviders, inheritedAttributeAspects);
  }

  @Override
  public boolean equals(Object obj) {
    if (obj == this) {
      return true;
    }

    if (!(obj instanceof AspectDescriptor)) {
      return false;
    }

    AspectDescriptor that = (AspectDescriptor) obj;
    return Objects.equals(aspectClass, that.aspectClass)
        && Objects.equals(aspectParameters, that.aspectParameters)
        && Objects.equals(inheritedRequiredProviders, that.inheritedRequiredProviders)
        && Objects.equals(inheritedAttributeAspects, that.inheritedAttributeAspects);
  }

  @Override
  public String toString() {
    return getDescription();
  }

  private static boolean isEmptyInheritedRequiredProviders(
      RequiredProviders inheritedRequiredProviders) {
    return (inheritedRequiredProviders == null
        || inheritedRequiredProviders.equals(RequiredProviders.acceptNoneBuilder().build()));
  }

  private static boolean isEmptyInheritedAttributeAspects(
      ImmutableSet<String> inheritedAttributeAspects) {
    return (inheritedAttributeAspects == null || inheritedAttributeAspects.isEmpty());
  }

  /**
   * Creates a presentable description of this aspect, available to Starlark via "Target.aspects".
   *
   * <p>The description is designed to be unique for each aspect descriptor, but not to be
   * parseable.
   */
  public String getDescription() {
    StringBuilder builder = new StringBuilder(aspectClass.getName());
    if (!aspectParameters.isEmpty()) {
      builder.append('[');
      ImmutableMultimap<String, String> attributes = aspectParameters.getAttributes();
      boolean first = true;
      for (Map.Entry<String, String> attribute : attributes.entries()) {
        if (!first) {
          builder.append(',');
        } else {
          first = false;
        }
        builder.append(attribute.getKey());
        builder.append("=\"");
        builder.append(TextFormat.escapeDoubleQuotesAndBackslashes(attribute.getValue()));
        builder.append("\"");
      }
      builder.append(']');
    }

    if (inheritsPropagationInfo) {
      if (inheritedAttributeAspects == null) {
        builder.append("[*]");
      } else {
        builder.append(inheritedAttributeAspects);
      }

      builder.append('[');
      builder.append(inheritedRequiredProviders);
      builder.append(']');
    }
    return builder.toString();
  }
}
