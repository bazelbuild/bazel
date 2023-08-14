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

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.Interner;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.util.HashCodes;
import com.google.protobuf.TextFormat;
import java.util.Map;

/**
 * A pair of {@link AspectClass} and {@link AspectParameters}.
 *
 * <p>Used for dependency resolution.
 */
@Immutable
public final class AspectDescriptor {

  private static final Interner<AspectDescriptor> interner = BlazeInterners.newWeakInterner();

  @VisibleForTesting
  public static AspectDescriptor of(AspectClass aspectClass, AspectParameters aspectParameters) {
    return interner.intern(new AspectDescriptor(aspectClass, aspectParameters));
  }

  private final AspectClass aspectClass;
  private final AspectParameters aspectParameters;

  private AspectDescriptor(AspectClass aspectClass, AspectParameters aspectParameters) {
    this.aspectClass = checkNotNull(aspectClass);
    this.aspectParameters = checkNotNull(aspectParameters);
  }

  public AspectClass getAspectClass() {
    return aspectClass;
  }

  public AspectParameters getParameters() {
    return aspectParameters;
  }

  @Override
  public int hashCode() {
    return HashCodes.hashObjects(aspectClass, aspectParameters);
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
    return aspectClass.equals(that.aspectClass) && aspectParameters.equals(that.aspectParameters);
  }

  @Override
  public String toString() {
    return getDescription();
  }

  /**
   * Creates a presentable description of this aspect, available to Starlark via "Target.aspects".
   *
   * <p>The description is designed to be unique for each aspect descriptor, but not to be
   * parseable.
   */
  public String getDescription() {
    if (aspectParameters.isEmpty()) {
      return aspectClass.getName();
    }

    StringBuilder builder = new StringBuilder(aspectClass.getName());
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
    return builder.toString();
  }
}
