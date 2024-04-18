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
package com.google.devtools.build.skyframe;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.events.Reportable;
import java.util.Objects;
import javax.annotation.Nullable;

/** Encapsulation of data stored by {@link NodeEntry} when the value has finished building. */
public abstract class ValueWithMetadata implements SkyValue {
  protected final SkyValue value;

  private static final NestedSet<Reportable> NO_EVENTS =
      NestedSetBuilder.emptySet(Order.STABLE_ORDER);

  private ValueWithMetadata(SkyValue value) {
    this.value = value;
  }

  /** Builds a value entry that has an error (and no value). */
  public static ValueWithMetadata error(
      ErrorInfo errorInfo, NestedSet<Reportable> transitiveEvents) {
    return (ValueWithMetadata) normal(null, errorInfo, transitiveEvents);
  }

  /**
   * Builds a SkyValue that has a value, and possibly an error, and possibly events/postables. If it
   * has only a value, returns just the value in order to save memory.
   */
  public static SkyValue normal(
      @Nullable SkyValue value,
      @Nullable ErrorInfo errorInfo,
      NestedSet<Reportable> transitiveEvents) {
    Preconditions.checkState(
        value != null || errorInfo != null, "Value and error cannot both be null");
    if (errorInfo == null) {
      return transitiveEvents.isEmpty()
          ? value
          : ValueWithEvents.createValueWithEvents(value, transitiveEvents);
    }
    return new ErrorInfoValue(errorInfo, value, transitiveEvents);
  }

  public boolean hasError() {
    return false;
  }

  @Nullable
  SkyValue getValue() {
    return value;
  }

  @Nullable
  abstract ErrorInfo getErrorInfo();

  public abstract NestedSet<Reportable> getTransitiveEvents();

  /** Implementation of {@link ValueWithMetadata} for the value case. */
  @VisibleForTesting
  public static class ValueWithEvents extends ValueWithMetadata {
    private final NestedSet<Reportable> transitiveEvents;

    private ValueWithEvents(SkyValue value, NestedSet<Reportable> transitiveEvents) {
      super(Preconditions.checkNotNull(value));
      this.transitiveEvents = Preconditions.checkNotNull(transitiveEvents);
    }

    private static ValueWithEvents createValueWithEvents(
        SkyValue value, NestedSet<Reportable> transitiveEvents) {
      if (value instanceof NotComparableSkyValue) {
        return new NotComparableValueWithEvents(value, transitiveEvents);
      } else {
        return new ValueWithEvents(value, transitiveEvents);
      }
    }

    @Nullable
    @Override
    ErrorInfo getErrorInfo() {
      return null;
    }

    @Override
    public NestedSet<Reportable> getTransitiveEvents() {
      return transitiveEvents;
    }

    /**
     * We override equals so that if the same value is written to a {@link NodeEntry} twice, it can
     * verify that the two values are equal, and avoid incrementing its version.
     */
    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (!(o instanceof ValueWithEvents that)) {
        return false;
      }

      // Shallow equals is a middle ground between using default equals, which might miss
      // nested sets with the same elements, and deep equality checking, which would be expensive.
      // All three choices are sound, since shallow equals and default equals are more
      // conservative than deep equals. Using shallow equals means that we may unnecessarily
      // consider some values unequal that are actually equal, but this is still a net win over
      // deep equals.
      return value.equals(that.value) && transitiveEvents.shallowEquals(that.transitiveEvents);
    }

    @Override
    public int hashCode() {
      return 31 * value.hashCode() + transitiveEvents.shallowHashCode();
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(this)
          .add("value", value)
          .add("transitiveEvents size", transitiveEvents.memoizedFlattenAndGetSize())
          .toString();
    }
  }

  private static final class NotComparableValueWithEvents extends ValueWithEvents
      implements NotComparableSkyValue {
    private NotComparableValueWithEvents(SkyValue value, NestedSet<Reportable> transitiveEvents) {
      super(value, transitiveEvents);
    }
  }

  /**
   * Implementation of {@link ValueWithMetadata} for the error case.
   *
   * <p>Mark NotComparableSkyValue because it's unlikely that re-evaluation gives the same error.
   */
  private static final class ErrorInfoValue extends ValueWithMetadata
      implements NotComparableSkyValue {

    private final ErrorInfo errorInfo;
    private final NestedSet<Reportable> transitiveEvents;

    ErrorInfoValue(
        ErrorInfo errorInfo, @Nullable SkyValue value, NestedSet<Reportable> transitiveEvents) {
      super(value);
      this.errorInfo = Preconditions.checkNotNull(errorInfo);
      this.transitiveEvents = Preconditions.checkNotNull(transitiveEvents);
    }

    @Override
    public boolean hasError() {
      return true;
    }

    @Nullable
    @Override
    ErrorInfo getErrorInfo() {
      return errorInfo;
    }

    @Override
    public NestedSet<Reportable> getTransitiveEvents() {
      return transitiveEvents;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (o == null || getClass() != o.getClass()) {
        return false;
      }

      ErrorInfoValue that = (ErrorInfoValue) o;

      // Shallow equals is a middle ground between using default equals, which might miss
      // nested sets with the same elements, and deep equality checking, which would be expensive.
      // All three choices are sound, since shallow equals and default equals are more
      // conservative than deep equals. Using shallow equals means that we may unnecessarily
      // consider some values unequal that are actually equal, but this is still a net win over
      // deep equals.
      return Objects.equals(this.value, that.value)
          && Objects.equals(this.errorInfo, that.errorInfo)
          && transitiveEvents.shallowEquals(that.transitiveEvents);
    }

    @Override
    public int hashCode() {
      return 31 * Objects.hash(value, errorInfo) + transitiveEvents.shallowHashCode();
    }

    @Override
    public String toString() {
      StringBuilder result = new StringBuilder();
      if (value != null) {
        result.append("Value: ").append(value);
      }
      if (errorInfo != null) {
        if (result.length() > 0) {
          result.append("; ");
        }
        result.append("Error: ").append(errorInfo);
      }
      return result.toString();
    }
  }

  @Nullable
  public static SkyValue justValue(SkyValue value) {
    if (value instanceof ValueWithMetadata) {
      return ((ValueWithMetadata) value).value;
    }
    return value;
  }

  public static ValueWithMetadata wrapWithMetadata(SkyValue value) {
    if (value instanceof ValueWithMetadata) {
      return (ValueWithMetadata) value;
    }
    return ValueWithEvents.createValueWithEvents(value, NO_EVENTS);
  }

  @Nullable
  public static ErrorInfo getMaybeErrorInfo(SkyValue value) {
    if (value instanceof ErrorInfoValue) {
      return ((ValueWithMetadata) value).getErrorInfo();
    }
    return null;
  }

  public static NestedSet<Reportable> getEvents(SkyValue value) {
    if (value instanceof ValueWithMetadata) {
      return ((ValueWithMetadata) value).getTransitiveEvents();
    }
    return NO_EVENTS;
  }
}
