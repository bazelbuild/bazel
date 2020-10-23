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
package com.google.devtools.build.skyframe;

import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.skyframe.SkyFunctionException.ReifiedSkyFunctionException;
import java.util.Collection;
import java.util.Objects;
import javax.annotation.Nullable;

/**
 * Information about why a {@link SkyValue} failed to evaluate successfully.
 *
 * <p>This is intended only for use in alternative {@code MemoizingEvaluator} implementations.
 */
public class ErrorInfo {

  /** Create an ErrorInfo from a {@link ReifiedSkyFunctionException}. */
  public static ErrorInfo fromException(ReifiedSkyFunctionException skyFunctionException,
      boolean isTransitivelyTransient) {
    SkyKey rootCauseSkyKey = skyFunctionException.getRootCauseSkyKey();
    Exception rootCauseException = skyFunctionException.getCause();
    return new ErrorInfo(
        NestedSetBuilder.create(Order.STABLE_ORDER, rootCauseSkyKey),
        Preconditions.checkNotNull(rootCauseException, "Cause is null"),
        rootCauseSkyKey,
        /*cycles=*/ ImmutableList.<CycleInfo>of(),
        skyFunctionException.isTransient(),
        isTransitivelyTransient || skyFunctionException.isTransient(),
        skyFunctionException.isCatastrophic());
  }

  /** Create an ErrorInfo from a {@link CycleInfo}. */
  public static ErrorInfo fromCycle(CycleInfo cycleInfo) {
    return new ErrorInfo(
        /*rootCauses=*/ NestedSetBuilder.<SkyKey>emptySet(Order.STABLE_ORDER),
        /*exception=*/ null,
        /*rootCauseOfException=*/ null,
        ImmutableList.of(cycleInfo),
        /*isDirectlyTransient=*/ false,
        /*isTransitivelyTransient=*/ false,
        /* isCatastrophic= */ false);
  }

  /** Create an ErrorInfo from a collection of existing errors. */
  public static ErrorInfo fromChildErrors(SkyKey currentValue, Collection<ErrorInfo> childErrors) {
    Preconditions.checkNotNull(currentValue, "currentValue must not be null");
    Preconditions.checkState(
        !childErrors.isEmpty(), "childErrors may not be empty %s", currentValue);

    NestedSetBuilder<SkyKey> rootCausesBuilder = NestedSetBuilder.stableOrder();
    ImmutableList.Builder<CycleInfo> cycleBuilder = ImmutableList.builder();
    Exception firstException = null;
    SkyKey firstChildKey = null;
    boolean isTransitivelyTransient = false;
    boolean isCatastrophic = false;
    for (ErrorInfo child : childErrors) {
      if (firstException == null) {
        // Arbitrarily pick the first error.
        firstException = child.getException();
        firstChildKey = child.getRootCauseOfException();
      }
      rootCausesBuilder.addTransitive(child.rootCauses);
      cycleBuilder.addAll(CycleInfo.prepareCycles(currentValue, child.cycles));
      isTransitivelyTransient |= child.isTransitivelyTransient();
      isCatastrophic |= child.isCatastrophic();
    }

    return new ErrorInfo(
        rootCausesBuilder.build(),
        firstException,
        firstChildKey,
        cycleBuilder.build(),
        /*isDirectlyTransient=*/ false,
        isTransitivelyTransient,
        isCatastrophic);
  }

  private final NestedSet<SkyKey> rootCauses;

  @Nullable private final Exception exception;
  private final SkyKey rootCauseOfException;

  private final ImmutableList<CycleInfo> cycles;

  private final boolean isDirectlyTransient;
  private final boolean isTransitivelyTransient;
  private final boolean isCatastrophic;

  public ErrorInfo(
      NestedSet<SkyKey> rootCauses,
      @Nullable Exception exception,
      SkyKey rootCauseOfException,
      ImmutableList<CycleInfo> cycles,
      boolean isDirectlyTransient,
      boolean isTransitivelyTransient,
      boolean isCatastrophic) {
    Preconditions.checkState(exception != null || !Iterables.isEmpty(cycles),
        "At least one of exception and cycles must be non-null/empty, respectively");
    Preconditions.checkState((exception == null) == (rootCauseOfException == null),
        "exception and rootCauseOfException must both be null or non-null, got %s  %s",
        exception, rootCauseOfException);

    this.rootCauses = rootCauses;
    this.exception = exception;
    this.rootCauseOfException = rootCauseOfException;
    this.cycles = cycles;
    this.isDirectlyTransient = isDirectlyTransient;
    this.isTransitivelyTransient = isTransitivelyTransient;
    this.isCatastrophic = isCatastrophic;
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof ErrorInfo)) {
      return false;
    }

    ErrorInfo other = (ErrorInfo) obj;
    if (rootCauses != other.rootCauses) {
      if (rootCauses == null || other.rootCauses == null) {
        return false;
      }
      if (!rootCauses.shallowEquals(other.rootCauses)) {
        return false;
      }
    }

    if (!Objects.equals(cycles, other.cycles)) {
      return false;
    }

    // Don't check the specific exception as most exceptions don't implement equality but at least
    // check their types and messages are the same.
    if (exception != other.exception) {
      if (exception == null || other.exception == null) {
        return false;
      }
      // Class objects are singletons with a single class loader.
      if (exception.getClass() != other.exception.getClass()) {
        return false;
      }
      if (!Objects.equals(exception.getMessage(), other.exception.getMessage())) {
        return false;
      }
    }

    if (!Objects.equals(rootCauseOfException, other.rootCauseOfException)) {
      return false;
    }

    return isDirectlyTransient == other.isDirectlyTransient
        && isTransitivelyTransient == other.isTransitivelyTransient
        && isCatastrophic == other.isCatastrophic;
  }

  @Override
  public int hashCode() {
    return Objects.hash(
        exception == null ? null : exception.getClass(),
        exception == null ? "" : exception.getMessage(),
        rootCauseOfException,
        cycles,
        isDirectlyTransient,
        isTransitivelyTransient,
        isCatastrophic,
        rootCauses == null ? 0 : rootCauses.shallowHashCode());
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("exception", exception)
        .add("rootCauses", rootCauses)
        .add("cycles", cycles)
        .add("isCatastrophic", isCatastrophic)
        .add("rootCauseOfException", rootCauseOfException)
        .add("isDirectlyTransient", isDirectlyTransient)
        .add("isTransitivelyTransient", isTransitivelyTransient)
        .toString();
  }

  /**
   * The root causes of a value that failed to build are its descendant values that failed to build.
   * If a value's descendants all built successfully, but it failed to, its root cause will be
   * itself. If a value depends on a cycle, but has no other errors, this method will return the
   * empty set.
   */
  public NestedSet<SkyKey> getRootCauses() {
    return rootCauses;
  }

  /**
   * The exception thrown when building a value. May be null if value's only error is depending
   * on a cycle.
   *
   * <p>The exception is used for reporting and thus may ultimately be rethrown by the caller.
   * As well, during a --nokeep_going evaluation, if an error value is encountered from an earlier
   * --keep_going build, the exception to be thrown is taken from here.
   */
  @Nullable public Exception getException() {
    return exception;
  }

  public SkyKey getRootCauseOfException() {
    return rootCauseOfException;
  }

  /**
   * Any cycles found when building this value.
   *
   * <p>If there are a large number of cycles, only a limited number are returned here.
   *
   * <p>If this value has a child through which there are multiple paths to the same cycle, only one
   * path is returned here. However, if there are multiple paths to the same cycle, each of which
   * goes through a different child, each of them is returned here.
   */
  public ImmutableList<CycleInfo> getCycleInfo() {
    return cycles;
  }

  /**
   * Returns true iff the error is directly transient, i.e. if there was a transient error
   * encountered during the computation itself.
   */
  public boolean isDirectlyTransient() {
    return isDirectlyTransient;
  }

  /**
   * Returns true iff the error is transitively transient, i.e. if retrying the same computation
   * could lead to a different result.
   */
  public boolean isTransitivelyTransient() {
    return isTransitivelyTransient;
  }

  /**
   * Returns true iff the error is catastrophic, i.e. it should halt even for a keepGoing update()
   * call.
   */
  public boolean isCatastrophic() {
    return isCatastrophic;
  }

}
