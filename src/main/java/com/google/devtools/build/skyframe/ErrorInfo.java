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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.skyframe.SkyFunctionException.ReifiedSkyFunctionException;

import java.io.IOException;
import java.io.Serializable;
import java.util.Collection;

import javax.annotation.Nullable;

/**
 * Information about why a {@link SkyValue} failed to evaluate successfully.
 *
 * <p>This is intended only for use in alternative {@code MemoizingEvaluator} implementations.
 */
public class ErrorInfo implements Serializable {

  /** Create an ErrorInfo from a {@link ReifiedSkyFunctionException}. */
  public static ErrorInfo fromException(ReifiedSkyFunctionException skyFunctionException) {
    SkyKey rootCauseSkyKey = skyFunctionException.getRootCauseSkyKey();
    Exception rootCauseException = skyFunctionException.getCause();
    return new ErrorInfo(
        NestedSetBuilder.create(Order.STABLE_ORDER, rootCauseSkyKey),
        Preconditions.checkNotNull(rootCauseException, "Cause null %s", rootCauseException),
        rootCauseSkyKey,
        /*cycles=*/ ImmutableList.<CycleInfo>of(),
        skyFunctionException.isTransient(),
        skyFunctionException.isCatastrophic());
  }

  /** Create an ErrorInfo from a {@link CycleInfo}. */
  static ErrorInfo fromCycle(CycleInfo cycleInfo) {
    return new ErrorInfo(
        /*rootCauses=*/ NestedSetBuilder.<SkyKey>emptySet(Order.STABLE_ORDER),
        /*exception=*/ null,
        /*rootCauseOfException=*/ null,
        ImmutableList.of(cycleInfo),
        /*isTransient=*/ false,
        /*isCatostrophic=*/ false);
  }

  /** Create an ErrorInfo from a collection of existing errors. */
  public static ErrorInfo fromChildErrors(SkyKey currentValue, Collection<ErrorInfo> childErrors) {
    Preconditions.checkNotNull(currentValue, "currentValue must not be null");
    Preconditions.checkState(!childErrors.isEmpty(), "childErrors may not be empty");

    NestedSetBuilder<SkyKey> rootCausesBuilder = NestedSetBuilder.stableOrder();
    ImmutableList.Builder<CycleInfo> cycleBuilder = ImmutableList.builder();
    Exception firstException = null;
    SkyKey firstChildKey = null;
    boolean isCatastrophic = false;
    // Arbitrarily pick the first error.
    for (ErrorInfo child : childErrors) {
      if (firstException == null) {
        firstException = child.getException();
        firstChildKey = child.getRootCauseOfException();
      }
      rootCausesBuilder.addTransitive(child.rootCauses);
      cycleBuilder.addAll(CycleInfo.prepareCycles(currentValue, child.cycles));
      isCatastrophic |= child.isCatastrophic();
    }

    return new ErrorInfo(
        rootCausesBuilder.build(),
        firstException,
        firstChildKey,
        cycleBuilder.build(),
        // Parent errors should not be transient -- we depend on the child's transience, if any, to
        // force re-evaluation if necessary.
        /*isTransient=*/ false,
        isCatastrophic);
  }

  // Non-final only to allow for serialization.
  private transient NestedSet<SkyKey> rootCauses;

  @Nullable private final Exception exception;
  private final SkyKey rootCauseOfException;

  private final ImmutableList<CycleInfo> cycles;

  private final boolean isTransient;
  private final boolean isCatastrophic;

  public ErrorInfo(NestedSet<SkyKey> rootCauses, @Nullable Exception exception,
      SkyKey rootCauseOfException, ImmutableList<CycleInfo> cycles, boolean isTransient,
      boolean isCatostrophic) {
    Preconditions.checkState(exception != null || !Iterables.isEmpty(cycles),
        "At least one of exception and cycles must be non-null/empty, respectively");
    Preconditions.checkState((exception == null) == (rootCauseOfException == null),
        "exception and rootCauseOfException must both be null or non-null, got %s  %s",
        exception, rootCauseOfException);

    this.rootCauses = rootCauses;
    this.exception = exception;
    this.rootCauseOfException = rootCauseOfException;
    this.cycles = cycles;
    this.isTransient = isTransient;
    this.isCatastrophic = isCatostrophic;
  }

  @Override
  public String toString() {
    return String.format("<ErrorInfo exception=%s rootCauses=%s cycles=%s>",
        exception, rootCauses, cycles);
  }

  /**
   * The root causes of a value that failed to build are its descendant values that failed to build.
   * If a value's descendants all built successfully, but it failed to, its root cause will be
   * itself. If a value depends on a cycle, but has no other errors, this method will return
   * the empty set.
   */
  public Iterable<SkyKey> getRootCauses() {
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
  public Iterable<CycleInfo> getCycleInfo() {
    return cycles;
  }

  /**
   * Returns true iff the error is transient, i.e. if retrying the same computation could lead to a
   * different result.
   */
  public boolean isTransient() {
    return isTransient;
  }


  /**
   * Returns true iff the error is catastrophic, i.e. it should halt even for a keepGoing update()
   * call.
   */
  public boolean isCatastrophic() {
    return isCatastrophic;
  }

  private void readObject(java.io.ObjectInputStream in) throws IOException, ClassNotFoundException {
    in.defaultReadObject();
    rootCauses = NestedSetBuilder.emptySet(Order.STABLE_ORDER);
  }
}
