// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote.common;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import java.io.IOException;
import java.util.function.Function;

/**
 * Exception which represents a collection of IOExceptions for the purpose of distinguishing remote
 * communication exceptions from those which occur on filesystems locally. This exception serves as
 * a trace point for the actual transfer, so that the intended operation can be observed in a stack,
 * with all constituent exceptions available for observation.
 */
public class BulkTransferException extends IOException {
  // No empty BulkTransferException is ever thrown.
  private boolean allCacheNotFoundException = true;

  public BulkTransferException() {}

  public BulkTransferException(IOException e) {
    add(e);
  }

  /**
   * Adds an IOException to the suppressed list.
   *
   * <p>If the IOException is already a BulkTransferException, the contained IOExceptions are added
   * instead.
   *
   * <p>The Java standard addSuppressed is final and this method stands in its place to selectively
   * filter and record whether all suppressed exceptions are CacheNotFoundExceptions.
   */
  public void add(IOException e) {
    if (e instanceof BulkTransferException bulkTransferException) {
      for (Throwable t : bulkTransferException.getSuppressed()) {
        checkState(t instanceof IOException);
        add(bulkTransferException);
      }
      return;
    }
    allCacheNotFoundException &= e instanceof CacheNotFoundException;
    super.addSuppressed(e);
  }

  public boolean allCausedByCacheNotFoundException() {
    return allCacheNotFoundException;
  }

  public static boolean allCausedByCacheNotFoundException(Throwable e) {
    return e instanceof BulkTransferException bulkTransferException
        && bulkTransferException.allCausedByCacheNotFoundException();
  }

  /**
   * Returns a map whose keys are the textual representation of a digest, and whose values are the
   * corresponding action inputs
   *
   * <p>Use {@code Function<String, ActionInput>} to avoid the heavy dependency on {@code
   * InputMetadataProvider}, whose getInput method provides the argument to this method.
   */
  public ImmutableMap<String, ActionInput> getLostInputs(
      Function<String, ActionInput> actionInputResolver) {
    if (!allCausedByCacheNotFoundException(this)) {
      return ImmutableMap.of();
    }

    ImmutableMap.Builder<String, ActionInput> lostInputs = ImmutableMap.builder();
    for (var suppressed : getSuppressed()) {
      CacheNotFoundException e = (CacheNotFoundException) suppressed;
      var missingDigest = e.getMissingDigest();
      var execPath = e.getExecPath();
      checkNotNull(execPath, "exec path not known for action input with digest %s", missingDigest);
      var actionInput = actionInputResolver.apply(execPath.getPathString());
      if (actionInput == null) {
        throw new IllegalStateException(
            "ActionInput not found for filename %s in CacheNotFoundException".formatted(execPath),
            this);
      }

      lostInputs.put(DigestUtil.toString(missingDigest), actionInput);
    }
    return lostInputs.buildKeepingLast();
  }

  @Override
  public String getMessage() {
    // If there is only one suppressed exception, displaying that in the message should be helpful.
    if (super.getSuppressed().length == 1) {
      return super.getSuppressed()[0].getMessage();
    }
    String errorSummary =
        String.format("%d errors during bulk transfer:", super.getSuppressed().length);
    String combinedSuberrors = Joiner.on('\n').join(super.getSuppressed());
    return Joiner.on('\n').join(errorSummary, combinedSuberrors);
  }
}
