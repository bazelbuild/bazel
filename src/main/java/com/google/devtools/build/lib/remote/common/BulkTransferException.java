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

import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ImportantOutputHandler.LostArtifacts;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
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
        if (t instanceof IOException ioException) {
          add(ioException);
        } else {
          throw new IllegalStateException("BulkTransferException contains non-IOException", t);
        }
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
   * Returns a {@link LostArtifacts} instance that is non-empty if and only if all suppressed
   * exceptions are caused by cache misses.
   */
  public LostArtifacts getLostArtifacts(Function<PathFragment, ActionInput> actionInputResolver) {
    if (!allCausedByCacheNotFoundException(this)) {
      return LostArtifacts.EMPTY;
    }

    var byDigestBuilder = ImmutableMap.<String, ActionInput>builder();
    for (var suppressed : getSuppressed()) {
      CacheNotFoundException e = (CacheNotFoundException) suppressed;
      var missingDigest = e.getMissingDigest();
      var execPath = e.getExecPath();
      if (execPath == null) {
        // This can happen if the lost artifact is not an input of the action, but a special output
        // such as stdout/stderr. This can't be solved by the rewinding that LostArtifacts would
        // trigger, but is rather a failure of the current action execution.
        return LostArtifacts.EMPTY;
      }
      var actionInput = actionInputResolver.apply(execPath);
      if (actionInput == null) {
        // This can happen if the lost artifact is not an input of the action, but an output that
        // e.g. failed to be retrieved from the remote cache after a cache hit. This also can't be
        // solved by the rewinding that LostArtifacts would trigger.
        return LostArtifacts.EMPTY;
      }
      byDigestBuilder.put(DigestUtil.toString(missingDigest), actionInput);
    }
    var byDigest = byDigestBuilder.buildKeepingLast();
    return new LostArtifacts(byDigest, Optional.empty());
  }

  @Override
  public String getMessage() {
    // Only report unique messages to avoid flooding the user, e.g. in case a remote cache server is
    // unavailable
    // and causing several identical messages. Also sort the messages, for more deterministic
    // result. All of this allows
    // more efficient event deduplication when reporting the returned aggregated message.
    List<String> uniqueSortedMessages =
        Arrays.stream(super.getSuppressed())
            .map(Throwable::getMessage)
            .filter(Objects::nonNull)
            .sorted()
            .distinct()
            .collect(toImmutableList());

    return switch (uniqueSortedMessages.size()) {
      case 0 -> "Unknown error during bulk transfer";
      case 1 -> Iterables.getOnlyElement(uniqueSortedMessages);
      default ->
          "Multiple errors during bulk transfer:\n" + Joiner.on("\n").join(uniqueSortedMessages);
    };
  }
}
