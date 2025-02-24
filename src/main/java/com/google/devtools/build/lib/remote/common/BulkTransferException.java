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

import static com.google.common.base.Preconditions.checkState;
import static com.google.common.collect.ImmutableMap.toImmutableMap;

import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.function.Function;
import java.util.stream.Collectors;

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

  // Avoid the heavy dependency on InputMetadataProvider, whose getInput method provides the
  // argument to this method.
  public ImmutableMap<String, ActionInput> getLostInputs(
      Function<String, ActionInput> actionInputResolver) {
    if (!allCausedByCacheNotFoundException(this)) {
      return ImmutableMap.of();
    }
    return Arrays.stream(getSuppressed())
        .map(CacheNotFoundException.class::cast)
        .collect(
            toImmutableMap(
                e -> DigestUtil.toString(e.getMissingDigest()),
                e -> {
                  Preconditions.checkNotNull(
                      e.getExecPath(),
                      "exec path not known for action input with digest %s",
                      e.getMissingDigest());
                  ActionInput actionInput =
                      actionInputResolver.apply(e.getExecPath().getPathString());
                  Preconditions.checkNotNull(
                      actionInput,
                      "ActionInput not found for filename %s in CacheNotFoundException",
                      e.getExecPath());
                  return actionInput;
                },
                (a, b) -> b));
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
            .collect(Collectors.toList());

    switch (uniqueSortedMessages.size()) {
      case 0:
        return "Unknown error during bulk transfer";
      case 1:
        return uniqueSortedMessages.iterator().next();
      default:
        return "Multiple errors during bulk transfer:\n"
            + Joiner.on('\n').join(uniqueSortedMessages);
    }
  }
}
