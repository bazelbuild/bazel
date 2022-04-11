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

import com.google.common.base.Joiner;
import java.io.IOException;

/**
 * Exception which represents a collection of IOExceptions for the purpose of distinguishing remote
 * communication exceptions from those which occur on filesystems locally. This exception serves as
 * a trace point for the actual transfer, so that the intended operation can be observed in a stack,
 * with all constituent exceptions available for observation.
 */
public class BulkTransferException extends IOException {
  // true since no empty BulkTransferException is ever thrown
  private boolean allCacheNotFoundException = true;

  public BulkTransferException() {}

  public BulkTransferException(IOException e) {
    add(e);
  }

  /**
   * Add an IOException to the suppressed list.
   *
   * <p>The Java standard addSuppressed is final and this method stands in its place to selectively
   * filter and record whether all suppressed exceptions are CacheNotFoundExceptions
   */
  public void add(IOException e) {
    allCacheNotFoundException &= e instanceof CacheNotFoundException;
    super.addSuppressed(e);
  }

  public boolean onlyCausedByCacheNotFoundException() {
    return allCacheNotFoundException;
  }

  public static boolean isOnlyCausedByCacheNotFoundException(Exception e) {
    return e instanceof BulkTransferException
        && ((BulkTransferException) e).onlyCausedByCacheNotFoundException();
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
