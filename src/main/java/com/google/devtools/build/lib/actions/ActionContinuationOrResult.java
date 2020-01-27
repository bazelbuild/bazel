// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.actions;

import com.google.common.util.concurrent.ListenableFuture;
import javax.annotation.Nullable;

/**
 * Represents either an action continuation or a final result, depending on the return value of the
 * {@link #isDone} method. Subclasses must implement {@link #getFuture} (but may return {@code null}
 * and {@link #execute}. Use {@link #of} to construct a final result.
 *
 * <p>Any clients <b>must</b> first call {@link #isDone} before calling any other method in this
 * class.
 */
public abstract class ActionContinuationOrResult {
  public static ActionContinuationOrResult of(ActionResult actionResult) {
    return new Finished(actionResult);
  }

  /** Returns true if this is a final result. */
  public boolean isDone() {
    return false;
  }

  /** Returns a future representing any ongoing concurrent work, or {@code null} otherwise. */
  @Nullable
  public abstract ListenableFuture<?> getFuture();

  /** Performs the next step in the process of executing the parent action. */
  public abstract ActionContinuationOrResult execute()
      throws ActionExecutionException, InterruptedException;

  /** Returns the final result. */
  public ActionResult get() {
    throw new IllegalStateException();
  }

  /** Represents a finished action result. */
  private static final class Finished extends ActionContinuationOrResult {
    private final ActionResult actionResult;

    private Finished(ActionResult actionResult) {
      this.actionResult = actionResult;
    }

    @Override
    public boolean isDone() {
      return true;
    }

    @Override
    public ListenableFuture<?> getFuture() {
      throw new IllegalStateException();
    }

    @Override
    public ActionContinuationOrResult execute() {
      throw new IllegalStateException();
    }

    @Override
    public ActionResult get() {
      return actionResult;
    }
  }
}
