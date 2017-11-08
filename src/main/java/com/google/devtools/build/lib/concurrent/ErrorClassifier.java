// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.concurrent;

import com.google.common.base.Preconditions;

/** A classifier for {@link Error}s and {@link Exception}s. Used by {@link AbstractQueueVisitor}. */
public abstract class ErrorClassifier {

  /**
   * Classification of an error thrown by an action.
   *
   * <p>N.B. - These enum values are ordered from least severe to most severe.
   */
  public enum ErrorClassification {
    /** Other running actions should be left alone.*/
    NOT_CRITICAL,
    /** All running actions should be stopped.*/
    CRITICAL,
    /** Same as {@link #CRITICAL}, but also log the error.*/
    CRITICAL_AND_LOG,
    /** Same as {@link #CRITICAL_AND_LOG}, but is even worse. */
    AS_CRITICAL_AS_POSSIBLE
  }

  /** Always treat exceptions as {@code NOT_CRITICAL}. */
  public static final ErrorClassifier DEFAULT =
      new ErrorClassifier() {
        @Override
        protected ErrorClassification classifyException(Exception e) {
          return ErrorClassification.NOT_CRITICAL;
        }
      };

  /**
   * Used by {@link #classify} to classify {@link Exception}s. (Note that {@link Error}s
   * are always classified as {@code AS_CRITICAL_AS_POSSIBLE}.)
   *
   * @param e the exception object to check
   */
  protected abstract ErrorClassification classifyException(Exception e);

  /**
   * Classify {@code e}. If {@code e} is an {@link Error}, it will be classified as
   * {@code AS_CRITICAL_AS_POSSIBLE}. Otherwise, calls {@link #classifyException}.
   */
  public final ErrorClassification classify(Throwable e) {
    if (e instanceof Error) {
      return ErrorClassification.AS_CRITICAL_AS_POSSIBLE;
    }
    Preconditions.checkArgument(e instanceof Exception, e);
    return classifyException((Exception) e);
  }
}
