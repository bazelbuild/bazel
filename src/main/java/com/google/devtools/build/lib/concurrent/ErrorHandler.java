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
package com.google.devtools.build.lib.concurrent;

import com.google.devtools.build.lib.concurrent.ErrorClassifier.ErrorClassification;

/** A way to inject custom handling of errors encountered by {@link AbstractQueueVisitor}. */
public interface ErrorHandler {

  /**
   * Called by {@link AbstractQueueVisitor} right after using {@link ErrorClassifier} to classify
   * the error, but right before actually acting on the classification.
   * 
   * <p>Note that {@link Error}s are always classified as
   * {@link ErrorClassification#CRITICAL_AND_LOG}.
   */
  void handle(Throwable t, ErrorClassification classification);

  /** An {@link ErrorHandler} that does nothing. */
  class NullHandler implements ErrorHandler {
    public static final NullHandler INSTANCE = new NullHandler();

    private NullHandler() {
    }

    @Override
    public void handle(Throwable t, ErrorClassification classification) {
    }
  }
}

