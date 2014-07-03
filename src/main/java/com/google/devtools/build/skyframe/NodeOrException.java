// Copyright 2014 Google Inc. All rights reserved.
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

import javax.annotation.Nullable;

/**
 * Wrapper for a node or the exception thrown when trying to build it.
 *
 * <p>This is intended only for use in alternative {@code AutoUpdatingGraph} implementations.
 *
 * @param <E> Exception class that may have been thrown when building requested node.
 */
public final class NodeOrException<E extends Throwable> {
  @Nullable private final Node node;
  @Nullable private final E exception;

  /** Gets the stored node. Throws an exception if one was thrown when building this node. */
  @Nullable public Node get() throws E {
    if (exception != null) {
      throw exception;
    }
    return node;
  }

  private NodeOrException(@Nullable Node node) {
    this.node = node;
    this.exception = null;
  }

  private NodeOrException(E exception) {
    this.exception = Preconditions.checkNotNull(exception);
    this.node = null;
  }

  /**
   * Returns a {@code NodeOrException} representing success.
   *
   * <p>This is intended only for use in alternative {@code AutoUpdatingGraph} implementations.
   */
  public static <F extends Throwable> NodeOrException<F> ofNode(Node node) {
    return new NodeOrException<>(node);
  }

  /**
   * Returns a {@code NodeOrException} representing failure.
   *
   * <p>This is intended only for use in alternative {@code AutoUpdatingGraph} implementations.
   */
  public static <F extends Throwable> NodeOrException<F> ofException(F exception) {
    return new NodeOrException<>(exception);
  }

  @SuppressWarnings("unchecked") // Cast to NodeOrException<F>.
  static <F extends Throwable> NodeOrException<F> ofNull() {
    return (NodeOrException<F>) NULL;
  }

  private static final NodeOrException<Throwable> NULL =
      new NodeOrException<>((Node) null);
}
