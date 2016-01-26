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

package com.google.devtools.build.lib.util;

import javax.annotation.Nullable;

/**
 * Like {@link com.google.common.base.Preconditions}, but with overloads that avoid varargs-based
 * array creation in {@link #checkArgument}, {@link #checkState}, and {@link #checkNotNull} for
 * calls that list up to seven error message arguments when {@code expression} is {@code true} (or
 * {@code reference} is non-{@code null}).
 *
 * <p>Throughout this file, functions delegate to {@link com.google.common.base.Preconditions}
 * methods of the same name for the purpose of readability and obvious semantics.
 */
public final class Preconditions {
  private Preconditions() {}

  /** See {@link com.google.common.base.Preconditions#checkArgument(boolean)}. */
  public static void checkArgument(boolean expression) {
    com.google.common.base.Preconditions.checkArgument(expression);
  }

  /** See {@link com.google.common.base.Preconditions#checkArgument(boolean, Object)}. */
  public static void checkArgument(boolean expression, @Nullable Object errorMessage) {
    com.google.common.base.Preconditions.checkArgument(expression, errorMessage);
  }

  /**
   * See {@link com.google.common.base.Preconditions#checkArgument(boolean, String, Object...)}.
   */
  public static void checkArgument(
      boolean expression,
      @Nullable String errorMessageTemplate,
      @Nullable Object... errorMessageArgs) {
      com.google.common.base.Preconditions.checkArgument(
          expression, errorMessageTemplate, errorMessageArgs);
  }

  /**
   * Conditionally forwards to the {@link com.google.common.base.Preconditions} method of the same
   * name, avoiding varargs-based array creation in the unexceptional case.
   *
   * <p>See {@link com.google.common.base.Preconditions#checkArgument(boolean, String, Object...)}.
   */
  public static void checkArgument(
      boolean expression, @Nullable String errorMessageTemplate, @Nullable Object arg0) {
    if (!expression) {
      com.google.common.base.Preconditions.checkArgument(false, errorMessageTemplate, arg0);
    }
  }

  /**
   * Conditionally forwards to the {@link com.google.common.base.Preconditions} method of the same
   * name, avoiding autoboxing in the unexceptional case.
   *
   * <p>See {@link com.google.common.base.Preconditions#checkArgument(boolean, String, Object...)}.
   */
  public static void checkArgument(
      boolean expression, @Nullable String errorMessageTemplate, int arg0) {
    if (!expression) {
      com.google.common.base.Preconditions.checkArgument(false, errorMessageTemplate, arg0);
    }
  }

  /**
   * Conditionally forwards to the {@link com.google.common.base.Preconditions} method of the same
   * name, avoiding varargs-based array creation in the unexceptional case.
   *
   * <p>See {@link com.google.common.base.Preconditions#checkArgument(boolean, String, Object...)}.
   */
  public static void checkArgument(
      boolean expression,
      @Nullable String errorMessageTemplate,
      @Nullable Object arg0,
      @Nullable Object arg1) {
    if (!expression) {
      com.google.common.base.Preconditions.checkArgument(false, errorMessageTemplate, arg0, arg1);
    }
  }

  /**
   * Conditionally forwards to the {@link com.google.common.base.Preconditions} method of the same
   * name, avoiding varargs-based array creation in the unexceptional case.
   *
   * <p>See {@link com.google.common.base.Preconditions#checkArgument(boolean, String, Object...)}.
   */
  public static void checkArgument(
      boolean expression,
      @Nullable String errorMessageTemplate,
      @Nullable Object arg0,
      @Nullable Object arg1,
      @Nullable Object arg2) {
    if (!expression) {
      com.google.common.base.Preconditions.checkArgument(
          false, errorMessageTemplate, arg0, arg1, arg2);
    }
  }

  /**
   * Conditionally forwards to the {@link com.google.common.base.Preconditions} method of the same
   * name, avoiding varargs-based array creation in the unexceptional case.
   *
   * <p>See {@link com.google.common.base.Preconditions#checkArgument(boolean, String, Object...)}.
   */
  public static void checkArgument(
      boolean expression,
      @Nullable String errorMessageTemplate,
      @Nullable Object arg0,
      @Nullable Object arg1,
      @Nullable Object arg2,
      @Nullable Object arg3) {
    if (!expression) {
      com.google.common.base.Preconditions.checkArgument(
          false, errorMessageTemplate, arg0, arg1, arg2, arg3);
    }
  }

  /**
   * Conditionally forwards to the {@link com.google.common.base.Preconditions} method of the same
   * name, avoiding varargs-based array creation in the unexceptional case.
   *
   * <p>See {@link com.google.common.base.Preconditions#checkArgument(boolean, String, Object...)}.
   */
  public static void checkArgument(
      boolean expression,
      @Nullable String errorMessageTemplate,
      @Nullable Object arg0,
      @Nullable Object arg1,
      @Nullable Object arg2,
      @Nullable Object arg3,
      @Nullable Object arg4) {
    if (!expression) {
      com.google.common.base.Preconditions.checkArgument(
          false, errorMessageTemplate, arg0, arg1, arg2, arg3, arg4);
    }
  }

  /**
   * Conditionally forwards to the {@link com.google.common.base.Preconditions} method of the same
   * name, avoiding varargs-based array creation in the unexceptional case.
   *
   * <p>See {@link com.google.common.base.Preconditions#checkArgument(boolean, String, Object...)}.
   */
  public static void checkArgument(
      boolean expression,
      @Nullable String errorMessageTemplate,
      @Nullable Object arg0,
      @Nullable Object arg1,
      @Nullable Object arg2,
      @Nullable Object arg3,
      @Nullable Object arg4,
      @Nullable Object arg5) {
    if (!expression) {
      com.google.common.base.Preconditions.checkArgument(
          false, errorMessageTemplate, arg0, arg1, arg2, arg3, arg4, arg5);
    }
  }

  /**
   * Conditionally forwards to the {@link com.google.common.base.Preconditions} method of the same
   * name, avoiding varargs-based array creation in the unexceptional case.
   *
   * <p>See {@link com.google.common.base.Preconditions#checkArgument(boolean, String, Object...)}.
   */
  public static void checkArgument(
      boolean expression,
      @Nullable String errorMessageTemplate,
      @Nullable Object arg0,
      @Nullable Object arg1,
      @Nullable Object arg2,
      @Nullable Object arg3,
      @Nullable Object arg4,
      @Nullable Object arg5,
      @Nullable Object arg6) {
    if (!expression) {
      com.google.common.base.Preconditions.checkArgument(
          false, errorMessageTemplate, arg0, arg1, arg2, arg3, arg4, arg5, arg6);
    }
  }

  /** See {@link com.google.common.base.Preconditions#checkState(boolean)}. */
  public static void checkState(boolean expression) {
    com.google.common.base.Preconditions.checkState(expression);
  }

  /** See {@link com.google.common.base.Preconditions#checkState(boolean, Object)}. */
  public static void checkState(boolean expression, @Nullable Object errorMessage) {
    com.google.common.base.Preconditions.checkState(expression, errorMessage);
  }

  /** See {@link com.google.common.base.Preconditions#checkState(boolean, String, Object...)}. */
  public static void checkState(
      boolean expression,
      @Nullable String errorMessageTemplate,
      @Nullable Object... errorMessageArgs) {
    com.google.common.base.Preconditions.checkState(
        expression, errorMessageTemplate, errorMessageArgs);
  }

  /**
   * Conditionally forwards to the {@link com.google.common.base.Preconditions} method of the same
   * name, avoiding varargs-based array creation in the unexceptional case.
   *
   * <p>See {@link com.google.common.base.Preconditions#checkState(boolean, String, Object...)}.
   */
  public static void checkState(
      boolean expression, @Nullable String errorMessageTemplate, @Nullable Object arg0) {
    if (!expression) {
      com.google.common.base.Preconditions.checkState(false, errorMessageTemplate, arg0);
    }
  }

  /**
   * Conditionally forwards to the {@link com.google.common.base.Preconditions} method of the same
   * name, avoiding varargs-based array creation in the unexceptional case.
   *
   * <p>See {@link com.google.common.base.Preconditions#checkState(boolean, String, Object...)}.
   */
  public static void checkState(
      boolean expression,
      @Nullable String errorMessageTemplate,
      @Nullable Object arg0,
      @Nullable Object arg1) {
    if (!expression) {
      com.google.common.base.Preconditions.checkState(false, errorMessageTemplate, arg0, arg1);
    }
  }

  /**
   * Conditionally forwards to the {@link com.google.common.base.Preconditions} method of the same
   * name, avoiding varargs-based array creation in the unexceptional case.
   *
   * <p>See {@link com.google.common.base.Preconditions#checkState(boolean, String, Object...)}.
   */
  public static void checkState(
      boolean expression,
      @Nullable String errorMessageTemplate,
      @Nullable Object arg0,
      @Nullable Object arg1,
      @Nullable Object arg2) {
    if (!expression) {
      com.google.common.base.Preconditions.checkState(
          false, errorMessageTemplate, arg0, arg1, arg2);
    }
  }

  /**
   * Conditionally forwards to the {@link com.google.common.base.Preconditions} method of the same
   * name, avoiding varargs-based array creation in the unexceptional case.
   *
   * <p>See {@link com.google.common.base.Preconditions#checkState(boolean, String, Object...)}.
   */
  public static void checkState(
      boolean expression,
      @Nullable String errorMessageTemplate,
      @Nullable Object arg0,
      @Nullable Object arg1,
      @Nullable Object arg2,
      @Nullable Object arg3) {
    if (!expression) {
      com.google.common.base.Preconditions.checkState(
          false, errorMessageTemplate, arg0, arg1, arg2, arg3);
    }
  }

  /**
   * Conditionally forwards to the {@link com.google.common.base.Preconditions} method of the same
   * name, avoiding varargs-based array creation in the unexceptional case.
   *
   * <p>See {@link com.google.common.base.Preconditions#checkState(boolean, String, Object...)}.
   */
  public static void checkState(
      boolean expression,
      @Nullable String errorMessageTemplate,
      @Nullable Object arg0,
      @Nullable Object arg1,
      @Nullable Object arg2,
      @Nullable Object arg3,
      @Nullable Object arg4) {
    if (!expression) {
      com.google.common.base.Preconditions.checkState(
          false, errorMessageTemplate, arg0, arg1, arg2, arg3, arg4);
    }
  }

  /**
   * Conditionally forwards to the {@link com.google.common.base.Preconditions} method of the same
   * name, avoiding varargs-based array creation in the unexceptional case.
   *
   * <p>See {@link com.google.common.base.Preconditions#checkState(boolean, String, Object...)}.
   */
  public static void checkState(
      boolean expression,
      @Nullable String errorMessageTemplate,
      @Nullable Object arg0,
      @Nullable Object arg1,
      @Nullable Object arg2,
      @Nullable Object arg3,
      @Nullable Object arg4,
      @Nullable Object arg5) {
    if (!expression) {
      com.google.common.base.Preconditions.checkState(
          false, errorMessageTemplate, arg0, arg1, arg2, arg3, arg4, arg5);
    }
  }

  /**
   * Conditionally forwards to the {@link com.google.common.base.Preconditions} method of the same
   * name, avoiding varargs-based array creation in the unexceptional case.
   *
   * <p>See {@link com.google.common.base.Preconditions#checkState(boolean, String, Object...)}.
   */
  public static void checkState(
      boolean expression,
      @Nullable String errorMessageTemplate,
      @Nullable Object arg0,
      @Nullable Object arg1,
      @Nullable Object arg2,
      @Nullable Object arg3,
      @Nullable Object arg4,
      @Nullable Object arg5,
      @Nullable Object arg6) {
    if (!expression) {
      com.google.common.base.Preconditions.checkState(
          false, errorMessageTemplate, arg0, arg1, arg2, arg3, arg4, arg5, arg6);
    }
  }

  /** See {@link com.google.common.base.Preconditions#checkNotNull(Object)}. */
  public static <T> T checkNotNull(T reference) {
    return com.google.common.base.Preconditions.checkNotNull(reference);
  }

  /** See {@link com.google.common.base.Preconditions#checkNotNull(Object, Object)}. */
  public static <T> T checkNotNull(T reference, @Nullable Object errorMessage) {
    return com.google.common.base.Preconditions.checkNotNull(reference, errorMessage);
  }

  /** See {@link com.google.common.base.Preconditions#checkNotNull(Object, String, Object...)}. */
  public static <T> T checkNotNull(
      T reference, @Nullable String errorMessageTemplate, @Nullable Object... errorMessageArgs) {
    return com.google.common.base.Preconditions.checkNotNull(
        reference, errorMessageTemplate, errorMessageArgs);
  }

  /**
   * Conditionally forwards to the {@link com.google.common.base.Preconditions} method of the same
   * name, avoiding varargs-based array creation in the unexceptional case.
   *
   * <p>See {@link com.google.common.base.Preconditions#checkNotNull(Object, String, Object...)}.
   */
  public static <T> T checkNotNull(
      T reference, @Nullable String errorMessageTemplate, @Nullable Object arg0) {
    if (reference == null) {
      com.google.common.base.Preconditions.checkNotNull(null, errorMessageTemplate, arg0);
    }
    return reference;
  }

  /**
   * Conditionally forwards to the {@link com.google.common.base.Preconditions} method of the same
   * name, avoiding varargs-based array creation in the unexceptional case.
   *
   * <p>See {@link com.google.common.base.Preconditions#checkNotNull(Object, String, Object...)}.
   */
  public static <T> T checkNotNull(
      T reference,
      @Nullable String errorMessageTemplate,
      @Nullable Object arg0,
      @Nullable Object arg1) {
    if (reference == null) {
      com.google.common.base.Preconditions.checkNotNull(null, errorMessageTemplate, arg0, arg1);
    }
    return reference;
  }

  /**
   * Conditionally forwards to the {@link com.google.common.base.Preconditions} method of the same
   * name, avoiding varargs-based array creation in the unexceptional case.
   *
   * <p>See {@link com.google.common.base.Preconditions#checkNotNull(Object, String, Object...)}.
   */
  public static <T> T checkNotNull(
      T reference,
      @Nullable String errorMessageTemplate,
      @Nullable Object arg0,
      @Nullable Object arg1,
      @Nullable Object arg2) {
    if (reference == null) {
      com.google.common.base.Preconditions.checkNotNull(
          null, errorMessageTemplate, arg0, arg1, arg2);
    }
    return reference;
  }

  /**
   * Conditionally forwards to the {@link com.google.common.base.Preconditions} method of the same
   * name, avoiding varargs-based array creation in the unexceptional case.
   *
   * <p>See {@link com.google.common.base.Preconditions#checkNotNull(Object, String, Object...)}.
   */
  public static <T> T checkNotNull(
      T reference,
      @Nullable String errorMessageTemplate,
      @Nullable Object arg0,
      @Nullable Object arg1,
      @Nullable Object arg2,
      @Nullable Object arg3) {
    if (reference == null) {
      com.google.common.base.Preconditions.checkNotNull(
          null, errorMessageTemplate, arg0, arg1, arg2, arg3);
    }
    return reference;
  }

  /**
   * Conditionally forwards to the {@link com.google.common.base.Preconditions} method of the same
   * name, avoiding varargs-based array creation in the unexceptional case.
   *
   * <p>See {@link com.google.common.base.Preconditions#checkNotNull(Object, String, Object...)}.
   */
  public static <T> T checkNotNull(
      T reference,
      @Nullable String errorMessageTemplate,
      @Nullable Object arg0,
      @Nullable Object arg1,
      @Nullable Object arg2,
      @Nullable Object arg3,
      @Nullable Object arg4) {
    if (reference == null) {
      com.google.common.base.Preconditions.checkNotNull(
          null, errorMessageTemplate, arg0, arg1, arg2, arg3, arg4);
    }
    return reference;
  }

  /**
   * Conditionally forwards to the {@link com.google.common.base.Preconditions} method of the same
   * name, avoiding varargs-based array creation in the unexceptional case.
   *
   * <p>See {@link com.google.common.base.Preconditions#checkNotNull(Object, String, Object...)}.
   */
  public static <T> T checkNotNull(
      T reference,
      @Nullable String errorMessageTemplate,
      @Nullable Object arg0,
      @Nullable Object arg1,
      @Nullable Object arg2,
      @Nullable Object arg3,
      @Nullable Object arg4,
      @Nullable Object arg5) {
    if (reference == null) {
      com.google.common.base.Preconditions.checkNotNull(
          null, errorMessageTemplate, arg0, arg1, arg2, arg3, arg4, arg5);
    }
    return reference;
  }

  /**
   * Conditionally forwards to the {@link com.google.common.base.Preconditions} method of the same
   * name, avoiding varargs-based array creation in the unexceptional case.
   *
   * <p>See {@link com.google.common.base.Preconditions#checkNotNull(Object, String, Object...)}.
   */
  public static <T> T checkNotNull(
      T reference,
      @Nullable String errorMessageTemplate,
      @Nullable Object arg0,
      @Nullable Object arg1,
      @Nullable Object arg2,
      @Nullable Object arg3,
      @Nullable Object arg4,
      @Nullable Object arg5,
      @Nullable Object arg6) {
    if (reference == null) {
      com.google.common.base.Preconditions.checkNotNull(
          null, errorMessageTemplate, arg0, arg1, arg2, arg3, arg4, arg5, arg6);
    }
    return reference;
  }
}
