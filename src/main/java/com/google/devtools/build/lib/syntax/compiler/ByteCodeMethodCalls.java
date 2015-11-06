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
package com.google.devtools.build.lib.syntax.compiler;

import net.bytebuddy.implementation.bytecode.StackManipulation;

/**
 * Keeps often used {@link StackManipulation}s which call often needed methods from the standard
 * library and others.
 *
 * <p>Kept in a central place to reduce possible human errors in getting reflection right and
 * improve code reuse. Inner classes are prefixed with "BC" to avoid import errors and to allow
 * cleaner code in this class.
 */
public class ByteCodeMethodCalls {

  /**
   * Byte code invocations for {@link Integer}.
   */
  public static class BCInteger {
    public static final StackManipulation valueOf =
        ByteCodeUtils.invoke(Integer.class, "valueOf", int.class);
  }
}
