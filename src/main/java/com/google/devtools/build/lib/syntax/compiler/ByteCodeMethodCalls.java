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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.SkylarkDict;

import net.bytebuddy.implementation.bytecode.StackManipulation;

import java.util.Iterator;
import java.util.List;
import java.util.Map;

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
   * Byte code invocations for {@link Object}.
   */
  public static class BCObject {
    public static final StackManipulation equals =
        ByteCodeUtils.invoke(Object.class, "equals", Object.class);
  }

  /**
   * Byte code invocations for {@link Boolean}.
   */
  public static class BCBoolean {
    public static final StackManipulation valueOf =
        ByteCodeUtils.invoke(Boolean.class, "valueOf", boolean.class);
  }

  /**
   * Byte code invocations for {@link ImmutableMap}.
   */
  public static class BCImmutableMap {
    public static final StackManipulation builder =
        ByteCodeUtils.invoke(ImmutableMap.class, "builder");

    public static final StackManipulation copyOf =
        ByteCodeUtils.invoke(ImmutableMap.class, "copyOf", Map.class);

    /**
     * Byte code invocations for {@link Builder}.
     */
    public static class Builder {
      public static final StackManipulation put =
          ByteCodeUtils.invoke(ImmutableMap.Builder.class, "put", Object.class, Object.class);

      public static final StackManipulation build =
          ByteCodeUtils.invoke(ImmutableMap.Builder.class, "build");
    }
  }

  /**
   * Byte code invocations for {@link SkylarkDict}.
   */
  public static class BCSkylarkDict {
    public static final StackManipulation of =
        ByteCodeUtils.invoke(SkylarkDict.class, "of", Environment.class);

    public static final StackManipulation copyOf =
        ByteCodeUtils.invoke(SkylarkDict.class, "copyOf", Environment.class, Map.class);

    public static final StackManipulation put =
        ByteCodeUtils.invoke(SkylarkDict.class, "put",
            Object.class, Object.class, Location.class, Environment.class);
  }

  /**
   * Byte code invocations for {@link ImmutableList}.
   */
  public static class BCImmutableList {
    public static final StackManipulation builder =
        ByteCodeUtils.invoke(ImmutableList.class, "builder");

    public static final StackManipulation copyOf =
        ByteCodeUtils.invoke(ImmutableList.class, "copyOf", Iterable.class);

    public static final StackManipulation iterator =
        ByteCodeUtils.invoke(ImmutableList.class, "iterator");

    /**
    * Byte code invocations for {@link ImmutableList.Builder}.
    */
    public static class Builder {
      public static final StackManipulation build =
          ByteCodeUtils.invoke(ImmutableList.Builder.class, "build");

      public static final StackManipulation add =
          ByteCodeUtils.invoke(ImmutableList.Builder.class, "add", Object.class);

      public static final StackManipulation addAll =
          ByteCodeUtils.invoke(ImmutableList.Builder.class, "addAll", Iterable.class);
    }
  }

  /**
   * Byte code invocations for {@link Integer}.
   */
  public static class BCInteger {
    public static final StackManipulation valueOf =
        ByteCodeUtils.invoke(Integer.class, "valueOf", int.class);
  }

  /**
   * Byte code invocations for {@link Iterator}.
   */
  public static class BCIterator {

    public static final StackManipulation hasNext = ByteCodeUtils.invoke(Iterator.class, "hasNext");

    public static final StackManipulation next = ByteCodeUtils.invoke(Iterator.class, "next");
  }

  /**
   * Byte code invocations for {@link List}.
   */
  public static class BCList {
    public static final StackManipulation add =
        ByteCodeUtils.invoke(List.class, "add", Object.class);
  }
}
