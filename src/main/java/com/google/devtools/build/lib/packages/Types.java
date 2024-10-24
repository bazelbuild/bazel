// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.packages;

import com.google.devtools.build.lib.packages.Type.DictType;
import com.google.devtools.build.lib.packages.Type.ListType;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstant;
import java.util.List;
import net.starlark.java.eval.StarlarkInt;

/**
 * Constants for {@link Type}s.
 *
 * <p>These constants are in a separate class from {@link Type} to break a class initialization
 * cycle, and prevent possible deadlocks.
 */
public final class Types {

  /** The type of a list of strings. */
  @SerializationConstant
  public static final ListType<String> STRING_LIST = ListType.create(Type.STRING);

  /** The type of a list of signed 32-bit Starlark integer values. */
  @SerializationConstant
  public static final ListType<StarlarkInt> INTEGER_LIST = ListType.create(Type.INTEGER);

  /** The type of a dictionary of {@linkplain Type#STRING strings}. */
  @SerializationConstant
  public static final DictType<String, String> STRING_DICT =
      DictType.create(Type.STRING, Type.STRING);

  /** The type of a dictionary of {@linkplain #STRING_LIST label lists}. */
  @SerializationConstant
  public static final DictType<String, List<String>> STRING_LIST_DICT =
      DictType.create(Type.STRING, STRING_LIST);

  private Types() {}
}
