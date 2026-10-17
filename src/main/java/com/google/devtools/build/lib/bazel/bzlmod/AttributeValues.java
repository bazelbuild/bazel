// Copyright 2023 The Bazel Authors. All rights reserved.
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
//

package com.google.devtools.build.lib.bazel.bzlmod;

import com.google.auto.value.AutoValue;
import com.google.common.collect.Interner;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import java.util.List;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;

/**
 * Wrapper around a dictionary of attribute names and values to facilitate de/serialization (see
 * {@link AttributeValuesAdapter}).
 *
 * <p>Note that all attribute values are stored as Starlark values (that is, {@link Starlark#valid}
 * -- so {@link net.starlark.java.eval.StarlarkInt} vs {@code int}, {@link
 * net.starlark.java.eval.StarlarkList} vs {@link List}, etc).
 */
@AutoValue
public abstract class AttributeValues {

  // Equal attribute values are common, e.g. between the result of evaluating a module extension and
  // the lockfile entry for it, which are both retained.
  private static final Interner<AttributeValues> interner = BlazeInterners.newWeakInterner();

  public static AttributeValues create(Dict<String, Object> attribs) {
    // Copies dicts that are still mutable, such as the kwargs of a tag, into a compact immutable
    // representation and returns immutable dicts as is.
    Dict<String, Object> immutableAttribs = Dict.immutableCopyOf(attribs);
    AttributeValues attributeValues = new AutoValue_AttributeValues(immutableAttribs);
    // Values that can still change, such as lists passed to a tag in a module file that is being
    // evaluated, must not be shared with other equal values.
    return immutableAttribs.values().stream().allMatch(AttributeValues::isDeeplyImmutable)
        ? interner.intern(attributeValues)
        : attributeValues;
  }

  private static boolean isDeeplyImmutable(Object value) {
    return switch (value) {
      case Dict<?, ?> dict ->
          dict.isImmutable() && dict.values().stream().allMatch(AttributeValues::isDeeplyImmutable);
      case Sequence<?> sequence ->
          sequence.isImmutable() && sequence.stream().allMatch(AttributeValues::isDeeplyImmutable);
      default -> Starlark.isImmutable(value);
    };
  }

  public abstract Dict<String, Object> attributes();
}
