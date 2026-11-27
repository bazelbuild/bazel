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
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkFloat;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.StarlarkSet;
import net.starlark.java.eval.Tuple;

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

  public static AttributeValues create(Dict<String, Object> attribs) {
    return new AutoValue_AttributeValues(attribs);
  }

  public abstract Dict<String, Object> attributes();

  // Starlark equality ignores the order of dict entries and set elements and treats an int as equal
  // to a float with the same value, but module extensions and repo rules can observe both.
  @Override
  public final boolean equals(Object o) {
    return o instanceof AttributeValues that && strictlyEqual(attributes(), that.attributes());
  }

  @Override
  public final int hashCode() {
    return attributes().hashCode();
  }

  private static boolean strictlyEqual(Object x, Object y) {
    if (x instanceof Dict<?, ?> xDict && y instanceof Dict<?, ?> yDict) {
      return elementsStrictlyEqual(xDict.keySet(), yDict.keySet())
          && elementsStrictlyEqual(xDict.values(), yDict.values());
    }
    if (x instanceof StarlarkList<?> xList && y instanceof StarlarkList<?> yList) {
      return elementsStrictlyEqual(xList, yList);
    }
    if (x instanceof Tuple xTuple && y instanceof Tuple yTuple) {
      return elementsStrictlyEqual(xTuple, yTuple);
    }
    if (x instanceof StarlarkSet<?> xSet && y instanceof StarlarkSet<?> ySet) {
      return elementsStrictlyEqual(xSet, ySet);
    }
    return (x instanceof StarlarkFloat) == (y instanceof StarlarkFloat) && x.equals(y);
  }

  private static boolean elementsStrictlyEqual(Collection<?> xs, Collection<?> ys) {
    if (xs.size() != ys.size()) {
      return false;
    }
    Iterator<?> yIterator = ys.iterator();
    for (Object x : xs) {
      if (!strictlyEqual(x, yIterator.next())) {
        return false;
      }
    }
    return true;
  }
}
