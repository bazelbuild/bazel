// Copyright 2017 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedMap;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.syntax.BuiltinCallable;
import com.google.devtools.build.lib.syntax.CallUtils;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import java.util.Map;

/** Base class for native implementations of {@link StructImpl}. */
// todo(vladmos,dslomov): make abstract once DefaultInfo stops instantiating it.
// TODO(adonovan): split NativeInfo into NativeInfo and NativeInfoWithExtraFields;
//  only a very few subclasses (e.g. ToolchainInfo) make use of NativeInfoWithFields.values,
//  and only they should pay for it.
public class NativeInfo extends StructImpl {
  protected final ImmutableSortedMap<String, Object> values;

  // Initialized lazily.
  private ImmutableSet<String> fieldNames;

  // TODO(adonovan): logically this should be a parameter of getValue
  // and getFieldNames or an instance field of this object.
  private static final StarlarkSemantics SEMANTICS = StarlarkSemantics.DEFAULT_SEMANTICS;

  @Override
  public Object getValue(String name) throws EvalException {
    // Starlark field
    Object x = values.get(name);
    if (x != null) {
      return x;
    }

    // @SkylarkCallable(structField=true) -- Java field
    if (getFieldNames().contains(name)) {
      try {
        return CallUtils.getField(SEMANTICS, this, name);
      } catch (InterruptedException exception) {
        // Struct fields on NativeInfo objects are supposed to behave well and not throw
        // exceptions, as they should be logicless field accessors. If this occurs, it's
        // indicative of a bad NativeInfo implementation.
        throw new IllegalStateException(
            String.format(
                "Access of field %s was unexpectedly interrupted, but should be "
                    + "uninterruptible. This is indicative of a bad provider implementation.",
                name));
      }
    }

    // to_json and to_proto should not be methods of struct or provider instances.
    // However, they are, for now, and it is important that they be consistently
    // returned by attribute lookup operations regardless of whether a field or method
    // is desired. TODO(adonovan): eliminate this hack.
    if (name.equals("to_json") || name.equals("to_proto")) {
      return new BuiltinCallable(this, name);
    }

    return null;
  }

  @Override
  public ImmutableCollection<String> getFieldNames() {
    if (fieldNames == null) {
      // TODO(adonovan): the assignment to this.fieldNames is not thread safe!
      // We cannot assume that build() is a constructor of an object all of
      // whose fields are final (and thus subject to a write barrier).
      //
      // Also, consider using a lazy union of the two underlying sets.
      fieldNames =
          ImmutableSet.<String>builder()
              .addAll(values.keySet())
              .addAll(CallUtils.getFieldNames(SEMANTICS, this))
              .build();
    }
    return fieldNames;
  }

  public NativeInfo(Provider provider) {
    this(provider, Location.BUILTIN);
  }

  public NativeInfo(Provider provider, Location loc) {
    this(provider, ImmutableMap.of(), loc);
  }

  // TODO(cparsons): Remove this constructor once ToolchainInfo stops using it.
  @Deprecated
  public NativeInfo(Provider provider, Map<String, Object> values, Location loc) {
    super(provider, loc);
    this.values = copyValues(values);
  }
}
