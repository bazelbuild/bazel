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
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.syntax.Location;

/**
 * Abstract base class for implementations of {@link StructImpl} that expose
 * StarlarkCallable-annotated fields (not just methods) to Starlark code. Subclasses must be
 * immutable.
 */
// TODO(adonovan): ensure that all subclasses are named *Info and not *Provider.
// (Info is to object as Provider is to class.)
@Immutable
public abstract class NativeInfo extends StructImpl {

  protected NativeInfo() {
    this(Location.BUILTIN);
  }

  // TODO(adonovan): most subclasses pass Location.BUILTIN most of the time.
  // Make only those classes that pass a real location pay for it.
  protected NativeInfo(@Nullable Location loc) {
    super(loc);
  }

  @Override
  public boolean isImmutable() {
    return true; // immutable and Starlark-hashable
  }

  // TODO(adonovan): logically this should be a parameter of getValue
  // and getFieldNames or an instance field of this object.
  private static final StarlarkSemantics SEMANTICS = StarlarkSemantics.DEFAULT;

  @Override
  public Object getValue(String name) throws EvalException {
    // TODO(adonovan): this seems unnecessarily complicated:
    // Starlark's x.name and getattr(x, name) already check the
    // annotated fields/methods first, so there's no need to handle them here.
    // Similarly, Starlark.dir checks annotated fields/methods first, so
    // there's no need for getFieldNames to report them.
    // The only code that would notice any difference is direct Java
    // calls to getValue/getField names; they should instead
    // use getattr and dir. However, dir does report methods,
    // not just fields.

    // @StarlarkMethod(structField=true) -- Java field
    if (getFieldNames().contains(name)) {
      try {
        return Starlark.getAnnotatedField(SEMANTICS, this, name);
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
    return null;
  }

  @Override
  public ImmutableCollection<String> getFieldNames() {
    return Starlark.getAnnotatedFieldNames(SEMANTICS, this);
  }
}
