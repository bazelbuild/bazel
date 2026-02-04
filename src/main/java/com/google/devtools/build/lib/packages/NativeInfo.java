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

import static com.google.common.base.MoreObjects.firstNonNull;

import com.google.common.base.Objects;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Ordering;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import java.util.List;
import javax.annotation.Nullable;
import net.starlark.java.eval.BuiltinFunction;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Mutability;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.syntax.Location;

/**
 * Abstract base class for implementations of {@link Info} that expose StarlarkCallable-annotated
 * fields (not just methods) to Starlark code. Subclasses must be immutable.
 */
// TODO(adonovan): ensure that all subclasses are named *Info and not *Provider.
// (Info is to object as Provider is to class.)
@Immutable
public abstract class NativeInfo implements Info {
  private final Location location;

  protected NativeInfo() {
    this(Location.BUILTIN);
  }

  // TODO(adonovan): most subclasses pass Location.BUILTIN most of the time.
  // Make only those classes that pass a real location pay for it.
  protected NativeInfo(@Nullable Location location) {
    this.location = firstNonNull(location, Location.BUILTIN);
  }

  @Override
  public final Location getCreationLocation() {
    return location;
  }

  @Override
  public boolean isImmutable() {
    return true; // immutable and Starlark-hashable
  }

  // TODO(b/408391489) repr, hash, equals for native providers are inefficient; implement them
  //  directly and remove getLegacyStarlarkMethodNames getLegacyFields
  private List<String> getLegacyStarlarkMethodNames() {
    return Ordering.natural()
        .sortedCopy(Starlark.dir(Mutability.IMMUTABLE, StarlarkSemantics.DEFAULT, this));
  }

  private ImmutableMap<String, Object> getLegacyFields() {
    ImmutableMap.Builder<String, Object> fields = ImmutableMap.builder();
    for (String fieldName : getLegacyStarlarkMethodNames()) {
      try {
        Object value =
            Starlark.getattr(
                Mutability.IMMUTABLE, StarlarkSemantics.DEFAULT, this, fieldName, null);
        if (value instanceof BuiltinFunction) {
          continue;
        }
        fields.put(fieldName, value);
      } catch (EvalException e) {
        fields.put(fieldName, Starlark.NONE);
      } catch (InterruptedException e) {
        // Struct fields on NativeInfo objects are supposed to behave well and not throw
        // exceptions, as they should be logicless field accessors. If this occurs, it's
        // indicative of a bad NativeInfo implementation.
        throw new IllegalStateException(
            String.format(
                "Access of field %s was unexpectedly interrupted, but should be "
                    + "uninterruptible. This is indicative of a bad provider implementation.",
                fieldName),
            e);
      }
    }
    return fields.buildOrThrow();
  }

  @Override
  public boolean equals(Object otherObject) {
    if (!(otherObject instanceof NativeInfo other)) {
      return false;
    }
    if (this == other) {
      return true;
    }
    if (!this.getProvider().equals(other.getProvider())) {
      return false;
    }
    // Compare objects' fields and their values
    if (!Objects.equal(getLegacyStarlarkMethodNames(), other.getLegacyStarlarkMethodNames())) {
      return false;
    }
    return Objects.equal(getLegacyFields(), other.getLegacyFields());
  }

  @Override
  public int hashCode() {
    return Objects.hashCode(getProvider(), getLegacyFields());
  }

  /**
   * Convert the object to string using Starlark syntax. The output tries to be reversible (but
   * there is no guarantee, it depends on the actual values).
   */
  @Override
  public void repr(Printer printer, StarlarkSemantics semantics) {
    boolean first = true;
    printer.append("struct(");
    for (var field : getLegacyFields().entrySet()) {
      if (!first) {
        printer.append(", ");
      }
      first = false;
      printer.append(field.getKey());
      printer.append(" = ");
      printer.repr(field.getValue(), semantics);
    }
    printer.append(")");
  }

  @Override
  public String toString() {
    return Starlark.repr(this, StarlarkSemantics.DEFAULT);
  }
}
