// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.skydoc.fakebuildapi;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Ordering;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkbuildapi.core.StructApi;
import com.google.devtools.build.lib.syntax.ClassObject;
import com.google.devtools.build.lib.syntax.Dict;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Printer;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Fake implementation of {@link StructApi}.
 */
public class FakeStructApi implements StructApi, ClassObject {

  private final Map<String, Object> objects;

  public FakeStructApi(Map<String, Object> objects) {
    this.objects = objects;
  }

  public FakeStructApi() {
    this(ImmutableMap.of());
  }

  @Override
  public String toProto(Location loc) throws EvalException {
    return "";
  }

  @Override
  public String toJson(Location loc) throws EvalException {
    return "";
  }

  /**
   * Wraps {@link ClassObject#getValue(String)}, returning null in cases where {@link EvalException}
   * would have been thrown.
   */
  @VisibleForTesting
  public Object getValueOrNull(String name) {
    try {
      return getValue(name);
    } catch (EvalException e) {
      throw new IllegalStateException("getValue should not throw an exception", e);
    }
  }

  /** Converts the object to string using Starlark syntax. */
  @Override
  public void repr(Printer printer) {
    boolean first = true;
    printer.append("struct(");
    for (String fieldName : Ordering.natural().sortedCopy(getFieldNames())) {
      if (!first) {
        printer.append(", ");
      }
      first = false;
      printer.append(fieldName);
      printer.append(" = ");
      printer.repr(getValueOrNull(fieldName));
    }

    printer.append(")");
  }

  @Nullable
  @Override
  public Object getValue(String name) throws EvalException {
    return objects.get(name);
  }

  @Override
  public ImmutableCollection<String> getFieldNames() {
    return ImmutableList.copyOf(objects.keySet());
  }

  @Nullable
  @Override
  public String getErrorMessageForUnknownField(String field) {
    return "";
  }

  /**
   * Fake implementation of {@link StructProviderApi}.
   */
  public static class FakeStructProviderApi implements StructProviderApi {

    @Override
    public StructApi createStruct(Dict<?, ?> kwargs, Location loc) throws EvalException {
      return new FakeStructApi(kwargs.getContents(String.class, Object.class, "kwargs"));
    }

    @Override
    public void repr(Printer printer) {}
  }
}
