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

import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkbuildapi.StructApi;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrinter;
import com.google.devtools.build.lib.syntax.ClassObject;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.SkylarkDict;
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

  // TODO(cparsons): Implement repr to match the real Struct's repr, as it affects the
  // "default value" documentation of functions.
  @Override
  public void repr(SkylarkPrinter printer) {}

  @Nullable
  @Override
  public Object getValue(String name) throws EvalException {
    return objects.get(name);
  }

  @Override
  public ImmutableCollection<String> getFieldNames() throws EvalException {
    return ImmutableList.of();
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
    public StructApi createStruct(SkylarkDict<?, ?> kwargs, Location loc) throws EvalException {
      return new FakeStructApi(kwargs.getContents(String.class, Object.class, "kwargs"));
    }

    @Override
    public void repr(SkylarkPrinter printer) {}
  }
}

