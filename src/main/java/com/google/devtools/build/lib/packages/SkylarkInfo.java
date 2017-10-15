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

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Sets;
import com.google.common.collect.Sets.SetView;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.NativeProvider.StructConstructor;
import com.google.devtools.build.lib.syntax.Concatable;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import java.util.Map;

/** Implementation of {@link Info} created from Skylark. */
public final class SkylarkInfo extends Info implements Concatable {
  protected final ImmutableMap<String, Object> values;

  public SkylarkInfo(Provider provider, Map<String, Object> kwargs, Location loc) {
    super(provider, loc);
    this.values = copyValues(kwargs);
  }

  public SkylarkInfo(StructConstructor provider, Map<String, Object> values, String message) {
    super(provider, values, message);
    this.values = copyValues(values);
  }

  @Override
  public Concatter getConcatter() {
    return StructConcatter.INSTANCE;
  }

  @Override
  public Object getValue(String name) {
    return values.get(name);
  }

  @Override
  public boolean hasKey(String name) {
    return values.containsKey(name);
  }

  @Override
  public ImmutableCollection<String> getKeys() {
    return values.keySet();
  }

  @Override
  public boolean isImmutable() {
    // If the provider is not yet exported the hash code of the object is subject to change
    if (!getProvider().isExported()) {
      return false;
    }
    for (Object item : values.values()) {
      if (!EvalUtils.isImmutable(item)) {
        return false;
      }
    }
    return true;
  }

  private static class StructConcatter implements Concatter {
    private static final StructConcatter INSTANCE = new StructConcatter();

    private StructConcatter() {}

    @Override
    public SkylarkInfo concat(Concatable left, Concatable right, Location loc)
        throws EvalException {
      SkylarkInfo lval = (SkylarkInfo) left;
      SkylarkInfo rval = (SkylarkInfo) right;
      if (!lval.getProvider().equals(rval.getProvider())) {
        throw new EvalException(
            loc,
            String.format(
                "Cannot concat %s with %s",
                lval.getProvider().getPrintableName(), rval.getProvider().getPrintableName()));
      }
      SetView<String> commonFields = Sets.intersection(lval.values.keySet(), rval.values.keySet());
      if (!commonFields.isEmpty()) {
        throw new EvalException(
            loc,
            "Cannot concat structs with common field(s): " + Joiner.on(",").join(commonFields));
      }
      return new SkylarkInfo(
          lval.getProvider(),
          ImmutableMap.<String, Object>builder().putAll(lval.values).putAll(rval.values).build(),
          loc);
    }
  }
}
