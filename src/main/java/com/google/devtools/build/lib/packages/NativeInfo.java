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
import com.google.devtools.build.lib.events.Location;
import java.util.Map;

/** Base class for native implementations of {@link Info}. */
// todo(vladmos,dslomov): make abstract once DefaultInfo stops instantiating it.
public class NativeInfo extends Info {
  protected final ImmutableMap<String, Object> values;

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

  public NativeInfo(NativeProvider<?> provider) {
    super(provider, Location.BUILTIN);
    this.values = ImmutableMap.of();
  }

  public NativeInfo(NativeProvider<?> provider, Map<String, Object> values, Location loc) {
    super(provider, loc);
    this.values = copyValues(values);
  }

  public NativeInfo(NativeProvider<?> provider, Map<String, Object> values) {
    this(provider, values, Location.BUILTIN);
  }
}
