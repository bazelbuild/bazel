// Copyright 2026 The Bazel Authors. All rights reserved.
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

package net.starlark.java.lib;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import java.util.Map;
import javax.annotation.Nullable;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.Structure;

/** A simple {@link Structure} implementation that wraps a map of fields. */
public class MapWrapperStructure implements Structure {
  protected final ImmutableMap<String, Object> fields;

  public MapWrapperStructure(Map<String, Object> fields) {
    this.fields = ImmutableMap.copyOf(fields);
  }

  @Override
  @Nullable
  public Object getValue(String name) {
    return fields.get(name);
  }

  @Override
  @Nullable
  public ImmutableSet<String> getFieldNames() {
    return fields.keySet();
  }

  @Override
  @Nullable
  public String getErrorMessageForUnknownField(String field) {
    return null;
  }

  @Override
  public void repr(Printer printer, StarlarkSemantics semantics) {
    boolean first = true;
    printer.append("struct(");
    for (var field : fields.entrySet()) {
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
}
