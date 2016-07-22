// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.resources;

import com.google.common.base.Preconditions;
import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;

import org.objectweb.asm.ClassWriter;
import org.objectweb.asm.Type;
import org.objectweb.asm.commons.InstructionAdapter;

import java.io.IOException;
import java.io.Writer;

/**
 * Models an int[] field initializer.
 */
public final class IntArrayFieldInitializer implements FieldInitializer {

  public static final String DESC = "[I";
  private final String fieldName;
  private final ImmutableCollection<Integer> values;

  public IntArrayFieldInitializer(String fieldName, ImmutableCollection<Integer> values) {
    this.fieldName = fieldName;
    this.values = values;
  }

  public static FieldInitializer of(String name, String value) {
    Preconditions.checkArgument(value.startsWith("{ "), "Expected list starting with { ");
    Preconditions.checkArgument(value.endsWith(" }"), "Expected list ending with } ");
    // Check for an empty list, which is "{ }".
    if (value.length() < 4) {
      return new IntArrayFieldInitializer(name, ImmutableList.<Integer>of());
    }
    ImmutableList.Builder<Integer> intValues = ImmutableList.builder();
    String trimmedValue = value.substring(2, value.length() - 2);
    Iterable<String> valueStrings = Splitter.on(',')
        .trimResults()
        .split(trimmedValue);
    for (String valueString : valueStrings) {
      intValues.add(Integer.decode(valueString));
    }
    return new IntArrayFieldInitializer(name, intValues.build());
  }

  @Override
  public boolean writeFieldDefinition(ClassWriter cw, int accessLevel, boolean isFinal) {
    cw.visitField(accessLevel, fieldName, DESC, null, null)
        .visitEnd();
    return true;
  }

  @Override
  public int writeCLInit(InstructionAdapter insts, String className) {
    insts.iconst(values.size());
    insts.newarray(Type.INT_TYPE);
    int curIndex = 0;
    for (Integer value : values) {
      insts.dup();
      insts.iconst(curIndex);
      insts.iconst(value);
      insts.astore(Type.INT_TYPE);
      ++curIndex;
    }
    insts.putstatic(className, fieldName, DESC);
    // Needs up to 4 stack slots for: the array ref for the putstatic, the dup of the array ref
    // for the store, the index, and the value to store.
    return 4;
  }

  @Override
  public void writeInitSource(Writer writer) throws IOException {
    StringBuilder builder = new StringBuilder();
    boolean first = true;
    for (Integer attrId : values) {
      if (first) {
        first = false;
        builder.append(String.format("0x%x", attrId));
      } else {
        builder.append(String.format(", 0x%x", attrId));
      }
    }
    writer.write(String.format("        public static int[] %s = { %s };\n",
        fieldName, builder.toString()));
  }
}
