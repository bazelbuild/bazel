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

import java.io.IOException;
import java.io.Writer;
import org.objectweb.asm.ClassWriter;
import org.objectweb.asm.commons.InstructionAdapter;

/**
 * Represents a field and its initializer (where initialization is either part of the field
 * definition, or done via code in the static clinit function).
 */
public interface FieldInitializer {
  /**
   * Write the bytecode for the field definition.
   *
   * @return true if the initializer is deferred to clinit code.
   */
  boolean writeFieldDefinition(ClassWriter cw, boolean isFinal, boolean annotateTransitiveFields);

  /**
   * Write the bytecode for the clinit portion of initializer.
   *
   * @return the number of stack slots needed for the code.
   */
  int writeCLInit(InstructionAdapter insts, String className);

  /** Write the source code for the initializer to the given writer. */
  void writeInitSource(Writer writer, boolean finalFields) throws IOException;

  String getFieldName();
}
