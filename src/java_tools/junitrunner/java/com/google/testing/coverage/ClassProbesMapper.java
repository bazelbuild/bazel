// Copyright 2016 The Bazel Authors. All Rights Reserved.
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

package com.google.testing.coverage;

import java.util.Map;
import java.util.TreeMap;
import org.jacoco.core.internal.flow.ClassProbesVisitor;
import org.jacoco.core.internal.flow.MethodProbesVisitor;
import org.objectweb.asm.FieldVisitor;

/** A visitor that maps each source code line to the probes corresponding to the lines. */
public class ClassProbesMapper extends ClassProbesVisitor {
  private Map<Integer, BranchExp> classLineToBranchExp;

  public Map<Integer, BranchExp> result() {
    return classLineToBranchExp;
  }

  /** Create a new probe mapper object. */
  public ClassProbesMapper() {
    classLineToBranchExp = new TreeMap<Integer, BranchExp>();
  }

  /** Returns a visitor for mapping method code. */
  @Override
  public MethodProbesVisitor visitMethod(
      int access, String name, String desc, String signature, String[] exceptions) {
    return new MethodProbesMapper() {
      @Override
      public void visitEnd() {
        super.visitEnd();
        classLineToBranchExp.putAll(result());
      }
    };
  }

  @Override
  public FieldVisitor visitField(
      int access, String name, String desc, String signature, Object value) {
    return super.visitField(access, name, desc, signature, value);
  }

  @Override
  public void visitTotalProbeCount(int count) {
    // Nothing to do. Maybe perform some sanity checks here.
  }
}
