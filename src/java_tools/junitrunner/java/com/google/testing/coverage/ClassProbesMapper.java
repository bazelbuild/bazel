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

import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import org.jacoco.core.internal.analysis.StringPool;
import org.jacoco.core.internal.analysis.filter.Filters;
import org.jacoco.core.internal.analysis.filter.IFilter;
import org.jacoco.core.internal.analysis.filter.IFilterContext;
import org.jacoco.core.internal.flow.ClassProbesVisitor;
import org.jacoco.core.internal.flow.MethodProbesVisitor;
import org.objectweb.asm.AnnotationVisitor;
import org.objectweb.asm.Attribute;
import org.objectweb.asm.FieldVisitor;

/** A visitor that maps each source code line to the probes corresponding to the lines. */
public class ClassProbesMapper extends ClassProbesVisitor implements IFilterContext {
  private Map<Integer, BranchExp> classLineToBranchExp;

  private IFilter allFilters = Filters.all();

  private StringPool stringPool;

  // IFilterContext state updating during visitations
  private String className;
  private String superClassName;
  private Set<String> classAnnotations = new HashSet<>();
  private Set<String> classAttributes = new HashSet<>();
  private String sourceFileName;
  private String sourceDebugExtension;

  public Map<Integer, BranchExp> result() {
    return classLineToBranchExp;
  }

  /** Create a new probe mapper object. */
  public ClassProbesMapper(String className) {
    classLineToBranchExp = new TreeMap<Integer, BranchExp>();
    stringPool = new StringPool();
    this.className = stringPool.get(className);
  }

  @Override
  public AnnotationVisitor visitAnnotation(final String desc, final boolean visible) {
    classAnnotations.add(desc);
    return super.visitAnnotation(desc, visible);
  }

  @Override
  public void visitAttribute(final Attribute attribute) {
    classAttributes.add(attribute.type);
  }

  @Override
  public void visitSource(final String source, final String debug) {
    sourceFileName = stringPool.get(source);
    sourceDebugExtension = debug;
  }

  @Override
  public void visit(
      int version,
      int access,
      String name,
      String signature,
      String superName,
      String[] interfaces) {
    superClassName = stringPool.get(name);
  }

  /** Returns a visitor for mapping method code. */
  @Override
  public MethodProbesVisitor visitMethod(
      int access, String name, String desc, String signature, String[] exceptions) {
    return new MethodProbesMapper(this, allFilters) {

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
    // Nothing to do. Maybe perform some checks here.
  }

  @Override
  public String getClassName() {
    return className;
  }

  @Override
  public String getSuperClassName() {
    return superClassName;
  }

  @Override
  public String getSourceDebugExtension() {
    return sourceDebugExtension;
  }

  @Override
  public String getSourceFileName() {
    return sourceFileName;
  }

  @Override
  public Set<String> getClassAnnotations() {
    return classAnnotations;
  }

  @Override
  public Set<String> getClassAttributes() {
    return classAttributes;
  }
}
