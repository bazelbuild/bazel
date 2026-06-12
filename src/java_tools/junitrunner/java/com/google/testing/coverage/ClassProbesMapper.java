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

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
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

/** A visitor that maps each source code line to the probes corresponding to the lines. */
public class ClassProbesMapper extends ClassProbesVisitor implements IFilterContext {
  private final Map<Integer, BranchExpression> branchExpressions;
  private final Map<Integer, CoverageExpression> lineExpressions;
  private final ArrayList<MethodInfo> methods;

  private final IFilter allFilters = Filters.all();

  private StringPool stringPool;

  // IFilterContext state that is updated during visitations
  private String className;
  private String superClassName;
  private Set<String> classAnnotations = new HashSet<>();
  private Set<String> classAttributes = new HashSet<>();
  private String sourceFileName;
  private String sourceDebugExtension;

  /** Returns a map of line number to the branch expressions on that line. */
  public Map<Integer, BranchExpression> getBranchExpressions() {
    return branchExpressions;
  }

  /** Returns a map of line number to the coverage expression on that line. */
  public Map<Integer, CoverageExpression> getLineExpressions() {
    return lineExpressions;
  }

  /** Returns a list of methods in the class. */
  public List<MethodInfo> getMethods() {
    return methods;
  }

  /**
   * Create a new probe mapper object.
   *
   * @param className The full name of the class being mapped where "." is replaced with "/". See
   *     https://asm.ow2.io/javadoc/org/objectweb/asm/Type.html#getInternalName()
   */
  public ClassProbesMapper(String className) {
    branchExpressions = new TreeMap<>();
    lineExpressions = new TreeMap<>();
    methods = new ArrayList<>();
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
    superClassName = stringPool.get(superName);
  }

  /** Returns a visitor for mapping method code. */
  @Override
  public MethodProbesVisitor visitMethod(
      int access, String name, String desc, String signature, String[] exceptions) {
    return new MethodProbesMapper(this, allFilters) {
      @Override
      public void visitEnd() {
        super.visitEnd();
        branchExpressions.putAll(this.getBranchExpressions());
        lineExpressions.putAll(this.getLineExpressions());
        String method = constructFunctionName(className, name, desc);
        methods.add(MethodInfo.create(method, getMethodLineStart(), getMethodExpression()));
      }
    };
  }

  @Override
  public void visitTotalProbeCount(int count) {
    // Nothing to do. Maybe perform some sanity checks here.
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

  private static String constructFunctionName(String clsName, String methodName, String desc) {
    // The lcov spec doesn't of course cover Java formats, so we output the method signature.
    // lcov_merger doesn't seem to care about these entries.
    return String.format("%s::%s %s", clsName, methodName, desc);
  }
}
