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
public class ClassProbesMapper extends ClassProbesVisitor {
  private Map<Integer, BranchExp> classLineToBranchExp;

  private IFilter filter;

  private SimpleFilterContext filterContext;

  private StringPool stringPool;

  public Map<Integer, BranchExp> result() {
    return classLineToBranchExp;
  }

  /** Create a new probe mapper object. */
  public ClassProbesMapper(String className) {
    classLineToBranchExp = new TreeMap<Integer, BranchExp>();
    filter = Filters.all();
    filterContext = new SimpleFilterContext();
    stringPool = new StringPool();
    filterContext.setClassName(stringPool.get(className));
  }

  @Override
  public AnnotationVisitor visitAnnotation(final String desc, final boolean visible) {
    filterContext.addClassAnnotations(desc);
    return super.visitAnnotation(desc, visible);
  }

  @Override
  public void visitAttribute(final Attribute attribute) {
    filterContext.addClassAttribute(attribute.type);
  }

  @Override
  public void visitSource(final String source, final String debug) {
    filterContext.setSourceFileName(stringPool.get(source));
    filterContext.setSourceDebugExtension(debug);
  }

  @Override
  public void visit(int version, int access, String name, String signature, String superName,
      String[] interfaces) {
    filterContext.setSuperClassName(stringPool.get(name));
  }

  /** Returns a visitor for mapping method code. */
  @Override
  public MethodProbesVisitor visitMethod(
      int access, String name, String desc, String signature, String[] exceptions) {
    return new MethodProbesMapper(filterContext, filter) {

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
  private class SimpleFilterContext implements IFilterContext {

    String sourceFileName;
    String sourceDebugExtension;
    String className;
    String superClassName;
    Set<String> classAnnotations = new HashSet<>();
    Set<String> classAttributes = new HashSet<>();

    public void setClassName(String className) {
      this.className = className;
    }

    @Override
    public String getClassName() {
      return className;
    }

    public void setSuperClassName(String superClassName) {
      this.superClassName = superClassName;
    }

    @Override
    public String getSuperClassName() {
      return superClassName;
    }

    public void addClassAnnotations(String annotation) {
      classAnnotations.add(annotation);
    }

    @Override
    public Set<String> getClassAnnotations() {
      return classAnnotations;
    }

    public void addClassAttribute(String attribute) {
      classAttributes.add(attribute);
    }

    @Override
    public Set<String> getClassAttributes() {
      return classAttributes;
    }


    public void setSourceFileName(String sourceFileName) {
      this.sourceFileName = sourceFileName;
    }

    @Override
    public String getSourceFileName() {
      return sourceFileName;
    }

    public void setSourceDebugExtension(String sourceDebugExtension) {
      this.sourceDebugExtension = sourceDebugExtension;
    }

    @Override
    public String getSourceDebugExtension() {
      return sourceDebugExtension;
    }
  }
}
