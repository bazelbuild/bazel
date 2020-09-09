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
package com.google.devtools.build.android.desugar.scan;

import static com.google.common.base.Preconditions.checkArgument;

import com.google.common.collect.ImmutableSet;
import javax.annotation.Nullable;
import org.objectweb.asm.AnnotationVisitor;
import org.objectweb.asm.ClassReader;
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.FieldVisitor;
import org.objectweb.asm.Handle;
import org.objectweb.asm.Label;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.Type;
import org.objectweb.asm.TypePath;

/** {@link ClassVisitor} that records references to classes starting with a given prefix. */
class PrefixReferenceScanner extends ClassVisitor {

  /**
   * Returns references with the given prefix in the given class.
   *
   * @param prefix an internal name prefix, typically a package such as {@code com/google/}
   */
  public static ImmutableSet<KeepReference> scan(ClassReader reader, String prefix) {
    PrefixReferenceScanner scanner = new PrefixReferenceScanner(prefix);
    // Frames irrelevant for Android so skip them.  Don't skip debug info in case the class we're
    // visiting has local variable tables (typically it doesn't anyways).
    reader.accept(scanner, ClassReader.SKIP_FRAMES);
    return scanner.roots.build();
  }

  private final ImmutableSet.Builder<KeepReference> roots = ImmutableSet.builder();
  private final PrefixReferenceMethodVisitor mv = new PrefixReferenceMethodVisitor();
  private final PrefixReferenceFieldVisitor fv = new PrefixReferenceFieldVisitor();
  private final PrefixReferenceAnnotationVisitor av = new PrefixReferenceAnnotationVisitor();

  private final String prefix;

  public PrefixReferenceScanner(String prefix) {
    super(Opcodes.ASM8);
    this.prefix = prefix;
  }

  @Override
  public void visit(
      int version,
      int access,
      String name,
      String signature,
      String superName,
      String[] interfaces) {
    checkArgument(!name.startsWith(prefix));
    if (superName != null) {
      classReference(superName);
    }
    classReferences(interfaces);
    super.visit(version, access, name, signature, superName, interfaces);
  }

  @Override
  public AnnotationVisitor visitAnnotation(String desc, boolean visible) {
    typeReference(desc);
    return av;
  }

  @Override
  public void visitOuterClass(String owner, String name, String desc) {
    classReference(owner);
    if (desc != null) {
      typeReference(Type.getMethodType(desc));
    }
  }

  @Override
  public AnnotationVisitor visitTypeAnnotation(
      int typeRef, TypePath typePath, String desc, boolean visible) {
    typeReference(desc);
    return av;
  }

  @Override
  public void visitInnerClass(String name, String outerName, String innerName, int access) {
    classReference(name);
    if (outerName != null) {
      classReference(outerName);
    }
  }

  @Override
  public FieldVisitor visitField(
      int access, String name, String desc, String signature, Object value) {
    typeReference(desc);
    return fv;
  }

  @Override
  public MethodVisitor visitMethod(
      int access, String name, String desc, String signature, String[] exceptions) {
    typeReference(Type.getMethodType(desc));
    classReferences(exceptions);
    return mv;
  }

  private void classReferences(@Nullable String[] internalNames) {
    if (internalNames != null) {
      for (String itf : internalNames) {
        classReference(itf);
      }
    }
  }

  // The following methods are package-private so they don't incur bridge methods when called from
  // inner classes below.

  void classReference(String internalName) {
    checkArgument(internalName.charAt(0) != '[' && internalName.charAt(0) != '(', internalName);
    checkArgument(!internalName.endsWith(";"), internalName);
    if (internalName.startsWith(prefix)) {
      roots.add(KeepReference.classReference(internalName));
    }
  }

  void objectReference(String internalName) {
    // don't call this for method types, convert to Type instead
    checkArgument(internalName.charAt(0) != '(', internalName);
    if (internalName.charAt(0) == '[') {
      typeReference(internalName);
    } else {
      classReference(internalName);
    }
  }

  void typeReference(String typeDesc) {
    // don't call this for method types, convert to Type instead
    checkArgument(typeDesc.charAt(0) != '(', typeDesc);

    int lpos = typeDesc.lastIndexOf('[') + 1;
    if (typeDesc.charAt(lpos) == 'L') {
      checkArgument(typeDesc.endsWith(";"), typeDesc);
      classReference(typeDesc.substring(lpos, typeDesc.length() - 1));
    } else {
      // else primitive or primitive array
      checkArgument(typeDesc.length() == lpos + 1, typeDesc);
      switch (typeDesc.charAt(lpos)) {
        case 'B':
        case 'C':
        case 'S':
        case 'I':
        case 'J':
        case 'D':
        case 'F':
        case 'Z':
          break;
        default:
          throw new AssertionError("Unexpected type descriptor: " + typeDesc);
      }
    }
  }

  void typeReference(Type type) {
    switch (type.getSort()) {
      case Type.ARRAY:
        typeReference(type.getElementType());
        break;
      case Type.OBJECT:
        classReference(type.getInternalName());
        break;

      case Type.METHOD:
        for (Type param : type.getArgumentTypes()) {
          typeReference(param);
        }
        typeReference(type.getReturnType());
        break;

      default:
        break;
    }
  }

  void fieldReference(String owner, String name, String desc) {
    objectReference(owner);
    typeReference(desc);
    if (owner.startsWith(prefix)) {
      roots.add(KeepReference.memberReference(owner, name, desc));
    }
  }

  void methodReference(String owner, String name, String desc) {
    checkArgument(desc.charAt(0) == '(', desc);
    objectReference(owner);
    typeReference(Type.getMethodType(desc));
    if (owner.startsWith(prefix)) {
      roots.add(KeepReference.memberReference(owner, name, desc));
    }
  }

  void handleReference(Handle handle) {
    switch (handle.getTag()) {
      case Opcodes.H_GETFIELD:
      case Opcodes.H_GETSTATIC:
      case Opcodes.H_PUTFIELD:
      case Opcodes.H_PUTSTATIC:
        fieldReference(handle.getOwner(), handle.getName(), handle.getDesc());
        break;

      default:
        methodReference(handle.getOwner(), handle.getName(), handle.getDesc());
        break;
    }
  }

  private class PrefixReferenceMethodVisitor extends MethodVisitor {

    public PrefixReferenceMethodVisitor() {
      super(Opcodes.ASM8);
    }

    @Override
    public AnnotationVisitor visitAnnotationDefault() {
      return av;
    }

    @Override
    public AnnotationVisitor visitAnnotation(String desc, boolean visible) {
      typeReference(desc);
      return av;
    }

    @Override
    public AnnotationVisitor visitTypeAnnotation(
        int typeRef, TypePath typePath, String desc, boolean visible) {
      // Adjust the type annotation descriptor for proguarded byte code. b/166658450
      if (desc.contains("/") && !desc.startsWith("L")) {
        desc = "L" + desc + ";";
      }
      Type type = Type.getType(desc);
      typeReference(type);
      return av;
    }

    @Override
    public AnnotationVisitor visitParameterAnnotation(int parameter, String desc, boolean visible) {
      typeReference(desc);
      return av;
    }

    @Override
    public void visitTypeInsn(int opcode, String type) {
      objectReference(type);
    }

    @Override
    public void visitFieldInsn(int opcode, String owner, String name, String desc) {
      fieldReference(owner, name, desc);
    }

    @Override
    @SuppressWarnings("deprecation") // Implementing deprecated method to be sure
    public void visitMethodInsn(int opcode, String owner, String name, String desc) {
      visitMethodInsn(opcode, owner, name, desc, opcode == Opcodes.INVOKEINTERFACE);
    }

    @Override
    public void visitMethodInsn(int opcode, String owner, String name, String desc, boolean itf) {
      methodReference(owner, name, desc);
    }

    @Override
    public void visitInvokeDynamicInsn(String name, String desc, Handle bsm, Object... bsmArgs) {
      typeReference(Type.getMethodType(desc));
      handleReference(bsm);
      for (Object bsmArg : bsmArgs) {
        visitConstant(bsmArg);
      }
    }

    @Override
    public void visitLdcInsn(Object cst) {
      visitConstant(cst);
    }

    private void visitConstant(Object cst) {
      if (cst instanceof Type) {
        typeReference((Type) cst);
      } else if (cst instanceof Handle) {
        handleReference((Handle) cst);
      } else {
        // Check for other expected types as javadoc recommends
        checkArgument(
            cst instanceof String
                || cst instanceof Integer
                || cst instanceof Long
                || cst instanceof Float
                || cst instanceof Double,
            "Unexpected constant: ",
            cst);
      }
    }

    @Override
    public void visitMultiANewArrayInsn(String desc, int dims) {
      typeReference(desc);
    }

    @Override
    public AnnotationVisitor visitInsnAnnotation(
        int typeRef, TypePath typePath, String desc, boolean visible) {
      typeReference(desc);
      return av;
    }

    @Override
    public void visitTryCatchBlock(Label start, Label end, Label handler, String type) {
      if (type != null) {
        classReference(type);
      }
    }

    @Override
    public AnnotationVisitor visitTryCatchAnnotation(
        int typeRef, TypePath typePath, String desc, boolean visible) {
      typeReference(desc);
      return av;
    }

    @Override
    public void visitLocalVariable(
        String name, String desc, String signature, Label start, Label end, int index) {
      typeReference(desc);
    }

    @Override
    public AnnotationVisitor visitLocalVariableAnnotation(
        int typeRef,
        TypePath typePath,
        Label[] start,
        Label[] end,
        int[] index,
        String desc,
        boolean visible) {
      typeReference(desc);
      return av;
    }
  }

  private class PrefixReferenceFieldVisitor extends FieldVisitor {

    public PrefixReferenceFieldVisitor() {
      super(Opcodes.ASM8);
    }

    @Override
    public AnnotationVisitor visitAnnotation(String desc, boolean visible) {
      typeReference(desc);
      return av;
    }

    @Override
    public AnnotationVisitor visitTypeAnnotation(
        int typeRef, TypePath typePath, String desc, boolean visible) {
      typeReference(desc);
      return av;
    }
  }

  private class PrefixReferenceAnnotationVisitor extends AnnotationVisitor {

    public PrefixReferenceAnnotationVisitor() {
      super(Opcodes.ASM8);
    }

    @Override
    public void visit(String name, Object value) {
      if (value instanceof Type) {
        typeReference((Type) value);
      }
    }

    @Override
    public void visitEnum(String name, String desc, String value) {
      fieldReference(desc.substring(1, desc.length() - 1), value, desc);
    }

    @Override
    public AnnotationVisitor visitAnnotation(String name, String desc) {
      typeReference(desc);
      return av;
    }

    @Override
    public AnnotationVisitor visitArray(String name) {
      return av;
    }
  }
}
