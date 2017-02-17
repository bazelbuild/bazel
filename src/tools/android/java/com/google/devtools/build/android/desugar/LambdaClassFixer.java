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
package com.google.devtools.build.android.desugar;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import java.util.HashSet;
import java.util.LinkedHashSet;
import org.objectweb.asm.AnnotationVisitor;
import org.objectweb.asm.ClassReader;
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.FieldVisitor;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.Type;
import org.objectweb.asm.tree.AbstractInsnNode;
import org.objectweb.asm.tree.MethodNode;
import org.objectweb.asm.tree.TypeInsnNode;

/**
 * Visitor intended to fix up lambda classes to match assumptions made in {@link LambdaDesugaring}.
 * Specifically this includes fixing visibilities and generating any missing factory methods.
 *
 * <p>Each instance can only visit one class.  This is because the signature of the needed factory
 * method is passed into the constructor.
 */
class LambdaClassFixer extends ClassVisitor {

  /** Magic method name used by {@link java.lang.invoke.LambdaMetafactory} */
  public static final String FACTORY_METHOD_NAME = "get$Lambda";

  private final LambdaInfo lambdaInfo;
  private final ClassReaderFactory factory;
  private final ImmutableSet<String> interfaceLambdaMethods;
  private final boolean allowDefaultMethods;
  private final HashSet<String> implementedMethods = new HashSet<>();
  private final LinkedHashSet<String> methodsToMoveIn = new LinkedHashSet<>();

  private String internalName;
  private ImmutableList<String> interfaces;

  private boolean hasState;
  private boolean hasFactory;

  private String desc;
  private String signature;
  private String[] exceptions;


  public LambdaClassFixer(ClassVisitor dest, LambdaInfo lambdaInfo, ClassReaderFactory factory,
      ImmutableSet<String> interfaceLambdaMethods, boolean allowDefaultMethods) {
    super(Opcodes.ASM5, dest);
    this.lambdaInfo = lambdaInfo;
    this.factory = factory;
    this.interfaceLambdaMethods = interfaceLambdaMethods;
    this.allowDefaultMethods = allowDefaultMethods;
  }

  public String getInternalName() {
    return internalName;
  }

  @Override
  public void visit(
      int version,
      int access,
      String name,
      String signature,
      String superName,
      String[] interfaces) {
    checkArgument(BitFlags.noneSet(access, Opcodes.ACC_INTERFACE), "Not a class: %s", name);
    checkState(internalName == null, "Already visited %s, can't reuse for %s", internalName, name);
    internalName = name;
    hasState = false;
    hasFactory = false;
    desc = null;
    this.signature = null;
    exceptions = null;
    this.interfaces = ImmutableList.copyOf(interfaces);
    super.visit(version, access, name, signature, superName, interfaces);
  }

  @Override
  public FieldVisitor visitField(
      int access, String name, String desc, String signature, Object value) {
    hasState = true;
    return super.visitField(access, name, desc, signature, value);
  }

  @Override
  public MethodVisitor visitMethod(
      int access, String name, String desc, String signature, String[] exceptions) {
    if (name.equals("writeReplace") && BitFlags.noneSet(access, Opcodes.ACC_STATIC)
        && desc.equals("()Ljava/lang/Object;")) {
      // Lambda serialization hooks use java/lang/invoke/SerializedLambda, which isn't available on
      // Android. Since Jack doesn't do anything special for serializable lambdas we just drop these
      // serialization hooks.
      // https://docs.oracle.com/javase/8/docs/platform/serialization/spec/output.html#a5324 gives
      // details on the role and signature of this method.
      return null;
    }
    if (BitFlags.noneSet(access, Opcodes.ACC_ABSTRACT | Opcodes.ACC_STATIC)) {
      // Keep track of instance methods implemented in this class for later.  Since this visitor
      // is intended for lambda classes, no need to look at the superclass.
      implementedMethods.add(name + ":" + desc);
    }
    if (FACTORY_METHOD_NAME.equals(name)) {
      hasFactory = true;
      access &= ~Opcodes.ACC_PRIVATE; // make factory method accessible
    } else if ("<init>".equals(name)) {
      this.desc = desc;
      this.signature = signature;
      this.exceptions = exceptions;
    }
    MethodVisitor methodVisitor =
        new LambdaClassMethodRewriter(super.visitMethod(access, name, desc, signature, exceptions));
    if (!lambdaInfo.bridgeMethod().equals(lambdaInfo.methodReference())) {
      // Skip UseBridgeMethod unless we actually need it
      methodVisitor =
          new UseBridgeMethod(methodVisitor, lambdaInfo, access, name, desc, signature, exceptions);
    }
    if (!FACTORY_METHOD_NAME.equals(name) && !"<init>".equals(name)) {
      methodVisitor = new LambdaClassInvokeSpecialRewriter(methodVisitor);
    }
    return methodVisitor;
  }

  @Override
  public void visitEnd() {
    checkState(!hasState || hasFactory,
        "Expected factory method for capturing lambda %s", internalName);
    if (!hasFactory) {
      // Fake factory method if LambdaMetafactory didn't generate it
      checkState(signature == null,
          "Didn't expect generic constructor signature %s %s", internalName, signature);

      // Since this is a stateless class we populate and use a static singleton field "$instance"
      String singletonFieldDesc = Type.getObjectType(internalName).getDescriptor();
      super.visitField(
          Opcodes.ACC_PRIVATE | Opcodes.ACC_STATIC | Opcodes.ACC_FINAL,
          "$instance",
          singletonFieldDesc,
          (String) null,
          (Object) null)
          .visitEnd();

      MethodVisitor codeBuilder =
          super.visitMethod(
              Opcodes.ACC_STATIC,
              "<clinit>",
              "()V",
              (String) null,
              new String[0]);
      codeBuilder.visitTypeInsn(Opcodes.NEW, internalName);
      codeBuilder.visitInsn(Opcodes.DUP);
      codeBuilder.visitMethodInsn(Opcodes.INVOKESPECIAL, internalName, "<init>",
          checkNotNull(desc, "didn't see a constructor for %s", internalName), /*itf*/ false);
      codeBuilder.visitFieldInsn(Opcodes.PUTSTATIC, internalName, "$instance", singletonFieldDesc);
      codeBuilder.visitInsn(Opcodes.RETURN);
      codeBuilder.visitMaxs(2, 0); // two values are pushed onto the stack
      codeBuilder.visitEnd();

      codeBuilder = // reuse codeBuilder variable to avoid accidental additions to previous method
          super.visitMethod(
              Opcodes.ACC_STATIC,
              FACTORY_METHOD_NAME,
              lambdaInfo.factoryMethodDesc(),
              (String) null,
              exceptions);
      codeBuilder.visitFieldInsn(Opcodes.GETSTATIC, internalName, "$instance", singletonFieldDesc);
      codeBuilder.visitInsn(Opcodes.ARETURN);
      codeBuilder.visitMaxs(1, 0); // one value on the stack
    }

    copyRewrittenLambdaMethods();

    if (!allowDefaultMethods) {
      copyBridgeMethods(interfaces);
    }
    super.visitEnd();
  }

  private void copyRewrittenLambdaMethods() {
    for (String rewritten : methodsToMoveIn) {
      String interfaceInternalName = rewritten.substring(0, rewritten.indexOf('#'));
      String methodName = rewritten.substring(interfaceInternalName.length() + 1);
      ClassReader bytecode = checkNotNull(factory.readIfKnown(interfaceInternalName),
          "Couldn't load interface with lambda method %s", rewritten);
      CopyOneMethod copier = new CopyOneMethod(methodName);
      // TODO(kmb): Set source file attribute for lambda classes so lambda debug info makes sense
      bytecode.accept(copier, ClassReader.SKIP_DEBUG);
    }
  }

  private void copyBridgeMethods(ImmutableList<String> interfaces) {
    for (String implemented : interfaces) {
      ClassReader bytecode = factory.readIfKnown(implemented);
      if (bytecode != null) {
        // Don't copy line numbers and local variable tables.  They would be misleading or wrong
        // and other methods in generated lambda classes don't have debug info either.
        bytecode.accept(new CopyBridgeMethods(), ClassReader.SKIP_DEBUG);
      } // else the interface is defined in a different Jar, which we can ignore here
    }
  }

  /**
   * Rewriter for methods in generated lambda classes.
   */
  private class LambdaClassMethodRewriter extends MethodVisitor {
    public LambdaClassMethodRewriter(MethodVisitor dest) {
      super(Opcodes.ASM5, dest);
    }

    @Override
    public void visitMethodInsn(int opcode, String owner, String name, String desc, boolean itf) {
      String method = owner + "#" + name;
      if (interfaceLambdaMethods.contains(method)) {
        // Rewrite invocations of lambda methods in interfaces to anticipate the lambda method being
        // moved into the lambda class (i.e., the class being visited here).
        checkArgument(opcode == Opcodes.INVOKESTATIC, "Cannot move instance method %s", method);
        owner = internalName;
        itf = false; // owner was interface but is now a class
        methodsToMoveIn.add(method);
      } else if (name.startsWith("lambda$")) {
        // Reflect renaming of lambda$ instance methods to avoid accidental overrides
        name = LambdaDesugaring.uniqueInPackage(owner, name);
      }
      super.visitMethodInsn(opcode, owner, name, desc, itf);
    }

    @Override
    public AnnotationVisitor visitAnnotation(String desc, boolean visible) {
      // Drop annotation that's part of the generated lambda class that's not available on Android.
      // Proguard complains about this otherwise.
      if ("Ljava/lang/invoke/LambdaForm$Hidden;".equals(desc)) {
        return null;
      }
      return super.visitAnnotation(desc, visible);
    }
  }

  /** Rewriter for invokespecial in generated lambda classes. */
  private static class LambdaClassInvokeSpecialRewriter extends MethodVisitor {

    public LambdaClassInvokeSpecialRewriter(MethodVisitor dest) {
      super(Opcodes.ASM5, dest);
    }

    @Override
    public void visitMethodInsn(int opcode, String owner, String name, String desc, boolean itf) {
      if (opcode == Opcodes.INVOKESPECIAL && name.startsWith("lambda$")) {
        opcode = itf ? Opcodes.INVOKEINTERFACE : Opcodes.INVOKEVIRTUAL;
      }

      super.visitMethodInsn(opcode, owner, name, desc, itf);
    }
  }

  /**
   * Visitor that copies bridge methods from the visited interface into the class visited by the
   * surrounding {@link LambdaClassFixer}. Descends recursively into interfaces extended by the
   * visited interface.
   */
  private class CopyBridgeMethods extends ClassVisitor {

    @SuppressWarnings("hiding") private ImmutableList<String> interfaces;

    public CopyBridgeMethods() {
      // No delegate visitor; instead we'll add methods to the outer class's delegate where needed
      super(Opcodes.ASM5);
    }

    @Override
    public void visit(
        int version,
        int access,
        String name,
        String signature,
        String superName,
        String[] interfaces) {
      checkArgument(BitFlags.isSet(access, Opcodes.ACC_INTERFACE));
      checkState(this.interfaces == null);
      this.interfaces = ImmutableList.copyOf(interfaces);
    }

    @Override
    public MethodVisitor visitMethod(
        int access, String name, String desc, String signature, String[] exceptions) {
      if ((access & (Opcodes.ACC_BRIDGE | Opcodes.ACC_ABSTRACT | Opcodes.ACC_STATIC))
          == Opcodes.ACC_BRIDGE) {
        // Only copy bridge methods--hand-written default methods are not supported--and only if
        // we haven't seen the method already.
        if (implementedMethods.add(name + ":" + desc)) {
          return new AvoidJacocoInit(
              LambdaClassFixer.super.visitMethod(access, name, desc, signature, exceptions));
        }
      }
      return null;
    }

    @Override
    public void visitEnd() {
      copyBridgeMethods(this.interfaces);
    }
  }

  private class CopyOneMethod extends ClassVisitor {

    private final String methodName;
    private int copied = 0;

    public CopyOneMethod(String methodName) {
      // No delegate visitor; instead we'll add methods to the outer class's delegate where needed
      super(Opcodes.ASM5);
      this.methodName = methodName;
    }

    @Override
    public void visit(
        int version,
        int access,
        String name,
        String signature,
        String superName,
        String[] interfaces) {
      checkArgument(BitFlags.isSet(access, Opcodes.ACC_INTERFACE));
    }

    @Override
    public MethodVisitor visitMethod(
        int access, String name, String desc, String signature, String[] exceptions) {
      if (name.equals(methodName)) {
        checkState(copied == 0, "Found unexpected second method %s with descriptor %s", name, desc);
        ++copied;
        return new AvoidJacocoInit(
            LambdaClassFixer.super.visitMethod(access, name, desc, signature, exceptions));
      }
      return null;
    }
  }

  /**
   * Method visitor that rewrites {@code $jacocoInit()} calls to equivalent field accesses.
   *
   * <p>This class should only be used to visit interface methods and assumes that the code in
   * {@code $jacocoInit()} is always executed in the interface's static initializer, which is the
   * case in the absence of hand-written static or default interface methods (which
   * {@link Java7Compatibility} makes sure of).
   */
  private static class AvoidJacocoInit extends MethodVisitor {
    public AvoidJacocoInit(MethodVisitor dest) {
      super(Opcodes.ASM5, dest);
    }

    @Override
    public void visitMethodInsn(int opcode, String owner, String name, String desc, boolean itf) {
      if (opcode == Opcodes.INVOKESTATIC && "$jacocoInit".equals(name)) {
        // Rewrite $jacocoInit() calls to just read the $jacocoData field
        super.visitFieldInsn(Opcodes.GETSTATIC, owner, "$jacocoData", "[Z");
      } else {
        super.visitMethodInsn(opcode, owner, name, desc, itf);
      }
    }
  }

  private static class UseBridgeMethod extends MethodNode {

    private final MethodVisitor dest;
    private final LambdaInfo lambdaInfo;

    public UseBridgeMethod(MethodVisitor dest, LambdaInfo lambdaInfo,
        int access, String name, String desc, String signature, String[] exceptions) {
      super(Opcodes.ASM5, access, name, desc, signature, exceptions);
      this.dest = dest;
      this.lambdaInfo = lambdaInfo;
    }

    @Override
    public void visitMethodInsn(int opcode, String owner, String name, String desc, boolean itf) {
      if (owner.equals(lambdaInfo.methodReference().getOwner())
          && name.equals(lambdaInfo.methodReference().getName())
          && desc.equals(lambdaInfo.methodReference().getDesc())) {
        if (lambdaInfo.methodReference().getTag() == Opcodes.H_NEWINVOKESPECIAL
            && lambdaInfo.bridgeMethod().getTag() != Opcodes.H_NEWINVOKESPECIAL) {
          // We're changing a constructor call to a factory method call, so we unfortunately need
          // to go find the NEW/DUP pair preceding the constructor call and remove it
          removeLastAllocation();
        }
        super.visitMethodInsn(
            LambdaDesugaring.invokeOpcode(lambdaInfo.bridgeMethod()),
            lambdaInfo.bridgeMethod().getOwner(),
            lambdaInfo.bridgeMethod().getName(),
            lambdaInfo.bridgeMethod().getDesc(),
            lambdaInfo.bridgeMethod().isInterface());

      } else {
        super.visitMethodInsn(opcode, owner, name, desc, itf);
      }
    }

    private void removeLastAllocation() {
      AbstractInsnNode insn = instructions.getLast();
      while (insn != null && insn.getPrevious() != null) {
        AbstractInsnNode prev = insn.getPrevious();
        if (prev.getOpcode() == Opcodes.NEW && insn.getOpcode() == Opcodes.DUP
            && ((TypeInsnNode) prev).desc.equals(lambdaInfo.methodReference().getOwner())) {
          instructions.remove(prev);
          instructions.remove(insn);
          return;
        }
        insn = prev;
      }
      throw new IllegalStateException(
          "Couldn't find allocation to rewrite ::new reference " + lambdaInfo.methodReference());
    }

    @Override
    public void visitEnd() {
      accept(dest);
    }
  }
}
