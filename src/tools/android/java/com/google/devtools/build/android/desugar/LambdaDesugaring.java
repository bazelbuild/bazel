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
import static java.lang.invoke.MethodHandles.publicLookup;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.android.desugar.io.BitFlags;
import java.io.IOException;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles.Lookup;
import java.lang.invoke.MethodType;
import java.lang.reflect.Constructor;
import java.lang.reflect.Executable;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import javax.annotation.Nullable;
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.Handle;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.Type;
import org.objectweb.asm.tree.AbstractInsnNode;
import org.objectweb.asm.tree.FieldInsnNode;
import org.objectweb.asm.tree.InsnNode;
import org.objectweb.asm.tree.MethodNode;
import org.objectweb.asm.tree.TypeInsnNode;

/**
 * Visitor that desugars classes with uses of lambdas into Java 7-looking code. This includes
 * rewriting lambda-related invokedynamic instructions as well as fixing accessibility of methods
 * that javac emits for lambda bodies.
 *
 * <p>Implementation note: {@link InvokeDynamicLambdaMethodCollector} needs to detect any class that
 * this visitor may rewrite, as we conditionally apply this visitor based on it.
 */
class LambdaDesugaring extends ClassVisitor {

  private final ClassLoader targetLoader;
  private final LambdaClassMaker lambdas;
  private final ImmutableSet.Builder<String> aggregateInterfaceLambdaMethods;
  private final Map<Handle, MethodReferenceBridgeInfo> bridgeMethods = new LinkedHashMap<>();
  private final ImmutableSet<MethodInfo> lambdaMethodsUsedInInvokeDyanmic;
  private final boolean allowDefaultMethods;

  private String internalName;
  private boolean isInterface;
  private int lambdaCount;

  public LambdaDesugaring(
      ClassVisitor dest,
      ClassLoader targetLoader,
      LambdaClassMaker lambdas,
      ImmutableSet.Builder<String> aggregateInterfaceLambdaMethods,
      ImmutableSet<MethodInfo> lambdaMethodsUsedInInvokeDyanmic,
      boolean allowDefaultMethods) {
    super(Opcodes.ASM8, dest);
    this.targetLoader = targetLoader;
    this.lambdas = lambdas;
    this.aggregateInterfaceLambdaMethods = aggregateInterfaceLambdaMethods;
    this.lambdaMethodsUsedInInvokeDyanmic = lambdaMethodsUsedInInvokeDyanmic;
    this.allowDefaultMethods = allowDefaultMethods;
  }

  @Override
  public void visit(
      int version,
      int access,
      String name,
      String signature,
      String superName,
      String[] interfaces) {
    checkState(internalName == null, "not intended for reuse but reused for %s", name);
    internalName = name;
    isInterface = BitFlags.isSet(access, Opcodes.ACC_INTERFACE);
    super.visit(version, access, name, signature, superName, interfaces);
  }

  @Override
  public void visitEnd() {
    for (Map.Entry<Handle, MethodReferenceBridgeInfo> bridge : bridgeMethods.entrySet()) {
      Handle original = bridge.getKey();
      Handle neededMethod = bridge.getValue().bridgeMethod();
      checkState(
          neededMethod.getTag() == Opcodes.H_INVOKESTATIC
              || neededMethod.getTag() == Opcodes.H_INVOKEVIRTUAL,
          "Cannot generate bridge method %s to reach %s",
          neededMethod,
          original);
      checkState(
          bridge.getValue().referenced() != null,
          "Need referenced method %s to generate bridge %s",
          original,
          neededMethod);

      int access = Opcodes.ACC_BRIDGE | Opcodes.ACC_SYNTHETIC | Opcodes.ACC_FINAL;
      if (neededMethod.getTag() == Opcodes.H_INVOKESTATIC) {
        access |= Opcodes.ACC_STATIC;
      }
      MethodVisitor bridgeMethod =
          super.visitMethod(
              access,
              neededMethod.getName(),
              neededMethod.getDesc(),
              (String) null,
              toInternalNames(bridge.getValue().referenced().getExceptionTypes()));

      // Bridge is a factory method calling a constructor
      if (original.getTag() == Opcodes.H_NEWINVOKESPECIAL) {
        bridgeMethod.visitTypeInsn(Opcodes.NEW, original.getOwner());
        bridgeMethod.visitInsn(Opcodes.DUP);
      }

      int slot = 0;
      if (neededMethod.getTag() != Opcodes.H_INVOKESTATIC) {
        bridgeMethod.visitVarInsn(Opcodes.ALOAD, slot++);
      }
      Type neededType = Type.getMethodType(neededMethod.getDesc());
      for (Type arg : neededType.getArgumentTypes()) {
        bridgeMethod.visitVarInsn(arg.getOpcode(Opcodes.ILOAD), slot);
        slot += arg.getSize();
      }
      bridgeMethod.visitMethodInsn(
          invokeOpcode(original),
          original.getOwner(),
          original.getName(),
          original.getDesc(),
          original.isInterface());
      bridgeMethod.visitInsn(neededType.getReturnType().getOpcode(Opcodes.IRETURN));

      bridgeMethod.visitMaxs(0, 0); // rely on class writer to compute these
      bridgeMethod.visitEnd();
    }
    super.visitEnd();
  }

  // If this method changes then InvokeDynamicLambdaMethodCollector may need changes well
  @Override
  public MethodVisitor visitMethod(
      int access, String name, String desc, String signature, String[] exceptions) {
    if (name.equals("$deserializeLambda$") && BitFlags.isSet(access, Opcodes.ACC_SYNTHETIC)) {
      // Android doesn't do anything special for lambda serialization so drop the special
      // deserialization hook that javac generates.  This also makes sure we don't reference
      // java/lang/invoke/SerializedLambda, which doesn't exist on Android.
      return null;
    }
    if (name.startsWith("lambda$")
        && BitFlags.isSet(access, Opcodes.ACC_SYNTHETIC)
        && lambdaMethodsUsedInInvokeDyanmic.contains(MethodInfo.create(internalName, name, desc))) {
      if (!allowDefaultMethods && isInterface && BitFlags.isSet(access, Opcodes.ACC_STATIC)) {
        // There must be a lambda in the interface (which in the absence of hand-written default or
        // static interface methods must mean it's in the <clinit> method or inside another lambda).
        // We'll move this method out of this class, so just record and drop it here.
        // (Note lambda body methods have unique names, so we don't need to remember desc here.)
        aggregateInterfaceLambdaMethods.add(internalName + '#' + name);
        return null;
      }
      if (BitFlags.isSet(access, Opcodes.ACC_PRIVATE)) {
        // Make lambda body method accessible from lambda class
        access &= ~Opcodes.ACC_PRIVATE;
        if (allowDefaultMethods && isInterface) {
          // java 8 requires interface methods to have exactly one of ACC_PUBLIC and ACC_PRIVATE
          access |= Opcodes.ACC_PUBLIC;
        } else {
          // Method was private so it can be final, which should help VMs perform dispatch.
          access |= Opcodes.ACC_FINAL;
        }
      }
      // Guarantee unique lambda body method name to avoid accidental overriding. This wouldn't be
      // be necessary for static methods but in visitOuterClass we don't know whether a potential
      // outer lambda$ method is static or not, so we just always do it.
      name = uniqueInPackage(internalName, name);
    }
    MethodVisitor dest = super.visitMethod(access, name, desc, signature, exceptions);
    return dest != null
        ? new InvokedynamicRewriter(dest, access, name, desc, signature, exceptions)
        : null;
  }

  // If this method changes then InvokeDynamicLambdaMethodCollector may need changes well
  @Override
  public void visitOuterClass(String owner, String name, String desc) {
    if (name != null && name.startsWith("lambda$")) {
      // Reflect renaming of lambda$ methods.  Proguard gets grumpy if we leave this inconsistent.
      name = uniqueInPackage(owner, name);
    }
    super.visitOuterClass(owner, name, desc);
  }

  // When adding visitXxx methods here then InvokeDynamicLambdaMethodCollector may need changes well

  static String uniqueInPackage(String owner, String name) {
    String suffix = "$" + owner.substring(owner.lastIndexOf('/') + 1);
    // For idempotency, we only attach the package-unique suffix if it isn't there already.  This
    // prevents a cumulative effect when processing a class more than once (which can happen with
    // Bazel, e.g., when re-importing a deploy.jar).  During reprocessing, invokedynamics are
    // already removed, so lambda$ methods have regular call sites that we would also have to re-
    // adjust if we just blindly appended something to lambda$ method names every time we see them.
    return name.endsWith(suffix) ? name : name + suffix;
  }

  /**
   * Makes {@link #visitEnd} generate a bridge method for the given method handle if the referenced
   * method will be invisible to the generated lambda class.
   *
   * @return struct containing either {@code invokedMethod} or {@code invokedMethod} and a handle
   *     representing the bridge method that will be generated for {@code invokedMethod}.
   */
  private MethodReferenceBridgeInfo queueUpBridgeMethodIfNeeded(Handle invokedMethod)
      throws ClassNotFoundException {
    if (invokedMethod.getName().startsWith("lambda$")) {
      // We adjust lambda bodies to be visible
      return MethodReferenceBridgeInfo.noBridge(invokedMethod);
    }

    // invokedMethod is a method reference if we get here
    Executable invoked = findTargetMethod(invokedMethod);
    if (isVisibleToLambdaClass(invoked, invokedMethod.getOwner())) {
      // Referenced method is visible to the generated class, so nothing to do
      return MethodReferenceBridgeInfo.noBridge(invokedMethod);
    }

    // We need a bridge method if we get here
    checkState(
        !isInterface,
        "%s is an interface and shouldn't need bridge to %s",
        internalName,
        invokedMethod);
    checkState(
        !invokedMethod.isInterface(),
        "%s's lambda classes can't see interface method: %s",
        internalName,
        invokedMethod);
    MethodReferenceBridgeInfo result = bridgeMethods.get(invokedMethod);
    if (result != null) {
      return result; // we're already queued up a bridge method for this method reference
    }

    String name = uniqueInPackage(internalName, "bridge$lambda$" + bridgeMethods.size());
    Handle bridgeMethod;
    switch (invokedMethod.getTag()) {
      case Opcodes.H_INVOKESTATIC:
        bridgeMethod =
            new Handle(
                invokedMethod.getTag(), internalName, name, invokedMethod.getDesc(), /*itf*/ false);
        break;
      case Opcodes.H_INVOKEVIRTUAL:
      case Opcodes.H_INVOKESPECIAL: // we end up calling these using invokevirtual
        bridgeMethod =
            new Handle(
                Opcodes.H_INVOKEVIRTUAL,
                internalName,
                name,
                invokedMethod.getDesc(), /*itf*/
                false);
        break;
      case Opcodes.H_NEWINVOKESPECIAL:
        {
          // Call invisible constructor through generated bridge "factory" method, so we need to
          // compute the descriptor for the bridge method from the constructor's descriptor
          String desc =
              Type.getMethodDescriptor(
                  Type.getObjectType(invokedMethod.getOwner()),
                  Type.getArgumentTypes(invokedMethod.getDesc()));
          bridgeMethod =
              new Handle(Opcodes.H_INVOKESTATIC, internalName, name, desc, /*itf*/ false);
          break;
        }
      case Opcodes.H_INVOKEINTERFACE:
        // Shouldn't get here
      default:
        throw new UnsupportedOperationException("Cannot bridge " + invokedMethod);
    }
    result = MethodReferenceBridgeInfo.bridge(invokedMethod, invoked, bridgeMethod);
    MethodReferenceBridgeInfo old = bridgeMethods.put(invokedMethod, result);
    checkState(old == null, "Already had bridge %s so we don't also want %s", old, result);
    return result;
  }

  /**
   * Checks whether the referenced method would be visible by an unrelated class in the same package
   * as the currently visited class.
   */
  private boolean isVisibleToLambdaClass(Executable invoked, String owner) {
    int modifiers = invoked.getModifiers();
    if (Modifier.isPrivate(modifiers)) {
      return false;
    }
    if (Modifier.isPublic(modifiers)) {
      return true;
    }
    // invoked is protected or package-private, either way we need it to be in the same package
    // because the additional visibility protected gives doesn't help lambda classes, which are in
    // a different class hierarchy (and typically just extend Object)
    return packageName(internalName).equals(packageName(owner));
  }

  private Executable findTargetMethod(Handle invokedMethod) throws ClassNotFoundException {
    Type descriptor = Type.getMethodType(invokedMethod.getDesc());
    Class<?> owner = loadFromInternal(invokedMethod.getOwner());
    if (invokedMethod.getTag() == Opcodes.H_NEWINVOKESPECIAL) {
      for (Constructor<?> c : owner.getDeclaredConstructors()) {
        if (Type.getType(c).equals(descriptor)) {
          return c;
        }
      }
    } else {
      for (Method m : owner.getDeclaredMethods()) {
        if (m.getName().equals(invokedMethod.getName()) && Type.getType(m).equals(descriptor)) {
          return m;
        }
      }
    }
    throw new IllegalArgumentException("Referenced method not found: " + invokedMethod);
  }

  private Class<?> loadFromInternal(String internalName) throws ClassNotFoundException {
    return targetLoader.loadClass(internalName.replace('/', '.'));
  }

  static int invokeOpcode(Handle invokedMethod) {
    switch (invokedMethod.getTag()) {
      case Opcodes.H_INVOKESTATIC:
        return Opcodes.INVOKESTATIC;
      case Opcodes.H_INVOKEVIRTUAL:
        return Opcodes.INVOKEVIRTUAL;
      case Opcodes.H_INVOKESPECIAL:
      case Opcodes.H_NEWINVOKESPECIAL: // Must be preceded by NEW
        return Opcodes.INVOKESPECIAL;
      case Opcodes.H_INVOKEINTERFACE:
        return Opcodes.INVOKEINTERFACE;
      default:
        throw new UnsupportedOperationException("Don't know how to call " + invokedMethod);
    }
  }

  private static String[] toInternalNames(Class<?>[] classes) {
    String[] result = new String[classes.length];
    for (int i = 0; i < classes.length; ++i) {
      result[i] = Type.getInternalName(classes[i]);
    }
    return result;
  }

  private static String packageName(String internalClassName) {
    int lastSlash = internalClassName.lastIndexOf('/');
    return lastSlash > 0 ? internalClassName.substring(0, lastSlash) : "";
  }

  /**
   * Desugaring that replaces invokedynamics for {@link java.lang.invoke.LambdaMetafactory} with
   * static factory method invocations and triggers a class to be generated for each invokedynamic.
   */
  private class InvokedynamicRewriter extends MethodNode {

    private final MethodVisitor dest;

    public InvokedynamicRewriter(
        MethodVisitor dest,
        int access,
        String name,
        String desc,
        String signature,
        String[] exceptions) {
      super(Opcodes.ASM8, access, name, desc, signature, exceptions);
      this.dest = checkNotNull(dest, "Null destination for %s.%s : %s", internalName, name, desc);
    }

    @Override
    public void visitEnd() {
      accept(dest);
    }

    @Override
    public void visitInvokeDynamicInsn(String name, String desc, Handle bsm, Object... bsmArgs) {
      if (!"java/lang/invoke/LambdaMetafactory".equals(bsm.getOwner())) {
        // Not an invokedynamic for a lambda expression
        super.visitInvokeDynamicInsn(name, desc, bsm, bsmArgs);
        return;
      }

      try {
        Lookup lookup = createLookup(internalName);
        ArrayList<Object> args = new ArrayList<>(bsmArgs.length + 3);
        args.add(lookup);
        args.add(name);
        args.add(MethodType.fromMethodDescriptorString(desc, targetLoader));
        for (Object bsmArg : bsmArgs) {
          args.add(toJvmMetatype(lookup, bsmArg));
        }

        // Both bootstrap methods in LambdaMetafactory expect a MethodHandle as their 5th argument
        // so we can assume bsmArgs[1] (the 5th arg) to be a Handle.
        MethodReferenceBridgeInfo bridgeInfo = queueUpBridgeMethodIfNeeded((Handle) bsmArgs[1]);

        // Resolve the bootstrap method in "host configuration" (this tool's default classloader)
        // since targetLoader may only contain stubs that we can't actually execute.
        // generateLambdaClass() below will invoke the bootstrap method, so a stub isn't enough,
        // and ultimately we don't care if the bootstrap method was even on the bootclasspath
        // when this class was compiled (although it must've been since javac is unhappy otherwise).
        MethodHandle bsmMethod = toMethodHandle(publicLookup(), bsm, /*target*/ false);
        // Give generated classes to have more stable names (b/35643761).  Use BSM's naming scheme
        // but with separate counter for each surrounding class.
        String lambdaClassName = internalName + "$$Lambda$" + (lambdaCount++);
        Type[] capturedTypes = Type.getArgumentTypes(desc);
        boolean needFactory =
            capturedTypes.length != 0
                && !attemptAllocationBeforeArgumentLoads(lambdaClassName, capturedTypes);
        lambdas.generateLambdaClass(
            internalName,
            LambdaInfo.create(
                lambdaClassName,
                desc,
                needFactory,
                bridgeInfo.methodReference(),
                bridgeInfo.bridgeMethod()),
            bsmMethod,
            args);
        if (desc.startsWith("()")) {
          // For stateless lambda classes we'll generate a singleton instance that we can just load
          checkState(capturedTypes.length == 0);
          super.visitFieldInsn(
              Opcodes.GETSTATIC,
              lambdaClassName,
              LambdaClassFixer.SINGLETON_FIELD_NAME,
              desc.substring("()".length()));
        } else if (needFactory) {
          // If we were unable to inline the allocation of the generated lambda class then
          // invoke factory method of generated lambda class with the arguments on the stack
          super.visitMethodInsn(
              Opcodes.INVOKESTATIC,
              lambdaClassName,
              LambdaClassFixer.FACTORY_METHOD_NAME,
              desc,
              /*itf*/ false);
        } else {
          // Otherwise we inserted a new/dup pair of instructions above and now just need to invoke
          // the constructor of generated lambda class with the arguments on the stack
          super.visitMethodInsn(
              Opcodes.INVOKESPECIAL,
              lambdaClassName,
              "<init>",
              Type.getMethodDescriptor(Type.VOID_TYPE, capturedTypes),
              /*itf*/ false);
        }
      } catch (IOException | ReflectiveOperationException e) {
        throw new IllegalStateException(
            "Couldn't desugar invokedynamic for "
                + internalName
                + "."
                + name
                + " using "
                + bsm
                + " with arguments "
                + Arrays.toString(bsmArgs),
            e);
      }
    }

    /**
     * Tries to insert a new/dup for the given class name before expected existing instructions that
     * set up arguments for an invokedynamic factory method with the given types.
     *
     * <p>For lambda expressions and simple method references we can assume that arguments are set
     * up with loads of the captured (effectively) final variables. But method references, can in
     * general capture an expression, such as in {@code myObject.toString()::charAt} (a {@code
     * Function&lt;Integer, Character&gt;}), which can also cause null checks to be inserted. In
     * such more complicated cases this method may fail to insert a new/dup pair and returns {@code
     * false}.
     *
     * @param internalName internal name of the class to instantiate
     * @param paramTypes expected invokedynamic argument types, which also must be the parameters of
     *     {@code internalName}'s constructor.
     * @return {@code true} if we were able to insert a new/dup, {@code false} otherwise
     */
    private boolean attemptAllocationBeforeArgumentLoads(String internalName, Type[] paramTypes) {
      checkArgument(paramTypes.length > 0, "Expected at least one param for %s", internalName);
      // Walk backwards past loads corresponding to constructor arguments to find the instruction
      // after which we need to insert our NEW/DUP pair
      AbstractInsnNode insn = instructions.getLast();
      for (int i = paramTypes.length - 1; 0 <= i; --i) {
        if (insn.getOpcode() == Opcodes.GETFIELD) {
          // Lambdas in anonymous inner classes have to load outer scope variables from fields,
          // which manifest as an ALOAD followed by one or more GETFIELDs
          FieldInsnNode getfield = (FieldInsnNode) insn;
          checkState(
              getfield.desc.length() == 1
                  ? getfield.desc.equals(paramTypes[i].getDescriptor())
                  : paramTypes[i].getDescriptor().length() > 1,
              "Expected getfield for %s to set up parameter %s for %s but got %s : %s",
              paramTypes[i],
              i,
              internalName,
              getfield.name,
              getfield.desc);
          insn = insn.getPrevious();

          while (insn.getOpcode() == Opcodes.GETFIELD) {
            // Nested inner classes can cause a cascade of getfields from the outermost one inwards
            checkState(
                ((FieldInsnNode) insn).desc.startsWith("L"),
                "expect object type getfields to get to %s to set up parameter %s for %s, not: %s",
                paramTypes[i],
                i,
                internalName,
                ((FieldInsnNode) insn).desc);
            insn = insn.getPrevious();
          }

          checkState(
              insn.getOpcode() == Opcodes.ALOAD, // should be a this pointer to be precise
              "Expected aload before getfield for %s to set up parameter %s for %s but got %s",
              getfield.name,
              i,
              internalName,
              insn.getOpcode());
        } else if (!isPushForType(insn, paramTypes[i])) {
          // Otherwise expect load of a (effectively) final local variable or a constant. Not seeing
          // that means we're dealing with a method reference on some arbitrary expression,
          // <expression>::m. In that case we give up and keep using the factory method for now,
          // since inserting the NEW/DUP so the new object ends up in the right stack slot is hard
          // in that case. Note this still covers simple cases such as this::m or x::m, where x is a
          // local.
          checkState(
              paramTypes.length == 1,
              "Expected a load for %s to set up parameter %s for %s but got %s",
              paramTypes[i],
              i,
              internalName,
              insn.getOpcode());
          return false;
        }
        insn = insn.getPrevious();
      }

      TypeInsnNode newInsn = new TypeInsnNode(Opcodes.NEW, internalName);
      if (insn == null) {
        // Ran off the front of the instruction list
        instructions.insert(newInsn);
      } else {
        instructions.insert(insn, newInsn);
      }
      instructions.insert(newInsn, new InsnNode(Opcodes.DUP));
      return true;
    }

    /**
     * Returns whether a given instruction can be used to push argument of {@code type} on stack.
     */
    private /* static */ boolean isPushForType(AbstractInsnNode insn, Type type) {
      int opcode = insn.getOpcode();
      if (opcode == type.getOpcode(Opcodes.ILOAD)) {
        return true;
      }
      // b/62060793: AsyncAwait rewrites bytecode to convert java methods into state machine with
      // support of lambdas. Constant zero values are pushed on stack for all yet uninitialized
      // local variables. And SIPUSH instruction is used to advance an internal state of a state
      // machine.
      switch (type.getSort()) {
        case Type.BOOLEAN:
          return opcode == Opcodes.ICONST_0 || opcode == Opcodes.ICONST_1;

        case Type.BYTE:
        case Type.CHAR:
        case Type.SHORT:
        case Type.INT:
          return opcode == Opcodes.SIPUSH
              || opcode == Opcodes.ICONST_0
              || opcode == Opcodes.ICONST_1
              || opcode == Opcodes.ICONST_2
              || opcode == Opcodes.ICONST_3
              || opcode == Opcodes.ICONST_4
              || opcode == Opcodes.ICONST_5
              || opcode == Opcodes.ICONST_M1;

        case Type.LONG:
          return opcode == Opcodes.LCONST_0 || opcode == Opcodes.LCONST_1;

        case Type.FLOAT:
          return opcode == Opcodes.FCONST_0
              || opcode == Opcodes.FCONST_1
              || opcode == Opcodes.FCONST_2;

        case Type.DOUBLE:
          return opcode == Opcodes.DCONST_0 || opcode == Opcodes.DCONST_1;

        case Type.OBJECT:
        case Type.ARRAY:
          return opcode == Opcodes.ACONST_NULL;

        default:
          // Support for BIPUSH and LDC* opcodes is not implemented as there is no known use case.
          return false;
      }
    }

    private Lookup createLookup(String lookupClass) throws ReflectiveOperationException {
      Class<?> clazz = loadFromInternal(lookupClass);
      Constructor<Lookup> constructor = Lookup.class.getDeclaredConstructor(Class.class);
      constructor.setAccessible(true);
      return constructor.newInstance(clazz);
    }

    /**
     * Produces a {@link MethodHandle} or {@link MethodType} using {@link #targetLoader} for the
     * given ASM {@link Handle} or {@link Type}. {@code lookup} is only used for resolving {@link
     * Handle}s.
     */
    private Object toJvmMetatype(Lookup lookup, Object asm) throws ReflectiveOperationException {
      if (asm instanceof Number) {
        return asm;
      }
      if (asm instanceof Type) {
        Type type = (Type) asm;
        switch (type.getSort()) {
          case Type.OBJECT:
            return loadFromInternal(type.getInternalName());
          case Type.METHOD:
            return MethodType.fromMethodDescriptorString(type.getDescriptor(), targetLoader);
          default:
            throw new IllegalArgumentException("Cannot convert: " + asm);
        }
      }
      if (asm instanceof Handle) {
        return toMethodHandle(lookup, (Handle) asm, /*target*/ true);
      }
      throw new IllegalArgumentException("Cannot convert: " + asm);
    }

    /**
     * Produces a {@link MethodHandle} using either the context or {@link #targetLoader} class
     * loader, depending on {@code target}.
     */
    private MethodHandle toMethodHandle(Lookup lookup, Handle asmHandle, boolean target)
        throws ReflectiveOperationException {
      Class<?> owner = loadFromInternal(asmHandle.getOwner());
      MethodType signature =
          MethodType.fromMethodDescriptorString(
              asmHandle.getDesc(),
              target ? targetLoader : Thread.currentThread().getContextClassLoader());
      switch (asmHandle.getTag()) {
        case Opcodes.H_INVOKESTATIC:
          return lookup.findStatic(owner, asmHandle.getName(), signature);
        case Opcodes.H_INVOKEVIRTUAL:
        case Opcodes.H_INVOKEINTERFACE:
          return lookup.findVirtual(owner, asmHandle.getName(), signature);
        case Opcodes.H_INVOKESPECIAL: // we end up calling these using invokevirtual
          return lookup.findSpecial(owner, asmHandle.getName(), signature, owner);
        case Opcodes.H_NEWINVOKESPECIAL:
          return lookup.findConstructor(owner, signature);
        default:
          throw new UnsupportedOperationException("Cannot resolve " + asmHandle);
      }
    }
  }

  /**
   * Record of how a lambda class can reach its referenced method through a possibly-different
   * bridge method.
   *
   * <p>In a JVM, lambda classes are allowed to call the referenced methods directly, but we don't
   * have that luxury when the generated lambda class is evaluated using normal visibility rules.
   */
  @AutoValue
  abstract static class MethodReferenceBridgeInfo {
    public static MethodReferenceBridgeInfo noBridge(Handle methodReference) {
      return new AutoValue_LambdaDesugaring_MethodReferenceBridgeInfo(
          methodReference, (Executable) null, methodReference);
    }

    public static MethodReferenceBridgeInfo bridge(
        Handle methodReference, Executable referenced, Handle bridgeMethod) {
      checkArgument(!bridgeMethod.equals(methodReference));
      return new AutoValue_LambdaDesugaring_MethodReferenceBridgeInfo(
          methodReference, checkNotNull(referenced), bridgeMethod);
    }

    public abstract Handle methodReference();

    /** Returns {@code null} iff {@link #bridgeMethod} equals {@link #methodReference}. */
    @Nullable
    public abstract Executable referenced();

    public abstract Handle bridgeMethod();
  }
}
