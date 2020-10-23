// Copyright 2017 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;
import static org.objectweb.asm.Opcodes.ACC_STATIC;
import static org.objectweb.asm.Opcodes.ACC_SYNTHETIC;
import static org.objectweb.asm.Opcodes.INVOKEINTERFACE;
import static org.objectweb.asm.Opcodes.INVOKESTATIC;
import static org.objectweb.asm.Opcodes.INVOKEVIRTUAL;

import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.collect.FluentIterable;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.android.desugar.BytecodeTypeInference.InferredType;
import com.google.devtools.build.android.desugar.io.BitFlags;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import javax.annotation.Nullable;
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.Label;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.commons.ClassRemapper;
import org.objectweb.asm.commons.Remapper;
import org.objectweb.asm.tree.MethodNode;

/**
 * Desugar try-with-resources. This class visitor intercepts calls to the following methods, and
 * redirect them to ThrowableExtension.
 * <li>{@code Throwable.addSuppressed(Throwable)}
 * <li>{@code Throwable.getSuppressed()}
 * <li>{@code Throwable.printStackTrace()}
 * <li>{@code Throwable.printStackTrace(PrintStream)}
 * <li>{@code Throwable.printStackTrace(PringWriter)}
 */
public class TryWithResourcesRewriter extends ClassVisitor {

  private static final String RUNTIME_PACKAGE_INTERNAL_NAME =
      "com/google/devtools/build/android/desugar/runtime";

  static final String THROWABLE_EXTENSION_INTERNAL_NAME =
      RUNTIME_PACKAGE_INTERNAL_NAME + '/' + "ThrowableExtension";

  /** The extension classes for java.lang.Throwable. */
  static final ImmutableSet<String> THROWABLE_EXT_CLASS_INTERNAL_NAMES =
      ImmutableSet.of(
          THROWABLE_EXTENSION_INTERNAL_NAME,
          THROWABLE_EXTENSION_INTERNAL_NAME + "$AbstractDesugaringStrategy",
          THROWABLE_EXTENSION_INTERNAL_NAME + "$ConcurrentWeakIdentityHashMap",
          THROWABLE_EXTENSION_INTERNAL_NAME + "$ConcurrentWeakIdentityHashMap$WeakKey",
          THROWABLE_EXTENSION_INTERNAL_NAME + "$MimicDesugaringStrategy",
          THROWABLE_EXTENSION_INTERNAL_NAME + "$NullDesugaringStrategy",
          THROWABLE_EXTENSION_INTERNAL_NAME + "$ReuseDesugaringStrategy");

  /** The extension classes for java.lang.Throwable. All the names end with ".class" */
  static final ImmutableSet<String> THROWABLE_EXT_CLASS_INTERNAL_NAMES_WITH_CLASS_EXT =
      FluentIterable.from(THROWABLE_EXT_CLASS_INTERNAL_NAMES)
          .transform(
              new Function<String, String>() {
                @Override
                public String apply(String s) {
                  return s + ".class";
                }
              })
          .toSet();

  static final ImmutableMultimap<String, String> TARGET_METHODS =
      ImmutableMultimap.<String, String>builder()
          .put("addSuppressed", "(Ljava/lang/Throwable;)V")
          .put("getSuppressed", "()[Ljava/lang/Throwable;")
          .put("printStackTrace", "()V")
          .put("printStackTrace", "(Ljava/io/PrintStream;)V")
          .put("printStackTrace", "(Ljava/io/PrintWriter;)V")
          .build();

  static final ImmutableMap<String, String> METHOD_DESC_MAP =
      ImmutableMap.<String, String>builder()
          .put("(Ljava/lang/Throwable;)V", "(Ljava/lang/Throwable;Ljava/lang/Throwable;)V")
          .put("()[Ljava/lang/Throwable;", "(Ljava/lang/Throwable;)[Ljava/lang/Throwable;")
          .put("()V", "(Ljava/lang/Throwable;)V")
          .put("(Ljava/io/PrintStream;)V", "(Ljava/lang/Throwable;Ljava/io/PrintStream;)V")
          .put("(Ljava/io/PrintWriter;)V", "(Ljava/lang/Throwable;Ljava/io/PrintWriter;)V")
          .build();

  static final String CLOSE_RESOURCE_METHOD_NAME = "$closeResource";
  static final String CLOSE_RESOURCE_METHOD_DESC =
      "(Ljava/lang/Throwable;Ljava/lang/AutoCloseable;)V";

  private final ClassLoader classLoader;
  private final Set<String> visitedExceptionTypes;
  private final AtomicInteger numOfTryWithResourcesInvoked;
  /** Stores the internal class names of resources that need to be closed. */
  private final LinkedHashSet<String> resourceTypeInternalNames = new LinkedHashSet<>();

  private final boolean hasCloseResourceMethod;

  private String internalName;
  /**
   * Indicate whether the current class being desugared should be ignored. If the current class is
   * one of the runtime extension classes, then it should be ignored.
   */
  private boolean shouldCurrentClassBeIgnored;
  /**
   * A method node for $closeResource(Throwable, AutoCloseable). At then end, we specialize this
   * method node.
   */
  @Nullable private MethodNode closeResourceMethod;

  public TryWithResourcesRewriter(
      ClassVisitor classVisitor,
      ClassLoader classLoader,
      Set<String> visitedExceptionTypes,
      AtomicInteger numOfTryWithResourcesInvoked,
      boolean hasCloseResourceMethod) {
    super(Opcodes.ASM8, classVisitor);
    this.classLoader = classLoader;
    this.visitedExceptionTypes = visitedExceptionTypes;
    this.numOfTryWithResourcesInvoked = numOfTryWithResourcesInvoked;
    this.hasCloseResourceMethod = hasCloseResourceMethod;
  }

  @Override
  public void visit(
      int version,
      int access,
      String name,
      String signature,
      String superName,
      String[] interfaces) {
    super.visit(version, access, name, signature, superName, interfaces);
    internalName = name;
    shouldCurrentClassBeIgnored = THROWABLE_EXT_CLASS_INTERNAL_NAMES.contains(name);
    Preconditions.checkState(
        !shouldCurrentClassBeIgnored || !hasCloseResourceMethod,
        "The current class which will be ignored "
            + "contains $closeResource(Throwable, AutoCloseable).");
  }

  @Override
  public void visitEnd() {
    if (!resourceTypeInternalNames.isEmpty()) {
      checkNotNull(closeResourceMethod);
      for (String resourceInternalName : resourceTypeInternalNames) {
        boolean isInterface = isInterface(resourceInternalName.replace('/', '.'));
        // We use "this" to desugar the body of the close resource method.
        closeResourceMethod.accept(
            new CloseResourceMethodSpecializer(cv, resourceInternalName, isInterface));
      }
    } else {
      // It is possible that all calls to $closeResources(...) are in dead code regions, and the
      // calls are eliminated, which leaving the method $closeResources() unused. (b/78030676).
      // In this case, we just discard the method body.
      checkState(
          !hasCloseResourceMethod || closeResourceMethod != null,
          "There should be $closeResources(...) in the class file.");
    }
    super.visitEnd();
  }

  @Override
  public MethodVisitor visitMethod(
      int access, String name, String desc, String signature, String[] exceptions) {
    if (exceptions != null && exceptions.length > 0) {
      // collect exception types.
      Collections.addAll(visitedExceptionTypes, exceptions);
    }
    if (isSyntheticCloseResourceMethod(access, name, desc)) {
      checkState(closeResourceMethod == null, "The TWR rewriter has been used.");
      closeResourceMethod = new MethodNode(Opcodes.ASM8, access, name, desc, signature, exceptions);
      // Run the TWR desugar pass over the $closeResource(Throwable, AutoCloseable) first, for
      // example, to rewrite calls to AutoCloseable.close()..
      TryWithResourceVisitor twrVisitor =
          new TryWithResourceVisitor(
              internalName, name + desc, closeResourceMethod, classLoader, null);
      return twrVisitor;
    }

    MethodVisitor visitor = super.cv.visitMethod(access, name, desc, signature, exceptions);
    if (visitor == null || shouldCurrentClassBeIgnored) {
      return visitor;
    }

    BytecodeTypeInference inference = null;
    if (hasCloseResourceMethod) {
      /*
       * BytecodeTypeInference will run after the TryWithResourceVisitor, because when we are
       * processing a bytecode instruction, we need to know the types in the operand stack, which
       * are inferred after the previous instruction.
       */
      inference = new BytecodeTypeInference(access, internalName, name, desc);
      inference.setDelegateMethodVisitor(visitor);
      visitor = inference;
    }

    TryWithResourceVisitor twrVisitor =
        new TryWithResourceVisitor(internalName, name + desc, visitor, classLoader, inference);
    return twrVisitor;
  }

  public static boolean isSyntheticCloseResourceMethod(int access, String name, String desc) {
    return BitFlags.isSet(access, ACC_SYNTHETIC | ACC_STATIC)
        && CLOSE_RESOURCE_METHOD_NAME.equals(name)
        && CLOSE_RESOURCE_METHOD_DESC.equals(desc);
  }

  private boolean isInterface(String className) {
    // A generated class from desugaring a lambda expression or member reference isn't an interface.
    if (isDesugaredLambdaClass(className)) {
      return false;
    }
    try {
      Class<?> klass = classLoader.loadClass(className);
      return klass.isInterface();
    } catch (ClassNotFoundException e) {
      throw new AssertionError("Failed to load class when desugaring class " + internalName);
    }
  }

  public static boolean isCallToSyntheticCloseResource(
      String currentClassInternalName, int opcode, String owner, String name, String desc) {
    if (opcode != INVOKESTATIC) {
      return false;
    }
    if (!currentClassInternalName.equals(owner)) {
      return false;
    }
    if (!CLOSE_RESOURCE_METHOD_NAME.equals(name)) {
      return false;
    }
    if (!CLOSE_RESOURCE_METHOD_DESC.equals(desc)) {
      return false;
    }
    return true;
  }

  private class TryWithResourceVisitor extends MethodVisitor {

    private final ClassLoader classLoader;
    /** For debugging purpose. Enrich exception information. */
    private final String internalName;

    private final String methodSignature;
    @Nullable private final BytecodeTypeInference typeInference;

    public TryWithResourceVisitor(
        String internalName,
        String methodSignature,
        MethodVisitor methodVisitor,
        ClassLoader classLoader,
        @Nullable BytecodeTypeInference typeInference) {
      super(Opcodes.ASM8, methodVisitor);
      this.classLoader = classLoader;
      this.internalName = internalName;
      this.methodSignature = methodSignature;
      this.typeInference = typeInference;
    }

    @Override
    public void visitTryCatchBlock(Label start, Label end, Label handler, String type) {
      if (type != null) {
        visitedExceptionTypes.add(type); // type in a try-catch block must extend Throwable.
      }
      super.visitTryCatchBlock(start, end, handler, type);
    }

    @Override
    public void visitMethodInsn(int opcode, String owner, String name, String desc, boolean itf) {
      if (isCallToSyntheticCloseResource(internalName, opcode, owner, name, desc)) {
        checkNotNull(
            typeInference,
            "This method %s.%s has a call to $closeResource(Throwable, AutoCloseable) method, "
                + "but the type inference is null.",
            internalName,
            methodSignature);
        {
          // Check the exception type.
          InferredType exceptionClass = typeInference.getTypeOfOperandFromTop(1);
          if (!exceptionClass.isNull()) {
            Optional<String> exceptionClassInternalName = exceptionClass.getInternalName();
            checkState(
                exceptionClassInternalName.isPresent(),
                "The exception %s is not a reference type in %s.%s",
                exceptionClass,
                internalName,
                methodSignature);
            checkState(
                isAssignableFrom(
                    "java.lang.Throwable", exceptionClassInternalName.get().replace('/', '.')),
                "The exception type %s in %s.%s should be a subclass of java.lang.Throwable.",
                exceptionClassInternalName,
                internalName,
                methodSignature);
          }
        }

        InferredType resourceType = typeInference.getTypeOfOperandFromTop(0);
        Optional<String> resourceClassInternalName = resourceType.getInternalName();
        {
          // Check the resource type.
          checkState(
              resourceClassInternalName.isPresent(),
              "The resource class %s is not a reference type in %s.%s",
              resourceType,
              internalName,
              methodSignature);
          String resourceClassName = resourceClassInternalName.get().replace('/', '.');
          checkState(
              // For a resource class initialized from a lambda expression or an member reference,
              // it can implicitly be resolved with a close method.
              isDesugaredLambdaClass(resourceClassName) || hasCloseMethod(resourceClassName),
              "The resource class %s should have a close() method.",
              resourceClassName);
        }
        resourceTypeInternalNames.add(resourceClassInternalName.get());
        super.visitMethodInsn(
            opcode,
            owner,
            "$closeResource",
            "(Ljava/lang/Throwable;L" + resourceClassInternalName.get() + ";)V",
            itf);
        return;
      }

      if (!isMethodCallTargeted(opcode, owner, name, desc)) {
        super.visitMethodInsn(opcode, owner, name, desc, itf);
        return;
      }
      numOfTryWithResourcesInvoked.incrementAndGet();
      visitedExceptionTypes.add(checkNotNull(owner)); // owner extends Throwable.
      super.visitMethodInsn(
          INVOKESTATIC, THROWABLE_EXTENSION_INTERNAL_NAME, name, METHOD_DESC_MAP.get(desc), false);
    }

    private boolean isMethodCallTargeted(int opcode, String owner, String name, String desc) {
      if (opcode != INVOKEVIRTUAL) {
        return false;
      }
      if (!TARGET_METHODS.containsEntry(name, desc)) {
        return false;
      }
      if (visitedExceptionTypes.contains(owner)) {
        return true; // The owner is an exception that has been visited before.
      }
      return isAssignableFrom("java.lang.Throwable", owner.replace('/', '.'));
    }

    private boolean hasCloseMethod(String resourceClassName) {
      try {
        Class<?> klass = classLoader.loadClass(resourceClassName);
        klass.getMethod("close");
        return true;
      } catch (ClassNotFoundException e) {
        throw new AssertionError(
            "Failed to load class "
                + resourceClassName
                + " when desugaring method "
                + internalName
                + "."
                + methodSignature,
            e);
      } catch (NoSuchMethodException e) {
        // There is no close() method in the class, so return false.
        return false;
      }
    }

    private boolean isAssignableFrom(String baseClassName, String subClassName) {
      try {
        Class<?> baseClass = classLoader.loadClass(baseClassName);
        Class<?> subClass = classLoader.loadClass(subClassName);
        return baseClass.isAssignableFrom(subClass);
      } catch (ClassNotFoundException e) {
        throw new AssertionError(
            "Failed to load class when desugaring method "
                + internalName
                + "."
                + methodSignature
                + " when checking the assignable relation for class "
                + baseClassName
                + " and "
                + subClassName,
            e);
      }
    }
  }

  /**
   * A class to specialize the method $closeResource(Throwable, AutoCloseable), which does
   *
   * <ul>
   *   <li>Rename AutoCloseable to the given concrete resource type.
   *   <li>Adjust the invoke instruction that calls AutoCloseable.close()
   * </ul>
   */
  private static class CloseResourceMethodSpecializer extends ClassRemapper {

    private final boolean isResourceAnInterface;
    private final String targetResourceInternalName;

    public CloseResourceMethodSpecializer(
        ClassVisitor cv, String targetResourceInternalName, boolean isResourceAnInterface) {
      super(
          cv,
          new Remapper() {
            @Override
            public String map(String typeName) {
              if (typeName.equals("java/lang/AutoCloseable")) {
                return targetResourceInternalName;
              } else {
                return typeName;
              }
            }
          });
      this.targetResourceInternalName = targetResourceInternalName;
      this.isResourceAnInterface = isResourceAnInterface;
    }

    @Override
    public MethodVisitor visitMethod(
        int access, String name, String desc, String signature, String[] exceptions) {
      MethodVisitor mv = super.visitMethod(access, name, desc, signature, exceptions);
      return new MethodVisitor(Opcodes.ASM8, mv) {
        @Override
        public void visitMethodInsn(
            int opcode, String owner, String name, String desc, boolean itf) {
          if (opcode == INVOKEINTERFACE
              && owner.endsWith("java/lang/AutoCloseable")
              && name.equals("close")
              && desc.equals("()V")
              && itf) {
            opcode = isResourceAnInterface ? INVOKEINTERFACE : INVOKEVIRTUAL;
            owner = targetResourceInternalName;
            itf = isResourceAnInterface;
          }
          super.visitMethodInsn(opcode, owner, name, desc, itf);
        }
      };
    }
  }

  private static boolean isDesugaredLambdaClass(String qualifiedClassName) {
    return qualifiedClassName.contains("$$Lambda$");
  }
}
