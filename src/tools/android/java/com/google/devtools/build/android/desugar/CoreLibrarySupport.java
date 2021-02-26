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
package com.google.devtools.build.android.desugar;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;
import static java.util.Collections.unmodifiableSet;
import static java.util.stream.Stream.concat;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.LinkedHashMultimap;
import com.google.common.collect.Multimap;
import com.google.devtools.build.android.desugar.io.BitFlags;
import com.google.devtools.build.android.desugar.io.CoreLibraryRewriter;
import com.google.devtools.build.android.desugar.langmodel.ClassName;
import com.google.devtools.build.android.desugar.langmodel.MemberUseKind;
import com.google.devtools.build.android.desugar.langmodel.MethodInvocationSite;
import com.google.devtools.build.android.desugar.langmodel.MethodKey;
import com.google.devtools.build.android.desugar.retarget.ClassMemberRetargetConfig;
import com.google.errorprone.annotations.Immutable;
import java.lang.reflect.Method;
import java.util.Collection;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import javax.annotation.Nullable;
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.Label;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.Type;
import org.objectweb.asm.commons.Remapper;

/** Helper that keeps track of which core library classes and methods we want to rewrite. */
class CoreLibrarySupport {

  private static final Object[] EMPTY_FRAME = new Object[0];
  private static final String[] EMPTY_LIST = new String[0];

  private final CoreLibraryRewriter rewriter;
  private final ClassLoader targetLoader;
  /** Internal name prefixes that we want to move to a custom package. */
  private final ImmutableSet<String> renamedPrefixes;

  private final ImmutableSet<String> excludeFromEmulation;
  /** Internal names of interfaces whose default and static interface methods we'll emulate. */
  private final ImmutableSet<Class<?>> emulatedInterfaces;

  private final ClassMemberRetargetConfig retargetConfig;

  /** ASM {@link Remapper} based on {@link #renamedPrefixes}. */
  private final Remapper corePackageRemapper =
      new Remapper() {
        @Override
        public String map(String typeName) {
          return isRenamedCoreLibrary(typeName) ? renameCoreLibrary(typeName) : typeName;
        }
      };

  /** For the collection of definitions of emulated default methods (deterministic iteration). */
  private final Multimap<String, EmulatedMethod> emulatedDefaultMethods =
      LinkedHashMultimap.create();
  /** Collect targets queried in {@link #getMoveTarget}. */
  private final Set<String> usedRuntimeHelpers = new LinkedHashSet<>();

  public CoreLibrarySupport(
      CoreLibraryRewriter rewriter,
      ClassLoader targetLoader,
      List<String> renamedPrefixes,
      List<String> emulatedInterfaces,
      List<String> excludeFromEmulation,
      ClassMemberRetargetConfig retargetConfig) {
    this.rewriter = rewriter;
    this.targetLoader = targetLoader;
    this.retargetConfig = retargetConfig;
    checkArgument(
        renamedPrefixes.stream()
            .allMatch(prefix -> prefix.startsWith("java/") || prefix.startsWith("javadesugar/")),
        "Unexpected renamedPrefixes: Actual (%s).",
        renamedPrefixes);
    this.renamedPrefixes = ImmutableSet.copyOf(renamedPrefixes);
    this.excludeFromEmulation = ImmutableSet.copyOf(excludeFromEmulation);

    ImmutableSet.Builder<Class<?>> classBuilder = ImmutableSet.builder();
    for (String itf : emulatedInterfaces) {
      checkArgument(itf.startsWith("java/util/"), itf);
      Class<?> clazz = loadFromInternal(rewriter.getPrefix() + itf);
      checkArgument(clazz.isInterface(), itf);
      classBuilder.add(clazz);
    }
    this.emulatedInterfaces = classBuilder.build();
  }

  public boolean isRenamedCoreLibrary(String internalName) {
    String unprefixedName = rewriter.unprefix(internalName);
    if (!(unprefixedName.startsWith("java/") || unprefixedName.startsWith("javadesugar/"))
        || renamedPrefixes.isEmpty()) {
      return false; // shortcut
    }
    // Rename any classes desugar might generate under java/ (for emulated interfaces) as well as
    // configured prefixes
    return looksGenerated(unprefixedName)
        || renamedPrefixes.stream().anyMatch(unprefixedName::startsWith);
  }

  public String renameCoreLibrary(String internalName) {
    internalName = rewriter.unprefix(internalName);
    if (internalName.startsWith("java/")) {
      return "j$/" + internalName.substring(/* cut away "java/" prefix */ 5);
    }
    if (internalName.startsWith("javadesugar/")) {
      return "jd$/" + internalName.substring(/* cut away "javadesugar/" prefix */ 12);
    }

    return internalName;
  }

  public Remapper getRemapper() {
    return corePackageRemapper;
  }

  @Nullable
  public String getMoveTarget(String owner, String name, String desc) {
    final String result;
    final MethodKey methodKey =
        MethodKey.create(ClassName.create(owner), name, desc)
            .acceptTypeMapper(ClassName.IN_PROCESS_LABEL_STRIPPER);
    MethodInvocationSite replacementSite =
        retargetConfig.findReplacementSite(
            MethodInvocationSite.builder()
                .setInvocationKind(MemberUseKind.INVOKEVIRTUAL)
                .setIsInterface(false)
                .setMethod(methodKey.acceptTypeMapper(ClassName.IN_PROCESS_LABEL_STRIPPER))
                .build());
    if (replacementSite == null) {
      result = null;
    } else {
      MethodKey targetMethod =
          replacementSite.acceptTypeMapper(ClassName.SHADOWED_TO_MIRRORED_TYPE_MAPPER).method();
      result = targetMethod.ownerName();
    }

    if (result != null) {
      // Remember that we need the move target so we can include it in the output later
      usedRuntimeHelpers.add(result);
    }
    return result;
  }

  /**
   * Returns {@code true} for java.* classes or interfaces that are subtypes of emulated interfaces.
   * Note that implies that this method always returns {@code false} for user-written classes.
   */
  public boolean isEmulatedCoreClassOrInterface(String internalName) {
    return getEmulatedCoreClassOrInterface(internalName) != null;
  }

  /** Includes the given method definition in any applicable core interface emulation logic. */
  public void registerIfEmulatedCoreInterface(
      int access, String owner, String name, String desc, String[] exceptions) {
    Class<?> emulated = getEmulatedCoreClassOrInterface(owner);
    if (emulated == null) {
      return;
    }
    checkArgument(emulated.isInterface(), "Shouldn't be called for a class: %s.%s", owner, name);
    checkArgument(
        BitFlags.noneSet(
            access,
            Opcodes.ACC_ABSTRACT | Opcodes.ACC_NATIVE | Opcodes.ACC_STATIC | Opcodes.ACC_BRIDGE),
        "Should only be called for default methods: %s.%s",
        owner,
        name);
    emulatedDefaultMethods.put(
        name + ":" + desc, EmulatedMethod.create(access, emulated, name, desc, exceptions));
  }

  /**
   * If the given invocation needs to go through a companion class of an emulated or renamed core
   * interface, this methods returns that interface. This is a helper method for {@link
   * CoreLibraryInvocationRewriter}.
   *
   * <p>This method can only return non-{@code null} if {@code owner} is a core library type. It
   * usually returns an emulated interface, unless the given invocation is a super-call to a core
   * class's implementation of an emulated method that's being moved (other implementations of
   * emulated methods in core classes are ignored). In that case the class is returned and the
   * caller can use {@link #getMoveTarget} to find out where to redirect the invokespecial to.
   */
  // TODO(kmb): Rethink this API and consider combining it with getMoveTarget().
  @Nullable
  public Class<?> getCoreInterfaceRewritingTarget(
      int opcode, String owner, String name, String desc, boolean itf) {
    if (looksGenerated(owner)) {
      // Regular desugaring handles generated classes, no emulation is needed
      return null;
    }
    if (!itf && opcode == Opcodes.INVOKESTATIC) {
      // Ignore static invocations on classes--they never need rewriting (unless moved but that's
      // handled separately).
      return null;
    }
    if ("<init>".equals(name)) {
      return null; // Constructors aren't rewritten
    }

    Class<?> clazz;
    if (isRenamedCoreLibrary(owner)) {
      // For renamed invocation targets we just need to do what InterfaceDesugaring does, that is,
      // only worry about invokestatic and invokespecial interface invocations; nothing to do for
      // classes and invokeinterface.  InterfaceDesugaring ignores bootclasspath interfaces,
      // so we have to do its work here for renamed interfaces.
      if (itf && (opcode == Opcodes.INVOKESTATIC || opcode == Opcodes.INVOKESPECIAL)) {
        clazz = loadFromInternal(owner);
      } else {
        return null;
      }
    } else {
      // If not renamed, see if the owner needs emulation.
      clazz = getEmulatedCoreClassOrInterface(owner);
      if (clazz == null) {
        return null;
      }
    }
    checkArgument(itf == clazz.isInterface(), "%s expected to be interface: %s", owner, itf);

    if (opcode == Opcodes.INVOKESTATIC) {
      // Static interface invocation always goes to the given owner
      checkState(itf); // we should've bailed out above.
      return clazz;
    }

    // See if the invoked method is a default method, which will need rewriting.  For invokespecial
    // we can only get here if its a default method, and invokestatic we handled above.
    Method callee = findInterfaceMethod(clazz, name, desc);
    if (callee != null && callee.isDefault()) {
      if (isExcluded(callee)) {
        return null;
      }

      if (!itf && opcode == Opcodes.INVOKESPECIAL) {
        // See if the invoked implementation is moved; note we ignore all other overrides in classes
        Class<?> impl = clazz; // we know clazz is not an interface because !itf
        while (impl != null) {
          String implName = impl.getName().replace('.', '/');
          if (getMoveTarget(implName, name, desc) != null) {
            return impl;
          }
          impl = impl.getSuperclass();
        }
      }

      Class<?> result = callee.getDeclaringClass();
      if (isRenamedCoreLibrary(result.getName().replace('.', '/'))
          || emulatedInterfaces.stream().anyMatch(emulated -> emulated.isAssignableFrom(result))) {
        return result;
      }
      // We get here if the declaring class is a supertype of an emulated interface.  In that case
      // use the emulated interface instead (since we don't desugar the supertype).  Fail in case
      // there are multiple possibilities.
      Iterator<Class<?>> roots =
          emulatedInterfaces.stream()
              .filter(
                  emulated -> emulated.isAssignableFrom(clazz) && result.isAssignableFrom(emulated))
              .iterator();
      checkState(roots.hasNext()); // must exist
      Class<?> substitute = roots.next();
      checkState(!roots.hasNext(), "Ambiguous emulation substitute: %s", callee);
      return substitute;
    } else {
      checkArgument(
          !itf || opcode != Opcodes.INVOKESPECIAL,
          "Couldn't resolve interface super call %s.super.%s : %s",
          owner,
          name,
          desc);
    }
    return null;
  }

  /**
   * Returns the given class if it's a core library class or interface with emulated default
   * methods. This is equivalent to calling {@link #isEmulatedCoreClassOrInterface} and then just
   * loading the class (using the target class loader).
   */
  public Class<?> getEmulatedCoreClassOrInterface(String internalName) {
    if (looksGenerated(internalName)) {
      // Regular desugaring handles generated classes, no emulation is needed
      return null;
    }
    {
      String unprefixedOwner = rewriter.unprefix(internalName);
      if (!unprefixedOwner.startsWith("java/util/") || isRenamedCoreLibrary(unprefixedOwner)) {
        return null;
      }
    }

    Class<?> clazz = loadFromInternal(internalName);
    if (emulatedInterfaces.stream().anyMatch(itf -> itf.isAssignableFrom(clazz))) {
      return clazz;
    }
    return null;
  }

  /** Returns targets queried in {@link #getMoveTarget}. */
  public Set<String> usedRuntimeHelpers() {
    return unmodifiableSet(usedRuntimeHelpers);
  }

  public void makeDispatchHelpers(GeneratedClassStore store) {
    LinkedHashMap<Class<?>, ClassVisitor> dispatchHelpers = new LinkedHashMap<>();
    for (Collection<EmulatedMethod> group : emulatedDefaultMethods.asMap().values()) {
      checkState(!group.isEmpty());
      Class<?> root =
          group.stream()
              .map(EmulatedMethod::owner)
              .max(DefaultMethodClassFixer.SubtypeComparator.INSTANCE)
              .get();
      checkState(
          group.stream().map(m -> m.owner()).allMatch(o -> root.isAssignableFrom(o)),
          "Not a single unique method: %s",
          group);
      String methodName = group.stream().findAny().get().name();

      ImmutableList<Class<?>> customOverrides = findCustomOverrides(root, methodName);

      for (EmulatedMethod methodDefinition : group) {
        Class<?> owner = methodDefinition.owner();
        ClassVisitor dispatchHelper =
            dispatchHelpers.computeIfAbsent(
                owner,
                clazz -> {
                  String className = clazz.getName().replace('.', '/') + "$$Dispatch";
                  ClassVisitor result = store.add(className);
                  result.visit(
                      Opcodes.V1_7,
                      // Must be public so dispatch methods can be called from anywhere
                      Opcodes.ACC_SYNTHETIC | Opcodes.ACC_PUBLIC,
                      className,
                      /*signature=*/ null,
                      "java/lang/Object",
                      EMPTY_LIST);
                  return result;
                });

        // Types to check for before calling methodDefinition's companion, sub- before super-types
        ImmutableList<Class<?>> typechecks =
            concat(group.stream().map(EmulatedMethod::owner), customOverrides.stream())
                .filter(o -> o != owner && owner.isAssignableFrom(o))
                .distinct() // should already be but just in case
                .sorted(DefaultMethodClassFixer.SubtypeComparator.INSTANCE)
                .collect(ImmutableList.toImmutableList());
        makeDispatchHelperMethod(dispatchHelper, methodDefinition, typechecks);
      }
    }
  }

  private ImmutableList<Class<?>> findCustomOverrides(Class<?> root, String methodName) {
    ImmutableList.Builder<Class<?>> customOverrides = ImmutableList.builder();
    for (Map.Entry<MethodInvocationSite, MethodInvocationSite> move :
        retargetConfig.invocationReplacements().entrySet()) {
      // move.getKey is a string <owner>#<name> which we validated in the constructor.
      // We need to take the string apart here to compare owner and name separately.
      if (!methodName.equals(move.getKey().method().name())) {
        continue;
      }
      Class<?> target = loadFromInternal(rewriter.getPrefix() + move.getKey().method().ownerName());
      if (!root.isAssignableFrom(target)) {
        continue;
      }
      checkState(!target.isInterface(), "can't move emulated interface method: %s", move);
      customOverrides.add(target);
    }
    return customOverrides.build();
  }

  private void makeDispatchHelperMethod(
      ClassVisitor helper, EmulatedMethod method, ImmutableList<Class<?>> typechecks) {
    checkArgument(method.owner().isInterface());
    String owner = method.owner().getName().replace('.', '/');
    Type methodType = Type.getMethodType(method.descriptor());
    String companionDesc =
        InterfaceDesugaring.companionDefaultMethodDescriptor(owner, method.descriptor());
    MethodVisitor dispatchMethod =
        helper.visitMethod(
            method.access() | Opcodes.ACC_STATIC,
            method.name(),
            companionDesc,
            /*signature=*/ null, // signature is invalid due to extra "receiver" argument
            method.exceptions().toArray(EMPTY_LIST));

    dispatchMethod.visitCode();
    {
      // See if the receiver might come with its own implementation of the method, and call it.
      // We do this by testing for the interface type created by EmulatedInterfaceRewriter
      Label fallthrough = new Label();
      String emulationInterface = renameCoreLibrary(owner);
      dispatchMethod.visitVarInsn(Opcodes.ALOAD, 0); // load "receiver"
      dispatchMethod.visitTypeInsn(Opcodes.INSTANCEOF, emulationInterface);
      dispatchMethod.visitJumpInsn(Opcodes.IFEQ, fallthrough);
      dispatchMethod.visitVarInsn(Opcodes.ALOAD, 0); // load "receiver"
      dispatchMethod.visitTypeInsn(Opcodes.CHECKCAST, emulationInterface);

      visitLoadArgs(dispatchMethod, methodType, 1 /* receiver already loaded above */);
      dispatchMethod.visitMethodInsn(
          Opcodes.INVOKEINTERFACE,
          emulationInterface,
          method.name(),
          method.descriptor(),
          /*isInterface=*/ true);
      dispatchMethod.visitInsn(methodType.getReturnType().getOpcode(Opcodes.IRETURN));

      dispatchMethod.visitLabel(fallthrough);
      // Trivial frame for the branch target: same empty stack as before
      dispatchMethod.visitFrame(Opcodes.F_SAME, 0, EMPTY_FRAME, 0, EMPTY_FRAME);
    }

    // Next, check for subtypes with specialized implementations and call them
    for (Class<?> tested : typechecks) {
      Label fallthrough = new Label();
      String testedName = tested.getName().replace('.', '/');

      // In case of a class this must be a member move; for interfaces use the companion.
      final String target;
      String calledMethod = method.name();
      if (tested.isInterface()) {
        target = InterfaceDesugaring.getCompanionClassName(testedName);
        calledMethod += InterfaceDesugaring.DEFAULT_COMPANION_METHOD_SUFFIX;
      } else {
        MethodKey targetMethod =
            checkNotNull(
                    retargetConfig.findReplacementSite(
                        MethodInvocationSite.builder()
                            .setInvocationKind(MemberUseKind.INVOKEVIRTUAL)
                            .setIsInterface(false)
                            .setMethod(
                                MethodKey.create(
                                        ClassName.create(testedName),
                                        method.name(),
                                        method.descriptor())
                                    .acceptTypeMapper(ClassName.IN_PROCESS_LABEL_STRIPPER))
                            .build()))
                .acceptTypeMapper(ClassName.SHADOWED_TO_MIRRORED_TYPE_MAPPER)
                .method();
        target = targetMethod.ownerName();
      }

      dispatchMethod.visitVarInsn(Opcodes.ALOAD, 0); // load "receiver"
      dispatchMethod.visitTypeInsn(Opcodes.INSTANCEOF, testedName);
      dispatchMethod.visitJumpInsn(Opcodes.IFEQ, fallthrough);
      dispatchMethod.visitVarInsn(Opcodes.ALOAD, 0); // load "receiver"
      dispatchMethod.visitTypeInsn(Opcodes.CHECKCAST, testedName); // make verifier happy

      visitLoadArgs(dispatchMethod, methodType, 1 /* receiver already loaded above */);
      dispatchMethod.visitMethodInsn(
          Opcodes.INVOKESTATIC,
          target,
          calledMethod,
          InterfaceDesugaring.companionDefaultMethodDescriptor(testedName, method.descriptor()),
          /*isInterface=*/ false);
      dispatchMethod.visitInsn(methodType.getReturnType().getOpcode(Opcodes.IRETURN));

      dispatchMethod.visitLabel(fallthrough);
      // Trivial frame for the branch target: same empty stack as before
      dispatchMethod.visitFrame(Opcodes.F_SAME, 0, EMPTY_FRAME, 0, EMPTY_FRAME);
    }

    // Call static type's default implementation in companion class
    dispatchMethod.visitVarInsn(Opcodes.ALOAD, 0); // load "receiver"
    visitLoadArgs(dispatchMethod, methodType, 1 /* receiver already loaded above */);
    dispatchMethod.visitMethodInsn(
        Opcodes.INVOKESTATIC,
        InterfaceDesugaring.getCompanionClassName(owner),
        method.name() + InterfaceDesugaring.DEFAULT_COMPANION_METHOD_SUFFIX,
        companionDesc,
        /*isInterface=*/ false);
    dispatchMethod.visitInsn(methodType.getReturnType().getOpcode(Opcodes.IRETURN));

    dispatchMethod.visitMaxs(0, 0);
    dispatchMethod.visitEnd();
  }

  private boolean isExcluded(Method method) {
    String unprefixedOwner =
        rewriter.unprefix(method.getDeclaringClass().getName().replace('.', '/'));
    return excludeFromEmulation.contains(unprefixedOwner + "#" + method.getName());
  }

  private Class<?> loadFromInternal(String internalName) {
    try {
      return targetLoader.loadClass(internalName.replace('/', '.'));
    } catch (ClassNotFoundException e) {
      throw (NoClassDefFoundError) new NoClassDefFoundError().initCause(e);
    }
  }

  private static Method findInterfaceMethod(Class<?> clazz, String name, String desc) {
    return collectImplementedInterfaces(clazz, new LinkedHashSet<>()).stream()
        // search more subtypes before supertypes
        .sorted(DefaultMethodClassFixer.SubtypeComparator.INSTANCE)
        .map(itf -> findMethod(itf, name, desc))
        .filter(Objects::nonNull)
        .findFirst()
        .orElse((Method) null);
  }

  private static Method findMethod(Class<?> clazz, String name, String desc) {
    for (Method m : clazz.getMethods()) {
      if (m.getName().equals(name) && Type.getMethodDescriptor(m).equals(desc)) {
        return m;
      }
    }
    return null;
  }

  private static Set<Class<?>> collectImplementedInterfaces(Class<?> clazz, Set<Class<?>> dest) {
    if (clazz.isInterface()) {
      if (!dest.add(clazz)) {
        return dest;
      }
    } else if (clazz.getSuperclass() != null) {
      collectImplementedInterfaces(clazz.getSuperclass(), dest);
    }

    for (Class<?> itf : clazz.getInterfaces()) {
      collectImplementedInterfaces(itf, dest);
    }
    return dest;
  }

  /**
   * Emits instructions to load a method's parameters as arguments of a method call assumed to have
   * compatible descriptor, starting at the given local variable slot.
   */
  private static void visitLoadArgs(MethodVisitor dispatchMethod, Type neededType, int slot) {
    for (Type arg : neededType.getArgumentTypes()) {
      dispatchMethod.visitVarInsn(arg.getOpcode(Opcodes.ILOAD), slot);
      slot += arg.getSize();
    }
  }

  /** Checks whether the given class is (likely) generated by desugar itself. */
  private static boolean looksGenerated(String owner) {
    return owner.contains("$$Lambda$")
        || owner.endsWith("$$CC")
        || owner.endsWith("$NestCC")
        || owner.endsWith("$$Dispatch");
  }

  @AutoValue
  @Immutable
  abstract static class EmulatedMethod {
    public static EmulatedMethod create(
        int access, Class<?> owner, String name, String desc, @Nullable String[] exceptions) {
      return new AutoValue_CoreLibrarySupport_EmulatedMethod(
          access,
          owner,
          name,
          desc,
          exceptions != null ? ImmutableList.copyOf(exceptions) : ImmutableList.of());
    }

    abstract int access();

    abstract Class<?> owner();

    abstract String name();

    abstract String descriptor();

    abstract ImmutableList<String> exceptions();
  }
}
