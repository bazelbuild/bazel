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

import com.google.common.collect.ImmutableList;
import java.lang.reflect.Method;
import java.util.LinkedHashSet;
import java.util.Objects;
import java.util.Set;
import javax.annotation.Nullable;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.Type;

/**
 * Helper that keeps track of which core library classes and methods we want to rewrite.
 */
class CoreLibrarySupport {

  private final CoreLibraryRewriter rewriter;
  private final ClassLoader targetLoader;
  /** Internal name prefixes that we want to move to a custom package. */
  private final ImmutableList<String> renamedPrefixes;
  /** Internal names of interfaces whose default and static interface methods we'll emulate. */
  private final ImmutableList<Class<?>> emulatedInterfaces;

  public CoreLibrarySupport(CoreLibraryRewriter rewriter, ClassLoader targetLoader,
      ImmutableList<String> renamedPrefixes, ImmutableList<String> emulatedInterfaces)
      throws ClassNotFoundException {
    this.rewriter = rewriter;
    this.targetLoader = targetLoader;
    checkArgument(
        renamedPrefixes.stream().allMatch(prefix -> prefix.startsWith("java/")), renamedPrefixes);
    this.renamedPrefixes = renamedPrefixes;
    ImmutableList.Builder<Class<?>> classBuilder = ImmutableList.builder();
    for (String itf : emulatedInterfaces) {
      checkArgument(itf.startsWith("java/util/"), itf);
      Class<?> clazz = targetLoader.loadClass((rewriter.getPrefix() + itf).replace('/', '.'));
      checkArgument(clazz.isInterface(), itf);
      classBuilder.add(clazz);
    }
    this.emulatedInterfaces = classBuilder.build();
  }

  public boolean isRenamedCoreLibrary(String internalName) {
    String unprefixedName = rewriter.unprefix(internalName);
    return renamedPrefixes.stream().anyMatch(prefix -> unprefixedName.startsWith(prefix));
  }

  public String renameCoreLibrary(String internalName) {
    internalName = rewriter.unprefix(internalName);
    return (internalName.startsWith("java/"))
        ? "j$/" + internalName.substring(/* cut away "java/" prefix */ 5)
        : internalName;
  }

  public boolean isEmulatedCoreLibraryInvocation(
      int opcode, String owner, String name, String desc, boolean itf) {
    return getEmulatedCoreLibraryInvocationTarget(opcode, owner, name, desc, itf) != null;
  }

  @Nullable
  public Class<?> getEmulatedCoreLibraryInvocationTarget(
      int opcode, String owner, String name, String desc, boolean itf) {
    if (owner.contains("$$Lambda$") || owner.endsWith("$$CC")) {
      return null;  // regular desugaring handles invocations on generated classes, no emulation
    }
    Class<?> clazz = getEmulatedCoreClassOrInterface(owner);
    if (clazz == null) {
      return null;
    }

    if (itf && opcode == Opcodes.INVOKESTATIC) {
      return clazz; // static interface method
    }

    Method callee = findInterfaceMethod(clazz, name, desc);
    if (callee != null && callee.isDefault()) {
      return callee.getDeclaringClass();
    }
    return null;
  }

  private Class<?> getEmulatedCoreClassOrInterface(String internalName) {
    {
      String unprefixedOwner = rewriter.unprefix(internalName);
      if (!unprefixedOwner.startsWith("java/util/") || isRenamedCoreLibrary(unprefixedOwner)) {
        return null;
      }
    }

    Class<?> clazz;
    try {
      clazz = targetLoader.loadClass(internalName.replace('/', '.'));
    } catch (ClassNotFoundException e) {
      throw (NoClassDefFoundError) new NoClassDefFoundError().initCause(e);
    }

    if (emulatedInterfaces.stream().anyMatch(itf -> itf.isAssignableFrom(clazz))) {
      return clazz;
    }
    return null;
  }

  private static Method findInterfaceMethod(Class<?> clazz, String name, String desc) {
    return collectImplementedInterfaces(clazz, new LinkedHashSet<>())
        .stream()
        // search more subtypes before supertypes
        .sorted(DefaultMethodClassFixer.InterfaceComparator.INSTANCE)
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
}
