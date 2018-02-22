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
import static com.google.common.base.Preconditions.checkState;

import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import java.lang.reflect.Method;
import java.util.LinkedHashSet;
import java.util.List;
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
  /** Map from {@code owner#name} core library members to their new owners. */
  private final ImmutableMap<String, String> memberMoves;

  public CoreLibrarySupport(CoreLibraryRewriter rewriter, ClassLoader targetLoader,
      ImmutableList<String> renamedPrefixes, ImmutableList<String> emulatedInterfaces,
      List<String> memberMoves) {
    this.rewriter = rewriter;
    this.targetLoader = targetLoader;
    checkArgument(
        renamedPrefixes.stream().allMatch(prefix -> prefix.startsWith("java/")), renamedPrefixes);
    this.renamedPrefixes = renamedPrefixes;
    ImmutableList.Builder<Class<?>> classBuilder = ImmutableList.builder();
    for (String itf : emulatedInterfaces) {
      checkArgument(itf.startsWith("java/util/"), itf);
      Class<?> clazz = loadFromInternal(rewriter.getPrefix() + itf);
      checkArgument(clazz.isInterface(), itf);
      classBuilder.add(clazz);
    }
    this.emulatedInterfaces = classBuilder.build();

    // We can call isRenamed and rename below b/c we initialized the necessary fields above
    ImmutableMap.Builder<String, String> movesBuilder = ImmutableMap.builder();
    Splitter splitter = Splitter.on("->").trimResults().omitEmptyStrings();
    for (String move : memberMoves) {
      List<String> pair = splitter.splitToList(move);
      checkArgument(pair.size() == 2, "Doesn't split as expected: %s", move);
      checkArgument(pair.get(0).startsWith("java/"), "Unexpected member: %s", move);
      int sep = pair.get(0).indexOf('#');
      checkArgument(sep > 0 && sep == pair.get(0).lastIndexOf('#'), "invalid member: %s", move);
      checkArgument(!isRenamedCoreLibrary(pair.get(0).substring(0, sep)),
          "Original renamed, no need to move it: %s", move);
      checkArgument(isRenamedCoreLibrary(pair.get(1)), "Target not renamed: %s", move);

      movesBuilder.put(pair.get(0), renameCoreLibrary(pair.get(1)));
    }
    this.memberMoves = movesBuilder.build();
  }

  public boolean isRenamedCoreLibrary(String internalName) {
    String unprefixedName = rewriter.unprefix(internalName);
    if (!unprefixedName.startsWith("java/") || renamedPrefixes.isEmpty()) {
      return false; // shortcut
    }
    // Rename any classes desugar might generate under java/ (for emulated interfaces) as well as
    // configured prefixes
    return unprefixedName.contains("$$Lambda$")
        || unprefixedName.endsWith("$$CC")
        || renamedPrefixes.stream().anyMatch(prefix -> unprefixedName.startsWith(prefix));
  }

  public String renameCoreLibrary(String internalName) {
    internalName = rewriter.unprefix(internalName);
    return (internalName.startsWith("java/"))
        ? "j$/" + internalName.substring(/* cut away "java/" prefix */ 5)
        : internalName;
  }

  @Nullable
  public String getMoveTarget(String owner, String name) {
    return memberMoves.get(rewriter.unprefix(owner) + '#' + name);
  }

  /**
   * Returns {@code true} for java.* classes or interfaces that are subtypes of emulated interfaces.
   * Note that implies that this method always returns {@code false} for user-written classes.
   */
  public boolean isEmulatedCoreClassOrInterface(String internalName) {
    return getEmulatedCoreClassOrInterface(internalName) != null;
  }

  /**
   * If the given invocation needs to go through a companion class of an emulated or renamed
   * core interface, this methods returns that interface.  This is a helper method for
   * {@link CoreLibraryInvocationRewriter}.
   *
   * <p>Always returns an interface (or {@code null}), even if {@code owner} is a class. Can only
   * return non-{@code null} if {@code owner} is a core library type.
   */
  @Nullable
  public Class<?> getCoreInterfaceRewritingTarget(
      int opcode, String owner, String name, String desc, boolean itf) {
    if (owner.contains("$$Lambda$") || owner.endsWith("$$CC")) {
      // Regular desugaring handles generated classes, no emulation is needed
      return null;
    }
    if (!itf && (opcode == Opcodes.INVOKESTATIC || opcode == Opcodes.INVOKESPECIAL)) {
      // Ignore staticly dispatched invocations on classes--they never need rewriting
      return null;
    }
    Class<?> clazz;
    if (isRenamedCoreLibrary(owner)) {
      // For renamed invocation targets we just need to do what InterfaceDesugaring does, that is,
      // only worry about invokestatic and invokespecial interface invocations; nothing to do for
      // invokevirtual and invokeinterface.  InterfaceDesugaring ignores bootclasspath interfaces,
      // so we have to do its work here for renamed interfaces.
      if (itf
          && (opcode == Opcodes.INVOKESTATIC || opcode == Opcodes.INVOKESPECIAL)) {
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
      return callee.getDeclaringClass();
    } else {
      checkArgument(opcode != Opcodes.INVOKESPECIAL,
          "Couldn't resolve interface super call %s.super.%s : %s", owner, name, desc);
    }
    return null;
  }

  private Class<?> getEmulatedCoreClassOrInterface(String internalName) {
    if (internalName.contains("$$Lambda$") || internalName.endsWith("$$CC")) {
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

  private Class<?> loadFromInternal(String internalName) {
    try {
      return targetLoader.loadClass(internalName.replace('/', '.'));
    } catch (ClassNotFoundException e) {
      throw (NoClassDefFoundError) new NoClassDefFoundError().initCause(e);
    }
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
