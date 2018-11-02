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

import com.google.devtools.build.android.desugar.io.BitFlags;
import java.util.Collections;
import java.util.LinkedHashSet;
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.Opcodes;

/**
 * Visitor that renames emulated interfaces and marks classes that extend emulated interfaces to
 * also implement the renamed interfaces.  {@link DefaultMethodClassFixer} makes sure the requisite
 * methods are present in all classes implementing the renamed interface.  Doing this helps with
 * dynamic dispatch on emulated interfaces.
 */
public class EmulatedInterfaceRewriter extends ClassVisitor {

  private static final String[] EMPTY_ARRAY = new String[0];

  private final CoreLibrarySupport support;

  public EmulatedInterfaceRewriter(ClassVisitor dest, CoreLibrarySupport support) {
    super(Opcodes.ASM7, dest);
    this.support = support;
  }

  @Override
  public void visit(
      int version,
      int access,
      String name,
      String signature,
      String superName,
      String[] interfaces) {
    boolean emulated = support.isEmulatedCoreClassOrInterface(name);
    {
      // 1. see if we should implement any additional interfaces.
      // Use LinkedHashSet to dedupe but maintain deterministic order
      LinkedHashSet<String> newInterfaces = new LinkedHashSet<>();
      if (interfaces != null && interfaces.length > 0) {
        // Make classes implementing emulated interfaces also implement the renamed interfaces we
        // create below.  This includes making the renamed interfaces extends each other as needed.
        Collections.addAll(newInterfaces, interfaces);
        for (String itf : interfaces) {
          if (support.isEmulatedCoreClassOrInterface(itf)) {
            newInterfaces.add(support.renameCoreLibrary(itf));
          }
        }
      }
      if (!emulated) {
        // For an immediate subclass of an emulated class, also fill in any interfaces implemented
        // by superclasses, similar to the additional default method stubbing performed in
        // DefaultMethodClassFixer in this situation.
        Class<?> superclass = support.getEmulatedCoreClassOrInterface(superName);
        while (superclass != null) {
          for (Class<?> implemented : superclass.getInterfaces()) {
            String itf = implemented.getName().replace('.', '/');
            if (support.isEmulatedCoreClassOrInterface(itf)) {
              newInterfaces.add(support.renameCoreLibrary(itf));
            }
          }
          superclass = superclass.getSuperclass();
        }
      }
      // Update implemented interfaces and signature if we did anything above
      if (interfaces == null
          ? !newInterfaces.isEmpty()
          : interfaces.length != newInterfaces.size()) {
        interfaces = newInterfaces.toArray(EMPTY_ARRAY);
        signature = null; // additional interfaces invalidate any signature
      }
    }

    // 2. see if we need to rename this interface itself
    if (BitFlags.isInterface(access) && emulated) {
      name = support.renameCoreLibrary(name);
    }
    super.visit(version, access, name, signature, superName, interfaces);
  }
}
