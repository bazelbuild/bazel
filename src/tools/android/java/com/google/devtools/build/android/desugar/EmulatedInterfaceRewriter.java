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

import java.util.ArrayList;
import java.util.Collections;
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.Opcodes;

/**
 * Visitor that renames emulated interfaces and marks classes that extend emulated interfaces to
 * also implement the renamed interfaces.  {@link DefaultMethodClassFixer} makes sure the requisite
 * methods are present.  Doing this helps with dynamic dispatch on emulated interfaces.
 */
public class EmulatedInterfaceRewriter extends ClassVisitor {

  private final CoreLibrarySupport support;

  public EmulatedInterfaceRewriter(ClassVisitor dest, CoreLibrarySupport support) {
    super(Opcodes.ASM6, dest);
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
    boolean isEmulated = support.isEmulatedCoreClassOrInterface(name);
    if (interfaces != null && interfaces.length > 0 && !isEmulated) {
      // Make classes implementing emulated interfaces also implement the renamed interfaces we
      // create below.
      ArrayList<String> newInterfaces = new ArrayList<>(interfaces.length + 2);
      Collections.addAll(newInterfaces, interfaces);
      for (String itf : interfaces) {
        if (support.isEmulatedCoreClassOrInterface(itf)) {
          newInterfaces.add(support.renameCoreLibrary(itf));
        }
      }
      if (interfaces.length != newInterfaces.size()) {
        interfaces = newInterfaces.toArray(interfaces);
        signature = null; // additional interfaces invalidate signature
      }
    }

    if (BitFlags.isInterface(access) && isEmulated) {
      name = support.renameCoreLibrary(name);
    }
    super.visit(version, access, name, signature, superName, interfaces);
  }
}
