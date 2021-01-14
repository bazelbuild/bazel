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

import static com.google.common.base.Preconditions.checkState;

import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.commons.ClassRemapper;
import org.objectweb.asm.commons.MethodRemapper;
import org.objectweb.asm.commons.Remapper;

/**
 * A visitor that renames packages so configured using {@link CoreLibrarySupport}. Additionally
 * generate bridge-like methods for core library overrides that should be preserved that call the
 * renamed variants.
 */
class CorePackageRenamer extends ClassRemapper {

  private String internalName;

  public CorePackageRenamer(ClassVisitor cv, CoreLibrarySupport support) {
    super(cv, new CorePackageRemapper(support));
  }

  @Override
  public void visit(
      int version,
      int access,
      String name,
      String signature,
      String superName,
      String[] interfaces) {
    checkState(internalName == null || internalName.equals(name), "Instance already used.");
    internalName = name;
    super.visit(version, access, name, signature, superName, interfaces);
  }

  @Override
  protected CoreMethodRemapper createMethodRemapper(MethodVisitor methodVisitor) {
    return new CoreMethodRemapper(methodVisitor, remapper);
  }

  private class CoreMethodRemapper extends MethodRemapper {

    public CoreMethodRemapper(MethodVisitor methodVisitor, Remapper remapper) {
      super(methodVisitor, remapper);
    }

    @Override
    public void visitMethodInsn(
        int opcode, String owner, String name, String descriptor, boolean isInterface) {
      CorePackageRemapper remapper = (CorePackageRemapper) this.remapper;
      remapper.didSomething = false;
      super.visitMethodInsn(opcode, owner, name, descriptor, isInterface);
      // TODO(b/79121791): Make this more precise: look for all unsupported core library members
      checkState(
          !remapper.didSomething
              || !owner.startsWith("android/")
              || owner.startsWith("android/arch/")
              || owner.startsWith("android/support/"),
          "%s calls %s.%s%s which is not supported with core library desugaring. Please file "
              + "a feature request to support this method",
          internalName,
          owner,
          name,
          descriptor);
    }

    @Override
    public void visitFieldInsn(int opcode, String owner, String name, String descriptor) {
      CorePackageRemapper remapper = (CorePackageRemapper) this.remapper;
      remapper.didSomething = false;
      super.visitFieldInsn(opcode, owner, name, descriptor);
      // TODO(b/79121791): Make this more precise: look for all unsupported core library members
      checkState(
          !remapper.didSomething
              || !owner.startsWith("android/")
              || owner.startsWith("android/arch/")
              || owner.startsWith("android/car/")
              || owner.startsWith("android/support/"),
          "%s accesses %s.%s: %s which is not supported with core library desugaring. Please file "
              + "a feature request to support this field",
          internalName,
          owner,
          name,
          descriptor);
    }
  }

  /** ASM {@link Remapper} based on {@link CoreLibrarySupport}. */
  private static class CorePackageRemapper extends Remapper {

    private final CoreLibrarySupport support;
    boolean didSomething = false;

    CorePackageRemapper(CoreLibrarySupport support) {
      this.support = support;
    }

    @Override
    public String map(String typeName) {
      if (support.isRenamedCoreLibrary(typeName)) {
        didSomething = true;
        return support.renameCoreLibrary(typeName);
      }
      return typeName;
    }
  }
}
