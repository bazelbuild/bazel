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

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.android.desugar.io.BitFlags;
import com.google.errorprone.annotations.Immutable;
import java.util.LinkedHashMap;
import java.util.Map;
import javax.annotation.Nullable;
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.Type;
import org.objectweb.asm.commons.ClassRemapper;
import org.objectweb.asm.commons.MethodRemapper;
import org.objectweb.asm.commons.Remapper;

/**
 * A visitor that renames packages so configured using {@link CoreLibrarySupport}. Additionally
 * generate bridge-like methods for core library overrides that should be preserved that call the
 * renamed variants.
 */
class CorePackageRenamer extends ClassRemapper {

  private final CoreLibrarySupport coreLibrarySupport;
  private final Map<String, PreservedMethod> preserveOriginals = new LinkedHashMap<>();
  private String internalName;

  public CorePackageRenamer(ClassVisitor cv, CoreLibrarySupport support) {
    super(cv, new CorePackageRemapper(support));
    coreLibrarySupport = support;
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
  public MethodVisitor visitMethod(
      int access, String name, String descriptor, String signature, String[] exceptions) {
    if (coreLibrarySupport.preserveOriginalMethod(access, internalName, name, descriptor)) {
      if (BitFlags.isSynthetic(access)) {
        // Idempotency: this is the preserved override, which we mark synthetic for simplicity.
        return cv != null ? cv.visitMethod(access, name, descriptor, signature, exceptions) : null;
      } else {
        PreservedMethod preserve =
            PreservedMethod.create(access, name, descriptor, signature, exceptions);
        preserveOriginals.put(name + ":" + descriptor, preserve);
      }
    }
    return super.visitMethod(access, name, descriptor, signature, exceptions);
  }

  @Override
  public void visitEnd() {
    if (cv == null) {
      return;
    }

    // Re-generate preserved method overrides to call their remapped implementations
    for (PreservedMethod preserve : preserveOriginals.values()) {
      CorePackageRemapper remapper = (CorePackageRemapper) this.remapper;
      remapper.didSomething = false;
      String remappedDesc = remapper.mapMethodDesc(preserve.desc());
      checkState(remapper.didSomething, "Unnecessarily preserving %s", preserve);

      // Create method using cv field instead of super so renaming isn't applied.  Use synthetic
      // flag so we can recognize and skip this method if desugar is run again over the output.
      // Also drop any "abstract" bit just in case.
      int access = (preserve.access() & ~Opcodes.ACC_ABSTRACT) | Opcodes.ACC_SYNTHETIC;
      MethodVisitor stubMethod =
          cv.visitMethod(
              access,
              preserve.name(),
              preserve.desc(),
              preserve.signature(),
              preserve.exceptions().toArray(new String[0]));

      // Load all the arguments and call the previously encountered method, converting desugared
      // core library types as needed as we go.
      int slot = 0;
      stubMethod.visitVarInsn(Opcodes.ALOAD, slot++); // receiver
      Type neededType = Type.getMethodType(preserve.desc());
      for (Type arg : neededType.getArgumentTypes()) {
        stubMethod.visitVarInsn(arg.getOpcode(Opcodes.ILOAD), slot);
        slot += arg.getSize();
        String argDesc = arg.getDescriptor();
        String mappedArg = remapper.mapDesc(argDesc);

        // Insert conversion if necessary
        if (!argDesc.equals(mappedArg)) {
          // TODO(kmb): May need to support array types
          checkState(arg.getSort() == Type.OBJECT, "Can only map object types: %s", arg);
          String converter = coreLibrarySupport.getFromCoreLibraryConverter(arg.getInternalName());
          String simpleClassName =
              arg.getInternalName().substring(arg.getInternalName().lastIndexOf('/') + 1);
          stubMethod.visitMethodInsn(
              Opcodes.INVOKESTATIC,
              /*owner=*/ converter,
              /*name=*/ "from" + simpleClassName, // naming convention for converter methods
              /*descriptor=*/ "(" + argDesc + ")" + mappedArg,
              /*isInterface=*/ false);
        }
      }
      stubMethod.visitMethodInsn(
          Opcodes.INVOKEVIRTUAL,
          internalName,
          preserve.name(),
          remappedDesc,
          /*isInterface=*/ false);

      // TODO(kmb): May need to support "to"-converting return types
      String returnDesc = neededType.getReturnType().getDescriptor();
      checkState(
          returnDesc.equals(remapper.mapDesc(returnDesc)),
          "Return value conversions not supported: %s",
          returnDesc);
      stubMethod.visitInsn(neededType.getReturnType().getOpcode(Opcodes.IRETURN));
      stubMethod.visitMaxs(slot, slot);
      stubMethod.visitEnd();
    }
    super.visitEnd();
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

  /** Undesugared core library override to preserve. */
  @AutoValue
  @Immutable
  abstract static class PreservedMethod {
    private static PreservedMethod create(
        int access,
        String name,
        String desc,
        @Nullable String signature,
        @Nullable String[] exceptions) {
      return new AutoValue_CorePackageRenamer_PreservedMethod(
          access,
          name,
          desc,
          signature,
          exceptions != null ? ImmutableList.copyOf(exceptions) : ImmutableList.of());
    }

    abstract int access();

    abstract String name();

    abstract String desc();

    @Nullable
    abstract String signature();

    abstract ImmutableList<String> exceptions();
  }
}
