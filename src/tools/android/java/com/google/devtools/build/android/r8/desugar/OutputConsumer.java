// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.r8.desugar;

import static com.google.common.base.Preconditions.checkArgument;
import static org.objectweb.asm.Opcodes.ACC_INTERFACE;
import static org.objectweb.asm.Opcodes.ASM7;
import static org.objectweb.asm.Opcodes.INVOKESTATIC;

import com.android.tools.r8.ByteDataView;
import com.android.tools.r8.ClassFileConsumer;
import com.android.tools.r8.DiagnosticsHandler;
import com.android.tools.r8.origin.Origin;
import com.android.tools.r8.origin.PathOrigin;
import com.android.tools.r8.utils.ExceptionDiagnostic;
import com.google.devtools.build.android.desugar.DependencyCollector;
import com.google.devtools.build.android.r8.DescriptorUtils;
import com.google.devtools.build.android.r8.Desugar;
import com.google.devtools.build.android.r8.ZipUtils;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.NavigableSet;
import java.util.TreeSet;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;
import org.objectweb.asm.ClassReader;
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.MethodVisitor;

/**
 * Class file consumer for collecting the output from desugaring.
 *
 * <p>When the classes are written the desugar dependency information is collected by an ASM pass
 * over the output, as the D8 desugaring engine does not have hooks for collecting this information
 * during desugaring.
 *
 * <p>The missing classes are collected during compilation as D8 produce these as warning
 * diagnostics with structured information on the missing classes.
 */
public class OutputConsumer implements ClassFileConsumer {

  private static class ClassFileData implements Comparable<ClassFileData> {
    private final String fileName;
    private final byte[] data;

    private ClassFileData(String fileName, byte[] data) {
      this.fileName = fileName;
      this.data = data;
    }

    @Override
    public int compareTo(ClassFileData other) {
      if (other == null) {
        return fileName.compareTo(null);
      }
      return fileName.compareTo(other.fileName);
    }
  }

  private final Path archive;
  private final Origin origin;
  private final DependencyCollector dependencyCollector;
  private final NavigableSet<ClassFileData> classFiles = new TreeSet<>();

  public OutputConsumer(Path archive, DependencyCollector dependencyCollector) {
    this.archive = archive;
    this.origin = new PathOrigin(archive);
    this.dependencyCollector = dependencyCollector;
  }

  @Override
  public void accept(ByteDataView data, String descriptor, DiagnosticsHandler handler) {
    classFiles.add(
        new ClassFileData(
            DescriptorUtils.descriptorToClassFileName(descriptor), data.copyByteData()));
  }

  @Override
  public void finished(DiagnosticsHandler handler) {
    try (ZipOutputStream out =
        new ZipOutputStream(new BufferedOutputStream(Files.newOutputStream(archive)))) {
      for (ClassFileData classFile : classFiles) {
        ZipUtils.addEntry(classFile.fileName, classFile.data, ZipEntry.STORED, out);
        new DesugaredClassFileDependencyCollector(classFile, dependencyCollector).run();
      }
      // Add dependency metadata if required.
      byte[] desugarDeps = dependencyCollector.toByteArray();
      if (desugarDeps != null) {
        ZipUtils.addEntry(Desugar.DESUGAR_DEPS_FILENAME, desugarDeps, ZipEntry.STORED, out);
      }
    } catch (IOException e) {
      handler.error(new ExceptionDiagnostic(e, origin));
    }
  }

  /** Record a missing interface in the output */
  public void missingImplementedInterface(String origin, String target) {
    checkArgument(DescriptorUtils.isBinaryName(origin), "'%s' is not a binary name", origin);
    checkArgument(DescriptorUtils.isBinaryName(target), "'%s' is not a binary name", target);
    dependencyCollector.missingImplementedInterface(origin, target);
  }

  private static class DesugaredClassFileDependencyCollector {

    private static class DependencyCollectorClassVisitor extends ClassVisitor {

      private final DependencyCollector dependencyCollector;
      private String className;
      private int methodCount;

      private DependencyCollectorClassVisitor(DependencyCollector dependencyCollector) {
        super(ASM7, null);
        this.dependencyCollector = dependencyCollector;
      }

      @Override
      public void visit(
          int version,
          int access,
          String name,
          String signature,
          String superName,
          String[] interfaces) {
        this.className = name;
        this.methodCount = 0;
        if ((access & ACC_INTERFACE) == ACC_INTERFACE) {
          dependencyCollector.recordExtendedInterfaces(name, interfaces);
        }
      }

      @Override
      public void visitEnd() {
        if (DescriptorUtils.isCompanionClassBinaryName(className)) {
          dependencyCollector.recordDefaultMethods(
              className.substring(0, className.length() - 4), methodCount);
        }
      }

      @Override
      public MethodVisitor visitMethod(
          int access, String name, String descriptor, String signature, String[] exceptions) {
        methodCount++;
        return new DependencyCollectorMethodVisitor(api);
      }

      private class DependencyCollectorMethodVisitor extends MethodVisitor {

        DependencyCollectorMethodVisitor(int api) {
          super(api);
        }

        @Override
        public void visitMethodInsn(
            final int opcode,
            final String owner,
            final String name,
            final String descriptor,
            final boolean isInterface) {
          super.visitMethodInsn(opcode, owner, name, descriptor, isInterface);
          if (isInterface) {
            return;
          }
          if (opcode != INVOKESTATIC) {
            return;
          }
          if (DescriptorUtils.isCompanionClassBinaryName(owner)) {
            dependencyCollector.assumeCompanionClass(className, owner);
          }
        }
      }
    }

    private final ClassFileData classFileData;
    private final DependencyCollector dependencyCollector;

    private DesugaredClassFileDependencyCollector(
        ClassFileData classFileData, DependencyCollector dependencyCollector) {
      this.classFileData = classFileData;
      this.dependencyCollector = dependencyCollector;
    }

    private void run() {
      new ClassReader(classFileData.data)
          .accept(
              new DependencyCollectorClassVisitor(dependencyCollector),
              ClassReader.SKIP_DEBUG | ClassReader.SKIP_FRAMES);
    }
  }
}
