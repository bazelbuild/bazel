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

import static com.google.common.base.Preconditions.checkArgument;

import com.google.common.collect.Iterators;
import com.google.common.collect.UnmodifiableIterator;
import com.google.common.io.ByteStreams;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;
import org.objectweb.asm.ClassReader;
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.ClassWriter;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;

/**
 * Test data generator for b/62456849. This class converts methods satisfying the following
 * conditions to synthetic methods.
 * <li>The name starts with "lambda$"
 * <li>Not synthetic
 */
public class Bug62456849TestDataGenerator {

  public static void main(String[] args) throws IOException {
    checkArgument(
        args.length == 2,
        "Usage: %s <input-jar> <output-jar>",
        Bug62456849TestDataGenerator.class.getName());
    Path inputJar = Paths.get(args[0]);
    checkArgument(Files.isRegularFile(inputJar), "The input jar %s is not a file", inputJar);
    Path outputJar = Paths.get(args[1]);

    try (ZipFile inputZip = new ZipFile(inputJar.toFile());
        ZipOutputStream outZip =
            new ZipOutputStream(new BufferedOutputStream(Files.newOutputStream(outputJar)))) {
      for (UnmodifiableIterator<? extends ZipEntry> it =
              Iterators.forEnumeration(inputZip.entries());
          it.hasNext(); ) {
        ZipEntry entry = it.next();
        String entryName = entry.getName();
        byte[] content =
            entryName.endsWith(".class")
                ? convertClass(inputZip, entry)
                : readEntry(inputZip, entry);
        writeToZipFile(outZip, entryName, content);
      }
    }
  }

  private static void writeToZipFile(ZipOutputStream outZip, String entryName, byte[] content)
      throws IOException {
    ZipEntry result = new ZipEntry(entryName);
    result.setTime(0L);
    outZip.putNextEntry(result);
    outZip.write(content);
    outZip.closeEntry();
  }

  private static byte[] readEntry(ZipFile file, ZipEntry entry) throws IOException {
    try (InputStream is = file.getInputStream(entry)) {
      return ByteStreams.toByteArray(is);
    }
  }

  private static byte[] convertClass(ZipFile file, ZipEntry entry) throws IOException {
    try (InputStream content = file.getInputStream(entry)) {
      ClassReader reader = new ClassReader(content);
      ClassWriter writer = new ClassWriter(0);
      ClassVisitor converter =
          new ClassVisitor(Opcodes.ASM8, writer) {
            @Override
            public MethodVisitor visitMethod(
                int access, String name, String desc, String signature, String[] exceptions) {
              if (name.startsWith("lambda$") && (access & Opcodes.ACC_SYNTHETIC) == 0) {
                access |= Opcodes.ACC_SYNTHETIC;
              }
              return super.visitMethod(access, name, desc, signature, exceptions);
            }
          };
      reader.accept(converter, 0);
      return writer.toByteArray();
    }
  }
}
