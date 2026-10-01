// Copyright 2026 The Bazel Authors. All rights reserved.
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

package test;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.jar.JarOutputStream;
import java.util.zip.ZipEntry;

/**
 * Generates a JAR containing a class with a {@code Record} attribute whose declared {@code
 * attribute_length} is smaller than the actual component data, and whose total serialized component
 * data exceeds 65,535 bytes (testing 32-bit {@code u4} length backpatching).
 */
public final class GenRecordLengthMismatch {
  public static void main(String[] args) throws IOException {
    try (JarOutputStream jos = new JarOutputStream(Files.newOutputStream(Paths.get(args[0])))) {
      ZipEntry foo = new ZipEntry("Foo.class");
      foo.setTime(0);
      jos.putNextEntry(foo);
      jos.write(dump("Foo", /* mismatchedLength= */ true));

      ZipEntry bar = new ZipEntry("Bar.class");
      bar.setTime(0);
      jos.putNextEntry(bar);
      jos.write(dump("Bar", /* mismatchedLength= */ false));
    }
  }

  private static byte[] dump(String className, boolean mismatchedLength) throws IOException {
    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    DataOutputStream dos = new DataOutputStream(baos);

    // Header: magic, minor (0), major (61 = Java 17)
    dos.writeInt(0xCAFEBABE);
    dos.writeShort(0);
    dos.writeShort(61);

    // Constant pool (count = 10, entries 1..9)
    dos.writeShort(10);
    // #1: Utf8 className
    dos.writeByte(1);
    dos.writeUTF(className);
    // #2: Class #1
    dos.writeByte(7);
    dos.writeShort(1);
    // #3: Utf8 "java/lang/Record"
    dos.writeByte(1);
    dos.writeUTF("java/lang/Record");
    // #4: Class #3
    dos.writeByte(7);
    dos.writeShort(3);
    // #5: Utf8 "Record"
    dos.writeByte(1);
    dos.writeUTF("Record");
    // #6: Utf8 "x"
    dos.writeByte(1);
    dos.writeUTF("x");
    // #7: Utf8 "I"
    dos.writeByte(1);
    dos.writeUTF("I");
    // #8: Utf8 "priv"
    dos.writeByte(1);
    dos.writeUTF("priv");
    // #9: Utf8 "()V"
    dos.writeByte(1);
    dos.writeUTF("()V");

    // access_flags (ACC_PUBLIC | ACC_FINAL | ACC_SUPER), this_class (#2), super_class (#4)
    dos.writeShort(0x0031);
    dos.writeShort(2);
    dos.writeShort(4);

    // interfaces_count (0), fields_count (0)
    dos.writeShort(0);
    dos.writeShort(0);

    // methods_count (1): private void priv() so ijar strips it when valid
    dos.writeShort(1);
    dos.writeShort(0x0002); // ACC_PRIVATE
    dos.writeShort(8); // "priv"
    dos.writeShort(9); // "()V"
    dos.writeShort(0); // attributes_count = 0

    // attributes_count (1)
    dos.writeShort(1);
    // Record attribute: attribute_name_index (#5 = "Record")
    dos.writeShort(5);
    int componentsCount = 65535;
    int actualPayloadSize = 2 + componentsCount * 6;
    dos.writeInt(mismatchedLength ? 6 : actualPayloadSize);
    dos.writeShort(componentsCount);
    for (int i = 0; i < componentsCount; i++) {
      // name_index (#6 = "x"), descriptor_index (#7 = "I"), attributes_count (0)
      dos.writeShort(6);
      dos.writeShort(7);
      dos.writeShort(0);
    }

    dos.flush();
    return baos.toByteArray();
  }

  private GenRecordLengthMismatch() {}
}
