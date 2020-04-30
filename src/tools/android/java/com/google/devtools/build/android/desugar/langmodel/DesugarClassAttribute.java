/*
 * Copyright 2020 The Bazel Authors. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.devtools.build.android.desugar.langmodel;

import com.google.protobuf.ExtensionRegistryLite;
import com.google.protobuf.InvalidProtocolBufferException;
import java.io.IOError;
import org.objectweb.asm.Attribute;
import org.objectweb.asm.ByteVector;
import org.objectweb.asm.ClassReader;
import org.objectweb.asm.ClassWriter;
import org.objectweb.asm.Label;

/** A custom class file attribute for desugar-specific operations. */
public class DesugarClassAttribute extends Attribute {

  private static final String ATTRIBUTE_TYPE = "DesugarClassInfo";

  private final DesugarClassInfo desugarClassInfo;

  public DesugarClassAttribute() {
    this(DesugarClassInfo.getDefaultInstance());
  }

  public DesugarClassAttribute(DesugarClassInfo desugarClassInfo) {
    super(ATTRIBUTE_TYPE);
    this.desugarClassInfo = desugarClassInfo;
  }

  public DesugarClassInfo getDesugarClassInfo() {
    return desugarClassInfo;
  }

  @Override
  protected Attribute read(
      ClassReader classReader,
      int offset,
      int length,
      char[] charBuffer,
      int codeAttributeOffset,
      Label[] labels) {
    byte[] classAttrBytes = new byte[length];
    for (int i = 0; i < length; i++) {
      classAttrBytes[i] = (byte) classReader.readByte(i + offset);
    }
    try {
      DesugarClassInfo attr =
          DesugarClassInfo.parseFrom(classAttrBytes, ExtensionRegistryLite.getEmptyRegistry());
      return new DesugarClassAttribute(attr);
    } catch (InvalidProtocolBufferException e) {
      throw new IOError(e);
    }
  }

  @Override
  protected ByteVector write(
      ClassWriter classWriter, byte[] code, int codeLength, int maxStack, int maxLocals) {
    ByteVector byteVector = new ByteVector();
    byte[] classAttrBytes = desugarClassInfo.toByteArray();
    byteVector.putByteArray(classAttrBytes, 0, classAttrBytes.length);
    return byteVector;
  }
}
