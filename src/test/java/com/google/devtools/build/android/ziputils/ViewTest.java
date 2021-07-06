// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.ziputils;

import static com.google.common.truth.Truth.assertWithMessage;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertThrows;

import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link View}. */
@RunWith(JUnit4.class)
public class ViewTest {

  private static final FakeFileSystem fileSystem = new FakeFileSystem();

  @Test
  public void testView() {
    // View takes ownership of constructor argument!
    // Subclasses are responsible for slicing, when needed.
    ByteBuffer buffer = ByteBuffer.allocate(100);
    TestView instance = new TestView(buffer);
    buffer.putInt(12345678);
    int fromBuf = buffer.getInt(0);
    int fromView = instance.getInt(0);
    assertWithMessage("must assume buffer ownership").that(fromView).isEqualTo(fromBuf);
    int posBuf = buffer.position();
    int posView = instance.buffer.position();
    assertWithMessage("must assume buffer ownership").that(posView).isEqualTo(posBuf);
  }

  @Test
  public void testAt() {
    long fileOffset = 0L;
    ByteBuffer buffer = ByteBuffer.allocate(100);
    TestView instance = new TestView(buffer);
    View<TestView> result = instance.at(fileOffset);
    assertWithMessage("didn't return this").that(result).isSameInstanceAs(instance);

    long resultValue = instance.fileOffset();
    assertWithMessage("didn't return set value").that(resultValue).isEqualTo(fileOffset);
  }

  @Test
  public void testFileOffset() {
    ByteBuffer buffer = ByteBuffer.allocate(100);
    TestView instance = new TestView(buffer);
    long expResult = -1L;
    long result = instance.fileOffset();
    assertWithMessage("default file offset should be -1").that(result).isEqualTo(expResult);
  }

  @Test
  public void testFinish() {
    ByteBuffer buffer = ByteBuffer.allocate(100);
    TestView instance = new TestView(buffer);
    int limit = instance.buffer.limit();
    int pos = instance.buffer.position();
    assertWithMessage("initial limit").that(limit).isEqualTo(100);
    assertWithMessage("initial position").that(pos).isEqualTo(0);
    instance.putInt(1234);
    limit = instance.buffer.limit();
    pos = instance.buffer.position();
    assertWithMessage("limit unchanged").that(limit).isEqualTo(100);
    assertWithMessage("position advanced").that(pos).isEqualTo(4);
    instance.buffer.flip();
    int finishedLimit = instance.buffer.limit();
    int finishedPos = instance.buffer.position();
    assertWithMessage("must set limit to position").that(finishedLimit).isEqualTo(pos);
    assertWithMessage("must set position to 0").that(finishedPos).isEqualTo(0);
  }

  @Test
  public void testWriteTo() throws Exception {
    FileChannel file = fileSystem.getOutputChannel("hello", false);
    byte[] bytes = "hello world".getBytes(UTF_8);
    ByteBuffer buffer = ByteBuffer.wrap(bytes);
    TestView instance = new TestView(buffer);
    int expResult = bytes.length;
    instance.buffer.rewind();
    int result = file.write(instance.buffer);
    file.close();
    assertWithMessage("incorrect number of bytes written").that(result).isEqualTo(expResult);
    byte[] bytesWritten = fileSystem.toByteArray("hello");
    assertWithMessage("incorrect bytes written").that(bytesWritten).isEqualTo(bytes);
  }

  @Test
  public void testGetBytes() {
    int off = 3;
    int len = 5;
    byte[] bytes = "hello world".getBytes(UTF_8);
    ByteBuffer buffer = ByteBuffer.wrap(bytes);
    TestView instance = new TestView(buffer);
    byte[] expResult = "lo wo".getBytes(UTF_8);
    byte[] result = instance.getBytes(off, len);
    assertWithMessage("incorrect bytes returned").that(result).isEqualTo(expResult);
    assertThrows(
        IndexOutOfBoundsException.class, () -> instance.getBytes(bytes.length - len + 1, len));
    assertThrows(IndexOutOfBoundsException.class, () -> instance.getBytes(-1, len));
  }

  @Test
  public void testGetString() {
    int off = 6;
    int len = 5;
    byte[] bytes = "hello world".getBytes(UTF_8);
    ByteBuffer buffer = ByteBuffer.wrap(bytes);
    TestView instance = new TestView(buffer);
    String expResult = "world";
    String result = instance.getString(off, len);
    assertWithMessage("didn't return this").that(result).isEqualTo(expResult);
    assertThrows(IndexOutOfBoundsException.class, () -> instance.getString(off + 1, len));
    assertThrows(IndexOutOfBoundsException.class, () -> instance.getString(-1, len));
  }

  @Test
  public void testByteOrder() {
    byte[] bytes = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    TestView instance = new TestView(ByteBuffer.wrap(bytes));
    int expValue = 0x08070605;
    int value = instance.getInt(4);
    assertWithMessage("Byte order incorrect").that(value).isEqualTo(expValue);
  }

  static class TestView extends View<TestView> {
    TestView(ByteBuffer buffer) {
      super(buffer);
    }

    // Will advance buffer position.
    public void putInt(int value) {
      buffer.putInt(value);
    }

    // Will not advance buffer position.
    public void putInt(int index, int value) {
      buffer.putInt(index, value);
    }

    // Will advance buffer position.
    public int getInt() {
      return buffer.getInt();
    }

    // Will not advance buffer position.
    public int getInt(int index) {
      return buffer.getInt(index);
    }
  }
}
