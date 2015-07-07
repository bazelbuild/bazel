// Copyright 2015 Google Inc. All rights reserved.
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

import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.fail;

import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;

/**
 * Unit tests for {@link View}.
 */
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
    assertEquals("must assume buffer ownership", fromBuf, fromView);
    int posBuf = buffer.position();
    int posView = instance.buffer.position();
    assertEquals("must assume buffer ownership", posBuf, posView);
  }

  @Test
  public void testAt() {
    long fileOffset = 0L;
    ByteBuffer buffer = ByteBuffer.allocate(100);
    TestView instance = new TestView(buffer);
    View<TestView> result = instance.at(fileOffset);
    assertSame("didn't return this", instance, result);

    long resultValue = instance.fileOffset();
    assertEquals("didn't return set value", fileOffset, resultValue);
  }

  @Test
  public void testFileOffset() {
    ByteBuffer buffer = ByteBuffer.allocate(100);
    TestView instance = new TestView(buffer);
    long expResult = -1L;
    long result = instance.fileOffset();
    assertEquals("default file offset should be -1", expResult, result);
  }

  @Test
  public void testFinish() {
    ByteBuffer buffer = ByteBuffer.allocate(100);
    TestView instance = new TestView(buffer);
    int limit = instance.buffer.limit();
    int pos = instance.buffer.position();
    assertEquals("initial limit", 100, limit);
    assertEquals("initial position", 0, pos);
    instance.putInt(1234);
    limit = instance.buffer.limit();
    pos = instance.buffer.position();
    assertEquals("limit unchanged", 100, limit);
    assertEquals("position advanced", 4, pos);
    instance.buffer.flip();
    int finishedLimit = instance.buffer.limit();
    int finishedPos = instance.buffer.position();
    assertEquals("must set limit to position", pos, finishedLimit);
    assertEquals("must set position to 0", 0, finishedPos);
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
    assertEquals("incorrect number of bytes written", expResult, result);
    byte[] bytesWritten = fileSystem.toByteArray("hello");
    Assert.assertArrayEquals("incorrect bytes written", bytes, bytesWritten);
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
    assertArrayEquals("incorrect bytes returned", expResult, result);
    try {
      instance.getBytes(bytes.length - len + 1, len);
      fail("expected Exception");
    } catch (IndexOutOfBoundsException ex) {
      // expected
    }
    try {
      instance.getBytes(-1, len);
      fail("expected Exception");
    } catch (IndexOutOfBoundsException ex) {
      // expected
    }
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
    assertEquals("didn't return this", expResult, result);
    try {
      instance.getString(off + 1, len);
      fail("expected Exception");
    } catch (IndexOutOfBoundsException ex) {
      // expected
    }
    try {
      instance.getString(-1, len);
      fail("expected Exception");
    } catch (IndexOutOfBoundsException ex) {
      // expected
    }
  }

  @Test
  public void testByteOrder() {
    byte[] bytes = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    TestView instance = new TestView(ByteBuffer.wrap(bytes));
    int expValue = 0x08070605;
    int value = instance.getInt(4);
    assertEquals("Byte order incorrect", expValue, value);
  }

  static class TestView extends View<TestView> {
    TestView(ByteBuffer buffer) {
      super(buffer);
    }

    // Will advance buffer position
    public void putInt(int value) {
      buffer.putInt(value);
    }

    // Will advance buffer position
    public int getInt() {
      return buffer.getInt();
    }

    // will not advance buffer position
    public void putInt(int index, int value) {
      buffer.putInt(index, value);
    }

    // will not advance buffer position
    public int getInt(int index) {
      return buffer.getInt(index);
    }
  }
}
