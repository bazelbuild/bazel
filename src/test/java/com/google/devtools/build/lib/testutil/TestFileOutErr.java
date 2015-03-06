// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.testutil;

import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.util.io.RecordingOutErr;
import com.google.devtools.build.lib.vfs.Path;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;

/**
 * An implementation of the FileOutErr that doesn't use a file.
 * This is useful for tests, as they often test the action directly
 * and would otherwise have to create files on the vfs.
 */
public class TestFileOutErr extends FileOutErr {

  RecordingOutErr recorder;

  public TestFileOutErr(TestFileOutErr arg) {
    this(arg.getOutputStream(), arg.getErrorStream());
  }

  public TestFileOutErr() {
    this(new ByteArrayOutputStream(), new ByteArrayOutputStream());
  }

  public TestFileOutErr(ByteArrayOutputStream stream) {
    super(null, null); // This is a pretty brutal overloading - We're just inheriting for the type.
    recorder = new RecordingOutErr(stream, stream);
  }

  public TestFileOutErr(ByteArrayOutputStream stream1, ByteArrayOutputStream stream2) {
    super(null, null); // This is a pretty brutal overloading - We're just inheriting for the type.
    recorder = new RecordingOutErr(stream1, stream2);
  }


  @Override
  public Path getOutputFile() {
    return null;
  }

  @Override
  public Path getErrorFile() {
    return null;
  }

  @Override
  public ByteArrayOutputStream getOutputStream() {
    return recorder.getOutputStream();
  }

  @Override
  public ByteArrayOutputStream getErrorStream() {
    return recorder.getErrorStream();
  }

  @Override
  public void printOut(String s) {
    recorder.printOut(s);
  }

  @Override
  public void printErr(String s) {
    recorder.printErr(s);
  }

  @Override
  public String toString() {
    return recorder.toString();
  }

  @Override
  public boolean hasRecordedOutput() {
    return recorder.hasRecordedOutput();
  }

  @Override
  public String outAsLatin1() {
    return recorder.outAsLatin1();
  }

  @Override
  public String errAsLatin1() {
    return recorder.errAsLatin1();
  }

  @Override
  public void dumpOutAsLatin1(OutputStream out) {
    try {
      recorder.getOutputStream().writeTo(out);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  @Override
  public void dumpErrAsLatin1(OutputStream out) {
    try {
      recorder.getErrorStream().writeTo(out);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  public String getRecordedOutput() {
    return recorder.outAsLatin1() + recorder.errAsLatin1();
  }

  public void reset() {
    recorder.reset();
  }
}
