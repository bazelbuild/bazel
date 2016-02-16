// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.java.turbine.javac;

import com.sun.tools.javac.api.ClientCodeWrapper.Trusted;
import com.sun.tools.javac.file.JavacFileManager;
import com.sun.tools.javac.util.Context;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URI;
import java.nio.ByteBuffer;
import java.nio.charset.CharsetDecoder;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.util.Map;

import javax.tools.FileObject;
import javax.tools.JavaFileManager;
import javax.tools.JavaFileObject;
import javax.tools.SimpleJavaFileObject;

/** A {@link JavacFileManager} that collects output into a zipfile. */
@Trusted
public class ZipOutputFileManager extends JavacFileManager {

  private final Map<String, OutputFileObject> files;

  public ZipOutputFileManager(Context context, Map<String, OutputFileObject> files) {
    super(context, true, StandardCharsets.UTF_8);
    this.files = files;
  }

  /**
   * Returns true if the file manager owns this location; otherwise it delegates to the underlying
   * implementation.
   */
  private boolean ownedLocation(Location location) {
    return location.isOutputLocation();
  }

  @Override
  public boolean hasLocation(Location location) {
    return ownedLocation(location) || super.hasLocation(location);
  }

  private OutputFileObject getOutput(String name, JavaFileObject.Kind kind, Location location) {
    if (files.containsKey(name)) {
      return files.get(name);
    }
    OutputFileObject result = new OutputFileObject(name, kind, location);
    files.put(name, result);
    return result;
  }

  @Override
  public JavaFileObject getJavaFileForOutput(
      Location location, String className, JavaFileObject.Kind kind, FileObject sibling)
      throws IOException {
    if (!ownedLocation(location)) {
      return super.getJavaFileForOutput(location, className, kind, sibling);
    }
    // The classname parameter will be something like
    // "com.google.common.base.Flag$String"; nested classes are delimited with
    // dollar signs, so the following transformation works as intended.
    return getOutput(className.replace('.', '/') + kind.extension, kind, location);
  }

  @Override
  public FileObject getFileForOutput(
      Location location, String packageName, String relativeName, FileObject sibling)
      throws IOException {
    if (!ownedLocation(location)) {
      return super.getFileForOutput(location, packageName, relativeName, sibling);
    }
    String path = "";
    if (packageName != null && !packageName.isEmpty()) {
      path = packageName.replace('.', '/') + '/';
    }
    path += relativeName;
    return getOutput(path, JavaFileObject.Kind.OTHER, location);
  }

  @Override
  public boolean isSameFile(FileObject a, FileObject b) {
    boolean at = a instanceof OutputFileObject;
    boolean bt = b instanceof OutputFileObject;
    if (at || bt) {
      if (at ^ bt) {
        return false;
      }
      return ((OutputFileObject) a).toUri().equals(((OutputFileObject) b).toUri());
    }
    return super.isSameFile(a, b);
  }

  /** A {@link JavaFileObject} that accumulates output in memory. */
  public static class OutputFileObject extends SimpleJavaFileObject {

    public final Location location;

    private final ByteArrayOutputStream buffer = new ByteArrayOutputStream();

    public OutputFileObject(String name, Kind kind, Location location) {
      super(URI.create("outputbuffer://" + name), kind);
      this.location = location;
    }

    @Override
    public OutputStream openOutputStream() {
      return buffer;
    }

    @Override
    public InputStream openInputStream() throws IOException {
      return new ByteArrayInputStream(asBytes());
    }

    @Override
    public CharSequence getCharContent(boolean ignoreEncodingErrors) throws IOException {
      CodingErrorAction errorAction =
          ignoreEncodingErrors ? CodingErrorAction.IGNORE : CodingErrorAction.REPORT;
      CharsetDecoder decoder =
          StandardCharsets.UTF_8
              .newDecoder()
              .onUnmappableCharacter(errorAction)
              .onMalformedInput(errorAction);
      return decoder.decode(ByteBuffer.wrap(asBytes()));
    }

    public byte[] asBytes() {
      return buffer.toByteArray();
    }
  }

  public static void preRegister(Context context, final Map<String, OutputFileObject> files) {
    context.put(
        JavaFileManager.class,
        new Context.Factory<JavaFileManager>() {
          @Override
          public JavaFileManager make(Context c) {
            return new ZipOutputFileManager(c, files);
          }
        });
  }
}
