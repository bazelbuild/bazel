/*
 * Copyright 2014-present Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may
 * not use this file except in compliance with the License. You may obtain
 * a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

package com.facebook.buck.apple;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.ImmutableSet;

/**
 * File types used in Apple targets.
 */
public final class FileTypes {

  // Utility class. Do not instantiate.
  private FileTypes() { }

  /**
   * Map of file extension to Apple UTI (Uniform Type Identifier).
   */
  public static final ImmutableMap<String, String> FILE_EXTENSION_TO_UTI =
      ImmutableMap.<String, String>builder()
          .put("a", "archive.ar")
          .put("app", "wrapper.application")
          .put("appex", "wrapper.app-extension")
          .put("bdic", "file")
          .put("bin", "archive.macbinary")
          .put("bmp", "image.bmp")
          .put("bundle", "wrapper.cfbundle")
          .put("c", "sourcecode.c.c")
          .put("cc", "sourcecode.cpp.cpp")
          .put("cpp", "sourcecode.cpp.cpp")
          .put("css", "text.css")
          .put("cxx", "sourcecode.cpp.cpp")
          .put("dart", "sourcecode")
          .put("dylib", "compiled.mach-o.dylib")
          .put("exp", "sourcecode.exports")
          .put("framework", "wrapper.framework")
          .put("fsh", "sourcecode.glsl")
          .put("gyp", "sourcecode")
          .put("gypi", "text")
          .put("h", "sourcecode.c.h")
          .put("hxx", "sourcecode.cpp.h")
          .put("icns", "image.icns")
          .put("java", "sourcecode.java")
          .put("jar", "archive.jar")
          .put("jpeg", "image.jpeg")
          .put("jpg", "image.jpeg")
          .put("js", "sourcecode.javascript")
          .put("json", "text.json")
          .put("m", "sourcecode.c.objc")
          .put("mm", "sourcecode.cpp.objcpp")
          .put("nib", "wrapper.nib")
          .put("o", "compiled.mach-o.objfile")
          .put("octest", "wrapper.cfbundle")
          .put("pdf", "image.pdf")
          .put("pl", "text.script.perl")
          .put("plist", "text.plist.xml")
          .put("pm", "text.script.perl")
          .put("png", "image.png")
          .put("proto", "text")
          .put("py", "text.script.python")
          .put("r", "sourcecode.rez")
          .put("rez", "sourcecode.rez")
          .put("rtf", "text.rtf")
          .put("s", "sourcecode.asm")
          .put("storyboard", "file.storyboard")
          .put("strings", "text.plist.strings")
          .put("tif", "image.tiff")
          .put("tiff", "image.tiff")
          .put("tcc", "sourcecode.cpp.cpp")
          .put("ttf", "file")
          .put("vsh", "sourcecode.glsl")
          .put("xcassets", "folder.assetcatalog")
          .put("xcconfig", "text.xcconfig")
          .put("xcodeproj", "wrapper.pb-project")
          .put("xcdatamodel", "wrapper.xcdatamodel")
          .put("xcdatamodeld", "wrapper.xcdatamodeld")
          .put("xctest", "wrapper.cfbundle")
          .put("xib", "file.xib")
          .put("y", "sourcecode.yacc")
          .put("zip", "archive.zip")
          .build();

  /**
   * Set of UTIs which only work as "lastKnownFileType" and not "explicitFileType"
   * in a PBXFileReference.
   *
   * Yes, really. Because Xcode.
   */
  public static final ImmutableSet<String> EXPLICIT_FILE_TYPE_BROKEN_UTIS =
    ImmutableSet.of("file.xib");

  /**
   * Multimap of Apple UTI (Uniform Type Identifier) to file extension(s).
   */
  public static final ImmutableMultimap<String, String> UTI_TO_FILE_EXTENSIONS;

  static {
    // Invert the map of (file extension -> UTI) pairs to
    // (UTI -> [file extension 1, ...]) pairs.
    ImmutableMultimap.Builder<String, String> builder = ImmutableMultimap.builder();
    for (ImmutableMap.Entry<String, String> entry : FILE_EXTENSION_TO_UTI.entrySet()) {
      builder.put(entry.getValue(), entry.getKey());
    }
    UTI_TO_FILE_EXTENSIONS = builder.build();
  }
}
