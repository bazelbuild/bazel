// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.xcode.common;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.xcode.util.Equaling;
import com.google.devtools.build.xcode.util.Value;

import java.nio.file.Path;
import java.util.List;

/**
 * Represents the path to an xcodeproj directory. Contains utilities for getting related
 * information, including the project.pbxproj file and the project <em>name</em>, which is the
 * .xcodeproj directory name without the ".xcodeproj" extension.
 *
 * @param <T> The type of the backing path, such as {@link java.nio.file.Path}.
 */
public class XcodeprojPath<T extends Comparable<T>>
    extends Value<XcodeprojPath<T>>
    implements Comparable<XcodeprojPath<T>> {
  public static final String PBXPROJ_FILE_NAME = "project.pbxproj";
  public static final String XCODEPROJ_DIRECTORY_SUFFIX = ".xcodeproj";

  /**
   * An object that knows how to create {@code XcodeprojPath}s from paths of some other type.
   */
  public static class Converter<T extends Comparable<T>> {
    private final PathTransformer<T> transformer;

    public Converter(PathTransformer<T> transformer) {
      this.transformer = checkNotNull(transformer);
    }

    /**
     * Converts a path to an XcodeprojPath. The given path may point to the {@code project.pbxproj}
     * file or the {@code *.xcodeproj} directory.
     */
    public XcodeprojPath<T> fromPath(T path) {
      if (Equaling.of(PBXPROJ_FILE_NAME, transformer.name(path))) {
        path = transformer.parent(path);
      }
      return new XcodeprojPath<T>(path, transformer);
    }

    /**
     * Converts normal paths to {@code ProjectFilePath}s using {@link #fromPath(Comparable)}.
     */
    public List<XcodeprojPath<T>> fromPaths(Iterable<? extends T> pbxprojFiles) {
      ImmutableList.Builder<XcodeprojPath<T>> result = new ImmutableList.Builder<>();
      for (T pbxprojFile : pbxprojFiles) {
        result.add(fromPath(pbxprojFile));
      }
      return result.build();
    }
  }

  private final T xcodeprojDirectory;
  private final PathTransformer<T> transformer;

  public XcodeprojPath(T xcodeprojDirectory, PathTransformer<T> transformer) {
    super(xcodeprojDirectory);
    checkArgument(transformer.name(xcodeprojDirectory).endsWith(XCODEPROJ_DIRECTORY_SUFFIX),
        "xcodeprojDirectory should end with %s, but it is '%s'", XCODEPROJ_DIRECTORY_SUFFIX,
        xcodeprojDirectory);
    this.xcodeprojDirectory = xcodeprojDirectory;
    this.transformer = transformer;
  }

  /** Returns a converter which works for the Java {@code Path} class. */
  public static Converter<Path> converter() {
    return new Converter<>(PathTransformer.FOR_JAVA_PATH);
  }

  public final T getXcodeprojDirectory() {
    return xcodeprojDirectory;
  }

  public final T getPbxprojFile() {
    return transformer.join(xcodeprojDirectory, PBXPROJ_FILE_NAME);
  }

  /**
   * Returns the package or directory in which the project is located. For instance, if the project
   * file is {@code /foo/bar/App.xcodeproj/project.pbxproj}, then this method returns
   * {@code /foo/bar}.
   */
  public final T getXcodeprojContainerDir() {
    return transformer.parent(xcodeprojDirectory);
  }

  /**
   * Returns the name of the xcodeproj directory without the {@code .xcodeproj} extension or the
   * containing directory. For instance, for an xcodeproj directory of
   * {@code /client/foo.xcodeproj}, this method returns {@code "foo"}.
   */
  public final String getProjectName() {
    String pathStr = transformer.name(xcodeprojDirectory);
    return pathStr.substring(0, pathStr.length() - XCODEPROJ_DIRECTORY_SUFFIX.length());
  }

  @Override
  public final int compareTo(XcodeprojPath<T> o) {
    return getXcodeprojDirectory().compareTo(o.getXcodeprojDirectory());
  }
}
