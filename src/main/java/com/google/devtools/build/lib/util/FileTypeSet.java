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
package com.google.devtools.build.lib.util;

import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import javax.annotation.concurrent.Immutable;

/** A set of FileTypes for grouped matching. */
@Immutable
@AutoCodec
public class FileTypeSet implements Predicate<String> {
  private final ImmutableSet<FileType> fileTypes;

  /** A set that matches all files. */
  @AutoCodec
  public static final FileTypeSet ANY_FILE =
      new FileTypeSet() {
        @Override
        public String toString() {
          return "any files";
        }

        @Override
        public boolean matches(String filename) {
          return true;
        }

        @Override
        public List<String> getExtensions() {
          return ImmutableList.of();
        }
      };

  /** A predicate that matches no files. */
  @AutoCodec
  public static final FileTypeSet NO_FILE =
      new FileTypeSet(ImmutableList.of()) {
        @Override
        public String toString() {
          return "no files";
        }

        @Override
        public boolean matches(String filename) {
          return false;
        }
      };

  private FileTypeSet() {
    this.fileTypes = null;
  }

  private FileTypeSet(FileType... fileTypes) {
    this.fileTypes = ImmutableSet.copyOf(fileTypes);
  }

  @AutoCodec.Instantiator
  @VisibleForSerialization
  FileTypeSet(Iterable<FileType> fileTypes) {
    this.fileTypes = ImmutableSet.copyOf(fileTypes);
  }

  /**
   * Returns a set that matches only the provided {@code fileTypes}.
   *
   * <p>If {@code fileTypes} is empty, the returned predicate will match no files.
   */
  public static FileTypeSet of(FileType... fileTypes) {
    if (fileTypes.length == 0) {
      return FileTypeSet.NO_FILE;
    } else {
      return new FileTypeSet(fileTypes);
    }
  }

  /**
   * Returns a set that matches only the provided {@code fileTypes}.
   *
   * <p>If {@code fileTypes} is empty, the returned predicate will match no files.
   */
  public static FileTypeSet of(Iterable<FileType> fileTypes) {
    if (Iterables.isEmpty(fileTypes)) {
      return FileTypeSet.NO_FILE;
    } else {
      return new FileTypeSet(fileTypes);
    }
  }

  /** Returns a copy of this {@link FileTypeSet} including the specified `fileTypes`. */
  public FileTypeSet including(FileType... fileTypes) {
    return new FileTypeSet(Iterables.concat(this.fileTypes, Arrays.asList(fileTypes)));
  }

  /** Returns true if the filename can be matched by any FileType in this set. */
  public boolean matches(String path) {
    for (FileType type : fileTypes) {
      if (type.apply(path)) {
        return true;
      }
    }
    return false;
  }

  @VisibleForSerialization
  ImmutableSet<FileType> getFileTypes() {
    return fileTypes;
  }

  /** Returns true if this predicate matches nothing. */
  public boolean isNone() {
    return this == FileTypeSet.NO_FILE;
  }

  @Override
  public boolean apply(String path) {
    return matches(path);
  }

  /** Returns the list of possible file extensions for this file type. Can be empty. */
  public List<String> getExtensions() {
    List<String> extensions = new ArrayList<>();
    for (FileType type : fileTypes) {
      extensions.addAll(type.getExtensions());
    }
    return extensions;
  }

  @Override
  public String toString() {
    return StringUtil.joinEnglishList(getExtensions());
  }
}
