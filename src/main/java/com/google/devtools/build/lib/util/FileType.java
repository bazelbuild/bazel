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

import com.google.common.base.Function;
import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import javax.annotation.concurrent.Immutable;

/**
 * A base class for FileType matchers.
 */
@Immutable
public abstract class FileType implements Predicate<String> {
  // A special file type
  public static final FileType NO_EXTENSION = new FileType() {
      @Override
      public boolean apply(String filename) {
        return filename.lastIndexOf('.') == -1;
      }
  };

  public static FileType of(final String ext) {
    return new FileType() {
      @Override
      public boolean apply(String filename) {
        return filename.endsWith(ext);
      }
      @Override
      public List<String> getExtensions() {
        return ImmutableList.of(ext);
      }
    };
  }

  public static FileType of(final Iterable<String> extensions) {
    return new FileType() {
      @Override
      public boolean apply(String filename) {
        for (String ext : extensions) {
          if (filename.endsWith(ext)) {
            return true;
          }
        }
        return false;
      }
      @Override
      public List<String> getExtensions() {
        return ImmutableList.copyOf(extensions);
      }
    };
  }

  public static FileType of(final String... extensions) {
    return of(Arrays.asList(extensions));
  }

  @Override
  public String toString() {
    return getExtensions().toString();
  }

  /**
   * Returns true if the the filename matches. The filename should be a basename (the filename
   * component without a path) for performance reasons.
   */
  @Override
  public abstract boolean apply(String filename);

  /**
   * Get a list of filename extensions this matcher handles. The first entry in the list (if
   * available) is the primary extension that code can use to construct output file names.
   * The list can be empty for some matchers.
   *
   * @return a list of filename extensions
   */
  public List<String> getExtensions() {
    return ImmutableList.of();
  }

  /** Return true if a file name is matched by the FileType */
  public boolean matches(String filename) {
    int slashIndex = filename.lastIndexOf('/');
    if (slashIndex != -1) {
      filename = filename.substring(slashIndex + 1);
    }
    return apply(filename);
  }

  /** Return true if a file referred by path is matched by the FileType */
  public boolean matches(Path path) {
    return apply(path.getBaseName());
  }

  /** Return true if a file referred by fragment is matched by the FileType */
  public boolean matches(PathFragment fragment) {
    return apply(fragment.getBaseName());
  }

  // Check FileTypes

  /**
   * An interface for entities that have a filename.
   */
  public interface HasFilename {
    /**
     * Returns the filename of this entity.
     */
    String getFilename();
  }

  public static final Function<HasFilename, String> TO_FILENAME =
      new Function<HasFilename, String>() {
        @Override
        public String apply(HasFilename input) {
          return input.getFilename();
        }
      };

  /**
   * Checks whether an Iterable<? extends HasFileType> contains any of the specified file types.
   *
   * <p>At least one FileType must be specified.
   */
  public static <T extends HasFilename> boolean contains(final Iterable<T> items,
      FileType... fileTypes) {
    Preconditions.checkState(fileTypes.length > 0, "Must specify at least one file type");
    final FileTypeSet fileTypeSet = FileTypeSet.of(fileTypes);
    for (T item : items)  {
      if (fileTypeSet.matches(item.getFilename())) {
        return true;
      }
    }
    return false;
  }

  /**
   * Checks whether a HasFileType is any of the specified file types.
   *
   * <p>At least one FileType must be specified.
   */
  public static <T extends HasFilename> boolean contains(T item, FileType... fileTypes) {
    return FileTypeSet.of(fileTypes).matches(item.getFilename());
  }


  private static <T extends HasFilename> Predicate<T> typeMatchingPredicateFor(
      final FileType matchingType) {
    return new Predicate<T>() {
      @Override
      public boolean apply(T item) {
        return matchingType.matches(item.getFilename());
      }
    };
  }

  private static <T extends HasFilename> Predicate<T> typeMatchingPredicateFor(
      final FileTypeSet matchingTypes) {
    return new Predicate<T>() {
      @Override
      public boolean apply(T item) {
        return matchingTypes.matches(item.getFilename());
      }
    };
  }

  private static <T extends HasFilename> Predicate<T> typeMatchingPredicateFrom(
      final Predicate<String> fileTypePredicate) {
    return new Predicate<T>() {
      @Override
      public boolean apply(T item) {
        return fileTypePredicate.apply(item.getFilename());
      }
    };
  }

  /**
   * A filter for Iterable<? extends HasFileType> that returns only those whose FileType matches the
   * specified Predicate.
   */
  public static <T extends HasFilename> Iterable<T> filter(final Iterable<T> items,
      final Predicate<String> predicate) {
    return Iterables.filter(items, typeMatchingPredicateFrom(predicate));
  }

  /**
   * A filter for Iterable<? extends HasFileType> that returns only those of the specified file
   * types.
   */
  public static <T extends HasFilename> Iterable<T> filter(final Iterable<T> items,
      FileType... fileTypes) {
    return filter(items, FileTypeSet.of(fileTypes));
  }

  /**
   * A filter for Iterable<? extends HasFileType> that returns only those of the specified file
   * types.
   */
  public static <T extends HasFilename> Iterable<T> filter(final Iterable<T> items,
      FileTypeSet fileTypes) {
    return Iterables.filter(items, typeMatchingPredicateFor(fileTypes));
  }

  /**
   * A filter for Iterable<? extends HasFileType> that returns only those of the specified file
   * type.
   */
  public static <T extends HasFilename> Iterable<T> filter(final Iterable<T> items,
      FileType fileType) {
    return Iterables.filter(items, typeMatchingPredicateFor(fileType));
  }

  /**
   * A filter for Iterable<? extends HasFileType> that returns everything except the specified file
   * type.
   */
  public static <T extends HasFilename> Iterable<T> except(final Iterable<T> items,
      FileType fileType) {
    return Iterables.filter(items, Predicates.not(typeMatchingPredicateFor(fileType)));
  }


  /**
   * A filter for List<? extends HasFileType> that returns only those of the specified file types.
   * The result is a mutable list, computed eagerly; see {@link #filter} for a lazy variant.
   */
  public static <T extends HasFilename> List<T> filterList(final Iterable<T> items,
      FileType... fileTypes) {
    if (fileTypes.length > 0) {
      return filterList(items, FileTypeSet.of(fileTypes));
    } else {
      return new ArrayList<>();
    }
  }

  /**
   * A filter for List<? extends HasFileType> that returns only those of the specified file type.
   * The result is a mutable list, computed eagerly.
   */
  public static <T extends HasFilename> List<T> filterList(final Iterable<T> items,
      final FileType fileType) {
    List<T> result = new ArrayList<>();
    for (T item : items)  {
      if (fileType.matches(item.getFilename())) {
        result.add(item);
      }
    }
    return result;
  }

  /**
   * A filter for List<? extends HasFileType> that returns only those of the specified file types.
   * The result is a mutable list, computed eagerly.
   */
  public static <T extends HasFilename> List<T> filterList(final Iterable<T> items,
      final FileTypeSet fileTypeSet) {
    List<T> result = new ArrayList<>();
    for (T item : items)  {
      if (fileTypeSet.matches(item.getFilename())) {
        result.add(item);
      }
    }
    return result;
  }
}
