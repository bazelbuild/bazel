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
package com.google.devtools.build.lib.buildeventstream;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.vfs.Path;
import java.net.URI;
import java.net.URISyntaxException;
import javax.annotation.Nullable;

/**
 * Interface for conversion of paths to URIs.
 */
public interface PathConverter {
  /** An implementation that throws on every call to {@link #apply(Path)}. */
  PathConverter NO_CONVERSION =
      path -> {
        throw new IllegalStateException(
            String.format(
                "Can't convert '%s', as it has not been declared as a referenced artifact of a"
                    + " build event",
                path.getPathString()));
      };

  /** A {@link PathConverter} that returns a path formatted as a URI with a {@code file} scheme. */
  // TODO(ulfjack): Make this a static final field.
  final class FileUriPathConverter implements PathConverter {
    @Override
    public String apply(Path path) {
      Preconditions.checkNotNull(path);
      return pathToUriString(path.getPathString());
    }

    /**
     * Returns the path encoded as an {@link URI}.
     *
     * <p>This concrete implementation returns URIs with "file" as the scheme. For Example: - On
     * Unix the path "/tmp/foo bar.txt" will be encoded as "file:///tmp/foo%20bar.txt". - On Windows
     * the path "C:\Temp\Foo Bar.txt" will be encoded as "file:///C:/Temp/Foo%20Bar.txt"
     *
     * <p>Implementors extending this class for special filesystems will likely need to override
     * this method.
     */
    @VisibleForTesting
    static String pathToUriString(String path) {
      if (!path.startsWith("/")) {
        // On Windows URI's need to start with a '/'. i.e. C:\Foo\Bar would be file:///C:/Foo/Bar
        path = "/" + path;
      }
      try {
        return new URI(
                "file",
                // Needs to be "" instead of null, so that toString() will append "//" after the
                // scheme.
                // We need this for backwards compatibility reasons as some consumers of the BEP are
                // broken.
                "",
                path,
                null,
                null)
            .toString();
      } catch (URISyntaxException e) {
        throw new IllegalStateException(e);
      }
    }
  }

  /**
   * Return the URI corresponding to the given path.
   *
   * <p>This method may return null, in which case the associated {@link BuildEventArtifactUploader}
   * was permanently unable to upload the file. The file should be omitted from the BEP stream.
   *
   * <p>This method may throw {@link IllegalStateException} if it is passed a path that
   * wasn't declared in {@link BuildEvent#referencedLocalFiles()}.
   */
  @Nullable
  String apply(Path path);
}
