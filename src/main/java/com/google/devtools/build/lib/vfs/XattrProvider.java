// Copyright 2022 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.vfs;

import java.io.IOException;

/**
 * Can perform xattrs on {@link Path} objects, including the common {@link Path#getFastDigest}. Some
 * extensions to Bazel may allow this data to be retrieved in a different way than just accessing
 * the xattr on the filesystem: they may do bulk filesystem operations for a path/directory at once.
 *
 * <p>Warning! these methods are called both *before* and *after* an action executes on its output
 * files. Thus, implementations must not cache the value for output files or other files like those
 * in external repositories that are generated during the build. An implementation is free to cache
 * it for immutable source files for the duration of the build.
 */
public interface XattrProvider {
  default byte[] getFastDigest(Path path) throws IOException {
    return path.getFastDigest();
  }

  default byte[] getxattr(Path path, String xattrName) throws IOException {
    return path.getxattr(xattrName);
  }

  default byte[] getxattr(Path path, String xattrName, Symlinks followSymlinks) throws IOException {
    return path.getxattr(xattrName, followSymlinks);
  }

  /** Delegates to another {@link XattrProvider}. */
  class DelegatingXattrProvider implements XattrProvider {
    private final XattrProvider delegate;

    public DelegatingXattrProvider(XattrProvider delegate) {
      this.delegate = delegate;
    }

    @Override
    public byte[] getFastDigest(Path path) throws IOException {
      return delegate.getFastDigest(path);
    }

    @Override
    public byte[] getxattr(Path path, String xattrName) throws IOException {
      return delegate.getxattr(path, xattrName);
    }

    @Override
    public byte[] getxattr(Path path, String xattrName, Symlinks followSymlinks)
        throws IOException {
      return delegate.getxattr(path, xattrName, followSymlinks);
    }
  }
}
