// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.runtime;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.devtools.build.lib.buildtool.BuildResult.BuildToolLogCollection;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.io.IOException;
import java.io.OutputStream;
import javax.annotation.Nullable;

/** Used when instrumentation output is written to a local file. */
final class LocalInstrumentationOutput implements InstrumentationOutput {
  private final Path path;
  private final String name;
  @Nullable private final String convenienceName;

  LocalInstrumentationOutput(String name, Path path, String convenienceName) {
    this.name = name;
    this.path = path;
    this.convenienceName = convenienceName;
  }

  @Override
  public void publish(BuildToolLogCollection buildToolLogCollection) {
    buildToolLogCollection.addLocalFile(name, path);
  }

  public void makeConvenienceLink() throws IOException {
    if (convenienceName != null) {
      var link = path.getParentDirectory().getChild(convenienceName);
      link.delete();
      link.createSymbolicLink(PathFragment.create(path.getBaseName()));
    }
  }

  @Override
  public OutputStream createOutputStream() throws IOException {
    return path.getOutputStream();
  }

  public OutputStream createOutputStream(boolean append, boolean internal) throws IOException {
    return path.getOutputStream(append, internal);
  }

  /** Builder for {@link LocalInstrumentationOutput}. */
  public static class Builder implements InstrumentationOutputBuilder {
    private String name;
    private Path path;
    private String convenienceName;

    @CanIgnoreReturnValue
    @Override
    public Builder setName(String name) {
      this.name = name;
      return this;
    }

    /** Sets the path to the local {@link InstrumentationOutput}. */
    @CanIgnoreReturnValue
    public Builder setPath(Path path) {
      this.path = path;
      return this;
    }

    /**
     * Set the convenience name for the instrumentation output. A symlink at <code>name</code> will
     * be created pointing to the output when {@link
     * LocalInstrumentationOutput#makeConvenienceLink()} is called.
     */
    @CanIgnoreReturnValue
    public Builder setConvenienceName(String name) {
      this.convenienceName = name;
      return this;
    }

    @Override
    public InstrumentationOutput build() {
      return new LocalInstrumentationOutput(
          checkNotNull(name, "Cannot create LocalInstrumentationOutputBuilder without name"),
          checkNotNull(path, "Cannot create LocalInstrumentationOutputBuilder without path"),
          convenienceName);
    }
  }
}
