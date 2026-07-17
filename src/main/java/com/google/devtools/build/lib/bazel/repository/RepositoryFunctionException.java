// Copyright 2025 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.bazel.repository;

import static com.google.common.base.Preconditions.checkState;

import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.repository.ExternalPackageException;
import com.google.devtools.build.lib.skyframe.AlreadyReportedException;
import com.google.devtools.build.skyframe.SkyFunctionException;
import java.io.IOException;
import net.starlark.java.eval.EvalException;

/**
 * Exception thrown when something goes wrong accessing a remote repository.
 *
 * <p>This exception should be used by child classes to limit the types of exceptions {@link
 * RepositoryFetchFunction} has to know how to catch.
 */
public class RepositoryFunctionException extends SkyFunctionException {

  /** Error reading or writing to the filesystem. */
  public RepositoryFunctionException(IOException cause, Transience transience) {
    super(cause, transience);
  }

  public RepositoryFunctionException(EvalException cause, Transience transience) {
    super(cause, transience);
  }

  public RepositoryFunctionException(
      AlreadyReportedRepositoryAccessException cause, Transience transience) {
    super(cause, transience);
  }

  /**
   * Encapsulates the exceptions that arise when accessing a repository. Error reporting should ONLY
   * be handled in {@link RepositoryFetchFunction}.
   */
  public static class AlreadyReportedRepositoryAccessException extends AlreadyReportedException {
    public AlreadyReportedRepositoryAccessException(Exception e) {
      super(e.getMessage(), e.getCause());
      checkState(
          e instanceof NoSuchPackageException
              || e instanceof IOException
              || e instanceof EvalException
              || e instanceof ExternalPackageException,
          e);
    }
  }
}
