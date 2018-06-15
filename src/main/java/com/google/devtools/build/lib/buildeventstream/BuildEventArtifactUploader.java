// Copyright 2018 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.lib.buildeventstream;

import com.google.devtools.build.lib.buildeventstream.PathConverter.FileUriPathConverter;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.util.Set;

/** Uploads artifacts referenced by the Build Event Protocol (BEP). */
public interface BuildEventArtifactUploader {
  public static final BuildEventArtifactUploader LOCAL_FILES_UPLOADER =
      new BuildEventArtifactUploader() {
    @Override
    public PathConverter upload(Set<Path> files) {
      return new FileUriPathConverter();
    }
  };

  /**
   * Uploads a set of files referenced by the protobuf representation of a {@link BuildEvent}.
   *
   * <p>Returns a {@link PathConverter} that must provide a name for each uploaded file as it should
   * appear in the BEP.
   */
  PathConverter upload(Set<Path> files) throws IOException, InterruptedException;
}
