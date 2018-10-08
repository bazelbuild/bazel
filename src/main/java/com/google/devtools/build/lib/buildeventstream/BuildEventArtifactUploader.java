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

import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.buildeventstream.BuildEvent.LocalFile;
import com.google.devtools.build.lib.vfs.Path;
import java.util.Map;

/** Uploads artifacts referenced by the Build Event Protocol (BEP). */
public interface BuildEventArtifactUploader {
  /**
   * Asynchronously uploads a set of files referenced by the protobuf representation of a {@link
   * BuildEvent}. This method is expected to return quickly.
   *
   * <p>This method must not throw any exceptions.
   *
   * <p>Returns a future to a {@link PathConverter} that must provide a name for each uploaded file
   * as it should appear in the BEP.
   */
  ListenableFuture<PathConverter> upload(Map<Path, LocalFile> files);

  /**
   * Shutdown any resources associated with the uploader.
   */
  void shutdown();
}
