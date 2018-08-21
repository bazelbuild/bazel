// Copyright 2018 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableMap;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ListeningExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.buildeventstream.BuildEvent.LocalFile;
import com.google.devtools.build.lib.vfs.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;
import javax.annotation.Nullable;

/** An uploader that simply turns paths into local file URIs. */
class LocalFilesArtifactUploader implements BuildEventArtifactUploader {
  private final ListeningExecutorService uploadExecutor =
      MoreExecutors.listeningDecorator(Executors.newCachedThreadPool());

  @Override
  public ListenableFuture<PathConverter> upload(Map<Path, LocalFile> files) {
    List<ListenableFuture<PathLookupResult>> lookups = new ArrayList<>();
    for (Path path : files.keySet()) {
      lookups.add(uploadExecutor.submit(() -> new PathLookupResult(path, path.isDirectory())));
    }
    return Futures.transform(
        Futures.allAsList(lookups),
        lookupList -> {
          ImmutableMap.Builder<Path, PathLookupResult> pathLookups = ImmutableMap.builder();
          for (PathLookupResult lookup : lookupList) {
            pathLookups.put(lookup.path, lookup);
          }
          return new PathConverterImpl(pathLookups.build());
        },
        MoreExecutors.directExecutor());
  }

  @Override
  public void shutdown() {
    // Intentionally left empty
  }

  private static class PathLookupResult {
    final Path path;
    final boolean isDirectory;

    private PathLookupResult(Path path, boolean isDirectory) {
      this.path = path;
      this.isDirectory = isDirectory;
    }
  }

  private static class PathConverterImpl implements PathConverter {
    private static final FileUriPathConverter FILE_URI_PATH_CONVERTER = new FileUriPathConverter();
    private final Map<Path, PathLookupResult> pathLookups;

    private PathConverterImpl(Map<Path, PathLookupResult> pathLookups) {
      this.pathLookups = pathLookups;
    }

    @Nullable
    @Override
    public String apply(Path path) {
      PathLookupResult result = pathLookups.get(path);
      if (result == null) {
        // We should throw here, the file wasn't declared in BuildEvent#referencedLocalFiles
        return null;
      }
      if (result.isDirectory) {
        return null;
      }
      return FILE_URI_PATH_CONVERTER.apply(result.path);
    }
  }
}
