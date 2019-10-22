// Copyright 2019 The Bazel Authors. All rights reserved.
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
//

package com.google.devtools.build.lib.bazel.rules.ninja.parser;

import com.google.common.util.concurrent.ListeningExecutorService;
import com.google.devtools.build.lib.bazel.rules.ninja.file.CollectingSynchronizedFuture;
import com.google.devtools.build.lib.bazel.rules.ninja.file.GenericParsingException;
import com.google.devtools.build.lib.bazel.rules.ninja.file.NinjaSeparatorPredicate;
import com.google.devtools.build.lib.bazel.rules.ninja.file.ParallelFileProcessing;
import com.google.devtools.build.lib.bazel.rules.ninja.file.ParallelFileProcessing.BlockParameters;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaFirstScanParser.Factory;
import com.google.devtools.build.lib.vfs.Path;
import java.nio.charset.Charset;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

public class NinjaPipeline {
  private final ListeningExecutorService service;
  private final CollectingSynchronizedFuture<
      Set<NinjaFirstScanParser>, GenericParsingException> future;
  private final Factory factory;

  public NinjaPipeline(
      ListeningExecutorService service,
      Charset charset,
      Path basePath) {
    this.service = service;
    future = new CollectingSynchronizedFuture<>(GenericParsingException.class);
    factory = new Factory(basePath, this::scheduleFile, charset);
  }

  public List<NinjaFirstScanParser> pipeline(Path path)
      throws GenericParsingException, InterruptedException {
    scheduleFile(path);
    List<Set<NinjaFirstScanParser>> result = future.getResult();
    // todo work with the values we got from parallel threads, replace variables,
    // todo work with the graph
    // todo for now, just return the result
    return result.stream().flatMap(Set::stream).collect(Collectors.toList());
  }

  private void scheduleFile(Path path) {
    ParallelFileProcessing<NinjaFirstScanParser> fileProcessing = new ParallelFileProcessing<NinjaFirstScanParser>(
        path, new BlockParameters(), factory, service, NinjaSeparatorPredicate.INSTANCE);
    future.add(service.submit(fileProcessing));
  }
}
