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

package com.google.devtools.build.lib.query2.query.output;

import com.google.common.collect.Sets;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.query2.engine.OutputFormatterCallback;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment;
import com.google.devtools.build.lib.query2.engine.SynchronizedDelegatingOutputFormatterCallback;
import com.google.devtools.build.lib.query2.engine.ThreadSafeOutputFormatterCallback;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Set;

/**
 * An output formatter that prints the names of the packages of the target
 * set, in lexicographical order without duplicates.
 */
class PackageOutputFormatter extends AbstractUnorderedFormatter {

  @Override
  public String getName() {
    return "package";
  }

  @Override
  public OutputFormatterCallback<Target> createPostFactoStreamCallback(
      OutputStream out, final QueryOptions options) {
    return new TextOutputFormatterCallback<Target>(out) {
      private final Set<String> packageNames = Sets.newTreeSet();

      @Override
      public void processOutput(Iterable<Target> partialResult) {

        for (Target target : partialResult) {
          packageNames.add(target.getLabel().getPackageIdentifier().toString());
        }
      }

      @Override
      public void close(boolean failFast) throws IOException {
        if (!failFast) {
          final String lineTerm = options.getLineTerminator();
          for (String packageName : packageNames) {
            writer.append(packageName).append(lineTerm);
          }
        }
        super.close(failFast);
      }
    };
  }

  @Override
  public ThreadSafeOutputFormatterCallback<Target> createStreamCallback(
      OutputStream out, QueryOptions options, QueryEnvironment<?> env) {
    return new SynchronizedDelegatingOutputFormatterCallback<>(
        createPostFactoStreamCallback(out, options));
  }
}