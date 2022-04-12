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

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.query2.engine.OutputFormatterCallback;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment;
import com.google.devtools.build.lib.query2.engine.SynchronizedDelegatingOutputFormatterCallback;
import com.google.devtools.build.lib.query2.engine.ThreadSafeOutputFormatterCallback;
import java.io.IOException;
import java.io.OutputStream;

/**
 * An output formatter that prints the labels of the resulting target set in
 * topological order, optionally with the target's kind.
 */
class LabelOutputFormatter extends AbstractUnorderedFormatter {

  private final boolean showKind;

  LabelOutputFormatter(boolean showKind) {
    this.showKind = showKind;
  }

  @Override
  public String getName() {
    return showKind ? "label_kind" : "label";
  }

  @Override
  public OutputFormatterCallback<Target> createPostFactoStreamCallback(
      OutputStream out, final QueryOptions options) {
    return new TextOutputFormatterCallback<Target>(out) {
      @Override
      public void processOutput(Iterable<Target> partialResult) throws IOException {
        String lineTerm = options.getLineTerminator();
        for (Target target : partialResult) {
          if (showKind) {
            writer.append(target.getTargetKind());
            writer.append(' ');
          }
          Label label = target.getLabel();
          writer.append(label.getCanonicalForm()).append(lineTerm);
        }
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