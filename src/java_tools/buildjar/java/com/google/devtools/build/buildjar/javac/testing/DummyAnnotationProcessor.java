// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.buildjar.javac.testing;

import java.io.IOException;
import java.io.Writer;
import java.util.Set;
import javax.annotation.processing.AbstractProcessor;
import javax.annotation.processing.Filer;
import javax.annotation.processing.RoundEnvironment;
import javax.annotation.processing.SupportedAnnotationTypes;
import javax.lang.model.SourceVersion;
import javax.lang.model.element.TypeElement;
import javax.tools.JavaFileObject;

/**
 * Annotation processor used for testing correct propagation of factories between Contexts.
 *
 * <p>It generates new source files for a given number of times, each source referring to the next
 * (so that compilation will fail if the generated sources aren't all included in the same
 * compilation round).
 *
 * <p>Being a universal processor, it does not claim any annotations and {@link #process} always
 * returns {@code false}, in order to interact correctly with other annotation processors.
 */
@SupportedAnnotationTypes("*")
public class DummyAnnotationProcessor extends AbstractProcessor {

  private final int maxRounds;
  private int round;

  /** By default, generates a new file for one round of annotation processing. */
  public DummyAnnotationProcessor() {
    this(1);
  }

  /** Generates new files for given number of annotation processing rounds. */
  public DummyAnnotationProcessor(int rounds) {
    round = 0;
    this.maxRounds = rounds;
  }

  @Override
  public SourceVersion getSupportedSourceVersion() {
    return SourceVersion.latest();
  }

  @Override
  public boolean process(Set<? extends TypeElement> annotations, RoundEnvironment roundEnv) {
    round++;
    if (round <= maxRounds) {
      String curr = "p.Gen" + round;
      String next = "p.Gen" + (round + 1);
      StringBuilder text = new StringBuilder();
      text.append("package p;\n");
      text.append("public class Gen").append(round).append(" {\n");
      if (round < maxRounds) {
        text.append("    ").append(next).append(" x;\n");
      }
      text.append("}\n");

      try {
        Filer filer = processingEnv.getFiler();
        JavaFileObject fo = filer.createSourceFile(curr);
        Writer out = fo.openWriter();
        try {
          out.write(text.toString());
        } finally {
          out.close();
        }
      } catch (IOException e) {
        throw new Error(e);
      }
    }

    return false;
  }
}
