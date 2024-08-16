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
package com.google.devtools.build.android;

import com.beust.jcommander.Parameter;
import com.google.common.base.Preconditions;
import java.lang.annotation.Annotation;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;

/** Utility class for JCommander-based Android options. */
public class AndroidOptionsUtils {

  private AndroidOptionsUtils() {}

  /**
   * Normalize boolean options to use --<flagname>=true or --<flagname>=false syntax.
   *
   * <p>This is useful for JCommander-based options.
   */
  public static String[] normalizeBooleanOptions(Object options, String[] args) {
    List<String> booleanOptions = new ArrayList<>();
    // The normalized arg list will be as long as the original args.
    List<String> normalizedArgs = new ArrayList<>(args.length);
    // Find the list of boolean fields
    for (Field field : options.getClass().getDeclaredFields()) {
      if (field.getType().equals(boolean.class)) {
        // Get the `names` from the annotation of this field.
        // Iterate through the annotations
        for (Annotation annotation : field.getAnnotations()) {
          if (annotation instanceof Parameter) {
            Parameter parameter = (Parameter) annotation;
            for (String name : parameter.names()) {
              // Strip leading dashes from the name and assert that the name starts with --.
              Preconditions.checkState(name.startsWith("--"));
              booleanOptions.add(name.substring(2));
            }
          }
        }
      }
    }

    // Iterate through the args and normalize boolean options with --<flagname> syntax.
    for (String arg : args) {
      for (String booleanOption : booleanOptions) {
        if (arg.equals("--no" + booleanOption)) {
          arg = "--" + booleanOption + "=false";
        } else if (arg.equals("--" + booleanOption)) {
          arg = "--" + booleanOption + "=true";
        }
      }
      normalizedArgs.add(arg);
    }
    return normalizedArgs.toArray(new String[0]);
  }
}
