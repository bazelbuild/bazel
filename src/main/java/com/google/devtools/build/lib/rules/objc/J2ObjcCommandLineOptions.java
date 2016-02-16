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

package com.google.devtools.build.lib.rules.objc;

import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.common.options.Converters;
import com.google.devtools.common.options.Option;

import java.util.List;

/**
 * Command-line options for J2ObjC translation of Java source code to ObjC.
 * These command line options are used by Java rules that can be transpiled
 * (specifically, J2ObjCAspects thereof).
 */
public class J2ObjcCommandLineOptions extends FragmentOptions {
  @Option(name = "j2objc_translation_flags",
      converter = Converters.CommaSeparatedOptionListConverter.class,
      allowMultiple = true,
      defaultValue = "",
      category = "flags",
      help = "Additional options to pass to the J2ObjC tool."
      )
  public List<String> translationFlags;

  @Option(name = "j2objc_dead_code_removal",
      defaultValue = "false",
      category = "undocumented",
      help = "Whether to perform J2ObjC dead code removal to strip unused code from the final app "
          + "bundle."
      )
  public boolean removeDeadCode;

  @Override
  public void addAllLabels(Multimap<String, Label> labelMap) {}
}
