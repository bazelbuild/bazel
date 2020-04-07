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
package com.google.devtools.build.lib.runtime;

import com.google.devtools.common.options.OptionsBase;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * An annotation that lets blaze commands specify their options and their help.
 * The annotations are processed by {@link BlazeCommand}.
 */
@Target(ElementType.TYPE)
@Retention(RetentionPolicy.RUNTIME)
public @interface Command {
  /**
   * The name of the command, as the user would type it.
   */
  String name();

  /**
   * Options processed by the command, indicated by options interfaces. These interfaces must
   * contain methods annotated with {@link com/google/devtools/build/lib/runtime/Command.java used
   * only in javadoc: com.google.devtools.common.options.Option}.
   */
  Class<? extends OptionsBase>[] options() default {};

  /**
   * The set of other Blaze commands that this annotation's command "inherits"
   * options from.  These classes must be annotated with {@link Command}.
   */
  Class<? extends BlazeCommand>[] inherits() default {};

  /**
   * A short description, which appears in 'blaze help'.
   */
  String shortDescription();

  /**
   * True if the configuration-specific options should be available for this command.
   */
  boolean usesConfigurationOptions() default false;

  /**
   * True if the command runs a build.
   */
  boolean builds() default false;

  /**
   * True if the command should not be shown in the output of 'blaze help'.
   */
  boolean hidden() default false;

  /**
   * Specifies whether this command allows a residue after the parsed options.
   * For example, a command might expect a list of targets to build in the
   * residue.
   */
  boolean allowResidue() default false;

  /**
   * Specifies whether the command line residue might have sensitive data, or arbitrary command
   * line values.
   */
  boolean hasSensitiveResidue() default false;

  /**
   * Returns true if this command wants to write binary data to stdout.
   * Enabling this flag will disable ANSI escape stripping for this command.
   * This should be used in conjunction with {@code Reporter#switchToAnsiAllowingHandler}.
   * See {@link RunCommand} for example usage.
   */
  boolean binaryStdOut() default false;

  /**
   * Returns true if this command wants to write binary data to stderr.
   * Enabling this flag will disable ANSI escape stripping for this command.
   * This should be used in conjunction with {@code Reporter#switchToAnsiAllowingHandler}.
   * See {@link RunCommand} for example usage.
   */
  boolean binaryStdErr() default false;

  /**
   * Returns true if this command may want to write to the command.log.
   *
   * <p>The clean command would typically set this to false so it can delete the command.log.
   */
  boolean writeCommandLog() default true;

  /**
   * The help message for this command.  If the value starts with "resource:",
   * the remainder is interpreted as the name of a text file resource (in the
   * .jar file that provides the Command implementation class).
   */
  String help();

  /**
   * Returns true iff this command may only be run from within a Blaze workspace. Broadly, this
   * should be true for any command that interprets the package-path, since it's potentially
   * confusing otherwise.
   */
  boolean mustRunInWorkspace() default true;

  /**
   * Returns true iff this command is allowed to run in the output directory,
   * i.e. $OUTPUT_BASE/_blaze_$USER/$MD5/... . No command should be allowed to run here,
   * but there are some legacy uses of 'blaze query'.
   */
  boolean canRunInOutputDirectory() default false;

  /**
   * Returns the type completion help for this command, that is the type arguments that this command
   * expects. It can be a whitespace separated list if the command take several arguments. The type
   * of each arguments can be <code>label</code>, <code>path</code>, <code>string</code>, ...
   * It can also be a comma separated list of values, e.g. <code>{value1,value2}<code>. If a command
   * accept several argument types, they can be combined with |, e.g <code>label|path</code>.
   */
  String completion() default "";
}
