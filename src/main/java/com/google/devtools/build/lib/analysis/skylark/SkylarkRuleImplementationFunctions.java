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
package com.google.devtools.build.lib.analysis.skylark;

import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.AliasProvider;
import com.google.devtools.build.lib.analysis.CommandHelper;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.LocationExpander;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.configuredtargets.AbstractConfiguredTarget;
import com.google.devtools.build.lib.analysis.stringtemplate.TemplateContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.ParamType;
import com.google.devtools.build.lib.skylarkinterface.SkylarkSignature;
import com.google.devtools.build.lib.syntax.BuiltinFunction;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.Printer;
import com.google.devtools.build.lib.syntax.Runtime;
import com.google.devtools.build.lib.syntax.SkylarkDict;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkList.MutableList;
import com.google.devtools.build.lib.syntax.SkylarkList.Tuple;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import com.google.devtools.build.lib.syntax.SkylarkSemantics;
import com.google.devtools.build.lib.syntax.SkylarkSignatureProcessor;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.syntax.Type.ConversionException;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

// TODO(bazel-team): function argument names are often duplicated,
// figure out a nicely readable way to get rid of the duplications.
/**
 * A helper class to provide an easier API for Skylark rule implementations
 * and hide the original Java API. This is experimental code.
 */
public class SkylarkRuleImplementationFunctions {

  // TODO(bazel-team): add all the remaining parameters
  // TODO(bazel-team): merge executable and arguments
  /**
   * A Skylark built-in function to create and register a SpawnAction using a dictionary of
   * parameters: action( inputs = [input1, input2, ...], outputs = [output1, output2, ...],
   * executable = executable, arguments = [argument1, argument2, ...], mnemonic = 'Mnemonic',
   * command = 'command', )
   */
  @SkylarkSignature(
    name = "action",
    doc =
        "DEPRECATED. Use <a href=\"actions.html#run\">ctx.actions.run()</a> or"
            + " <a href=\"actions.html#run_shell\">ctx.actions.run_shell()</a>. <br>"
            + "Creates an action that runs an executable or a shell command."
            + " You must specify either <code>command</code> or <code>executable</code>.\n"
            + "Actions and genrules are very similar, but have different use cases. Actions are "
            + "used inside rules, and genrules are used inside macros. Genrules also have make "
            + "variable expansion.",
    objectType = SkylarkRuleContext.class,
    returnType = Runtime.NoneType.class,
    parameters = {
      @Param(name = "self", type = SkylarkRuleContext.class, doc = "This RuleContext."),
      @Param(
        name = "outputs",
        type = SkylarkList.class,
        generic1 = Artifact.class,
        named = true,
        positional = false,
        doc = "list of the output files of the action."
      ),
      @Param(
        name = "inputs",
        allowedTypes = {
          @ParamType(type = SkylarkList.class),
          @ParamType(type = SkylarkNestedSet.class),
        },
        generic1 = Artifact.class,
        defaultValue = "[]",
        named = true,
        positional = false,
        doc = "list of the input files of the action."
      ),
      @Param(
        name = "executable",
        type = Object.class,
        allowedTypes = {
          @ParamType(type = Artifact.class),
          @ParamType(type = String.class),
          @ParamType(type = Runtime.NoneType.class),
        },
        defaultValue = "None",
        named = true,
        positional = false,
        doc = "the executable file to be called by the action."
      ),
      @Param(
        name = "arguments",
        allowedTypes = {
          @ParamType(type = SkylarkList.class),
        },
        defaultValue = "[]",
        named = true,
        positional = false,
        doc =
            "command line arguments of the action."
                + "Must be a list of strings or actions.args() objects."
      ),
      @Param(
        name = "mnemonic",
        type = String.class,
        noneable = true,
        defaultValue = "None",
        named = true,
        positional = false,
        doc = "a one-word description of the action, e.g. CppCompile or GoLink."
      ),
      @Param(
        name = "command",
        type = Object.class,
        allowedTypes = {
          @ParamType(type = String.class),
          @ParamType(type = SkylarkList.class, generic1 = String.class),
          @ParamType(type = Runtime.NoneType.class),
        },
        defaultValue = "None",
        named = true,
        positional = false,
        doc =
            "shell command to execute. It is usually preferable to "
                + "use <code>executable</code> instead. "
                + "Arguments are available with <code>$1</code>, <code>$2</code>, etc."
      ),
      @Param(
        name = "progress_message",
        type = String.class,
        noneable = true,
        defaultValue = "None",
        named = true,
        positional = false,
        doc =
            "progress message to show to the user during the build, "
                + "e.g. \"Compiling foo.cc to create foo.o\"."
      ),
      @Param(
        name = "use_default_shell_env",
        type = Boolean.class,
        defaultValue = "False",
        named = true,
        positional = false,
        doc = "whether the action should use the built in shell environment or not."
      ),
      @Param(
        name = "env",
        type = SkylarkDict.class,
        noneable = true,
        defaultValue = "None",
        named = true,
        positional = false,
        doc = "sets the dictionary of environment variables."
      ),
      @Param(
        name = "execution_requirements",
        type = SkylarkDict.class,
        noneable = true,
        defaultValue = "None",
        named = true,
        positional = false,
        doc =
            "information for scheduling the action. See "
                + "<a href=\"$BE_ROOT/common-definitions.html#common.tags\">tags</a> "
                + "for useful keys."
      ),
      @Param(
        // TODO(bazel-team): The name here isn't accurate anymore. This is technically experimental,
        // so folks shouldn't be too attached, but consider renaming to be more accurate/opaque.
        name = "input_manifests",
        type = SkylarkList.class,
        noneable = true,
        defaultValue = "None",
        named = true,
        positional = false,
        doc =
            "(Experimental) sets the input runfiles metadata; "
                + "they are typically generated by resolve_command."
      )
    },
    useLocation = true,
    useEnvironment = true
  )
  private static final BuiltinFunction createSpawnAction =
      new BuiltinFunction("action") {
        public Runtime.NoneType invoke(
            SkylarkRuleContext ctx,
            SkylarkList outputs,
            Object inputs,
            Object executableUnchecked,
            Object arguments,
            Object mnemonicUnchecked,
            Object commandUnchecked,
            Object progressMessage,
            Boolean useDefaultShellEnv,
            Object envUnchecked,
            Object executionRequirementsUnchecked,
            Object inputManifestsUnchecked,
            Location loc,
            Environment env)
            throws EvalException {
          checkDeprecated(
              "ctx.actions.run or ctx.actions.run_shell", "ctx.action", loc, env.getSemantics());
          ctx.checkMutable("action");
          if ((commandUnchecked == Runtime.NONE) == (executableUnchecked == Runtime.NONE)) {
            throw new EvalException(
                loc, "You must specify either 'command' or 'executable' argument");
          }
          boolean hasCommand = commandUnchecked != Runtime.NONE;
          if (!hasCommand) {
            ctx.actions()
                .run(
                    outputs,
                    inputs,
                    executableUnchecked,
                    arguments,
                    mnemonicUnchecked,
                    progressMessage,
                    useDefaultShellEnv,
                    envUnchecked,
                    executionRequirementsUnchecked,
                    inputManifestsUnchecked);

          } else {
            ctx.actions()
                .runShell(
                    outputs,
                    inputs,
                    arguments,
                    mnemonicUnchecked,
                    commandUnchecked,
                    progressMessage,
                    useDefaultShellEnv,
                    envUnchecked,
                    executionRequirementsUnchecked,
                    inputManifestsUnchecked);
          }
          return Runtime.NONE;
        }
      };

  static void checkDeprecated(
      String newApi, String oldApi, Location loc, SkylarkSemantics semantics)
      throws EvalException {
    if (semantics.incompatibleNewActionsApi()) {
      throw new EvalException(
          loc,
          "Use " + newApi + " instead of " + oldApi + ". \n"
              + "Use --incompatible_new_actions_api=false to temporarily disable this check.");
    }
  }

  @SkylarkSignature(
    name = "expand_location",
    doc =
        "Expands all <code>$(location ...)</code> templates in the given string by replacing "
            + "<code>$(location //x)</code> with the path of the output file of target //x. "
            + "Expansion only works for labels that point to direct dependencies of this rule or "
            + "that are explicitly listed in the optional argument <code>targets</code>. "
            + "<br/><br/>"
            + "<code>$(location ...)</code> will cause an error if the referenced target has "
            + "multiple outputs. In this case, please use <code>$(locations ...)</code> since it "
            + "produces a space-separated list of output paths. It can be safely used for a "
            + "single output file, too.",
    objectType = SkylarkRuleContext.class,
    returnType = String.class,
    parameters = {
      @Param(name = "self", type = SkylarkRuleContext.class, doc = "this context."),
      @Param(name = "input", type = String.class, doc = "string to be expanded."),
      @Param(
        name = "targets",
        type = SkylarkList.class,
        generic1 = AbstractConfiguredTarget.class,
        defaultValue = "[]",
        doc = "list of targets for additional lookup information."
      ),
    },
    useLocation = true,
    useEnvironment = true
  )
  private static final BuiltinFunction expandLocation =
      new BuiltinFunction("expand_location") {
        @SuppressWarnings("unused")
        public String invoke(
            SkylarkRuleContext ctx,
            String input,
            SkylarkList targets,
            Location loc,
            Environment env)
            throws EvalException {
          ctx.checkMutable("expand_location");
          try {
            return LocationExpander.withExecPaths(
                    ctx.getRuleContext(),
                    makeLabelMap(targets.getContents(TransitiveInfoCollection.class, "targets")))
                .expand(input);
          } catch (IllegalStateException ise) {
            throw new EvalException(loc, ise);
          }
        }
      };

  /**
   * Builds a map: Label -> List of files from the given labels
   *
   * @param knownLabels List of known labels
   * @return Immutable map with immutable collections as values
   */
  private static ImmutableMap<Label, ImmutableCollection<Artifact>> makeLabelMap(
      Iterable<TransitiveInfoCollection> knownLabels) {
    ImmutableMap.Builder<Label, ImmutableCollection<Artifact>> builder = ImmutableMap.builder();

    for (TransitiveInfoCollection current : knownLabels) {
      builder.put(
          AliasProvider.getDependencyLabel(current),
          ImmutableList.copyOf(current.getProvider(FileProvider.class).getFilesToBuild()));
    }

    return builder.build();
  }

  @SkylarkSignature(
    name = "file_action",
    doc = "DEPRECATED. Use <a href =\"actions.html#write\">ctx.actions.write</a> instead. <br>"
        + "Creates a file write action.",
    objectType = SkylarkRuleContext.class,
    returnType = Runtime.NoneType.class,
    parameters = {
      @Param(name = "self", type = SkylarkRuleContext.class, doc = "this context."),
      @Param(name = "output", type = Artifact.class, doc = "the output file."),
      @Param(name = "content", type = String.class, doc = "the contents of the file."),
      @Param(
        name = "executable",
        type = Boolean.class,
        defaultValue = "False",
        doc = "whether the output file should be executable (default is False)."
      )
    },
    useLocation = true,
    useEnvironment = true
  )
  private static final BuiltinFunction createFileWriteAction =
      new BuiltinFunction("file_action") {
        public Runtime.NoneType invoke(
            SkylarkRuleContext ctx, Artifact output, String content, Boolean executable,
            Location loc, Environment env)
            throws EvalException {
          checkDeprecated("ctx.actions.write", "ctx.file_action", loc, env.getSemantics());
          ctx.checkMutable("file_action");
          ctx.actions().write(output, content, executable);
          return Runtime.NONE;
        }
      };

  @SkylarkSignature(
    name = "empty_action",
    doc =
        "DEPRECATED. Use <a href=\"actions.html#do_nothing\">ctx.actions.do_nothing</a> instead."
            + " <br>"
            + "Creates an empty action that neither executes a command nor produces any "
            + "output, but that is useful for inserting 'extra actions'.",
    objectType = SkylarkRuleContext.class,
    returnType = Runtime.NoneType.class,
    parameters = {
      @Param(name = "self", type = SkylarkRuleContext.class, doc = "this context."),
      @Param(
        name = "mnemonic",
        type = String.class,
        named = true,
        positional = false,
        doc = "a one-word description of the action, e.g. CppCompile or GoLink."
      ),
      @Param(
        name = "inputs",
        allowedTypes = {
          @ParamType(type = SkylarkList.class),
          @ParamType(type = SkylarkNestedSet.class),
        },
        generic1 = Artifact.class,
        named = true,
        positional = false,
        defaultValue = "[]",
        doc = "list of the input files of the action."
      ),
    },
    useLocation = true,
    useEnvironment = true
  )
  private static final BuiltinFunction createEmptyAction =
      new BuiltinFunction("empty_action") {
        @SuppressWarnings("unused")
        public Runtime.NoneType invoke(SkylarkRuleContext ctx, String mnemonic, Object inputs,
            Location loc, Environment env)
            throws EvalException {
          checkDeprecated("ctx.actions.do_nothing", "ctx.empty_action", loc, env.getSemantics());
          ctx.checkMutable("empty_action");
          ctx.actions().doNothing(mnemonic, inputs);
          return Runtime.NONE;
        }
      };

  @SkylarkSignature(
    name = "template_action",
    doc = "DEPRECATED. "
        + "Use <a href=\"actions.html#expand_template\">ctx.actions.expand_template()</a> instead."
        + "<br>Creates a template expansion action.",
    objectType = SkylarkRuleContext.class,
    returnType = Runtime.NoneType.class,
    parameters = {
      @Param(name = "self", type = SkylarkRuleContext.class, doc = "this context."),
      @Param(
        name = "template",
        type = Artifact.class,
        named = true,
        positional = false,
        doc = "the template file, which is a UTF-8 encoded text file."
      ),
      @Param(
        name = "output",
        type = Artifact.class,
        named = true,
        positional = false,
        doc = "the output file, which is a UTF-8 encoded text file."
      ),
      @Param(
        name = "substitutions",
        type = SkylarkDict.class,
        named = true,
        positional = false,
        doc = "substitutions to make when expanding the template."
      ),
      @Param(
        name = "executable",
        type = Boolean.class,
        defaultValue = "False",
        named = true,
        positional = false,
        doc = "whether the output file should be executable (default is False)."
      )
    },
    useLocation = true,
    useEnvironment = true
  )
  private static final BuiltinFunction createTemplateAction =
      new BuiltinFunction("template_action", Arrays.<Object>asList(false)) {
        public Runtime.NoneType invoke(
            SkylarkRuleContext ctx,
            Artifact template,
            Artifact output,
            SkylarkDict<?, ?> substitutionsUnchecked,
            Boolean executable, Location loc, Environment env)
            throws EvalException {
          checkDeprecated("ctx.actions.expand_template", "ctx.template_action", loc,
              env.getSemantics());
          ctx.checkMutable("template_action");
          ctx.actions().expandTemplate(template, output, substitutionsUnchecked, executable);
          return Runtime.NONE;
        }
      };

  // TODO(bazel-team): Remove runfile states from Skylark.
  @SkylarkSignature(name = "runfiles",
      doc = "Creates a runfiles object.",
      objectType = SkylarkRuleContext.class,
      returnType = Runfiles.class,
      parameters = {
        @Param(name = "self", type = SkylarkRuleContext.class, doc = "This context."),
        @Param(name = "files", type = SkylarkList.class, generic1 = Artifact.class,
            defaultValue = "[]", doc = "The list of files to be added to the runfiles."),
        // TODO(bazel-team): If we have a memory efficient support for lazy list containing
        // NestedSets we can remove this and just use files = [file] + list(set)
        // Also, allow empty set for init
        @Param(name = "transitive_files", type = SkylarkNestedSet.class, generic1 = Artifact.class,
            noneable = true, defaultValue = "None",
            doc = "The (transitive) set of files to be added to the runfiles. The depset should "
            + "use the `default` order (which, as the name implies, is the default)."),
        @Param(name = "collect_data", type = Boolean.class, defaultValue = "False",
            doc = "Whether to collect the data "
            + "runfiles from the dependencies in srcs, data and deps attributes."),
        @Param(name = "collect_default", type = Boolean.class, defaultValue = "False",
            doc = "Whether to collect the default "
            + "runfiles from the dependencies in srcs, data and deps attributes."),
        @Param(name = "symlinks", type = SkylarkDict.class, defaultValue = "{}",
            doc = "The map of symlinks to be added to the runfiles, prefixed by workspace name."),
        @Param(name = "root_symlinks", type = SkylarkDict.class, defaultValue = "{}",
            doc = "The map of symlinks to be added to the runfiles.")},
      useLocation = true)
  private static final BuiltinFunction runfiles = new BuiltinFunction("runfiles") {
    public Runfiles invoke(SkylarkRuleContext ctx, SkylarkList files, Object transitiveFiles,
        Boolean collectData, Boolean collectDefault,
        SkylarkDict<?, ?> symlinks, SkylarkDict<?, ?> rootSymlinks,
        Location loc) throws EvalException, ConversionException {
      ctx.checkMutable("runfiles");
      Runfiles.Builder builder = new Runfiles.Builder(
          ctx.getRuleContext().getWorkspaceName(),
          ctx.getConfiguration().legacyExternalRunfiles());
      boolean checkConflicts = false;
      if (EvalUtils.toBoolean(collectData)) {
        builder.addRunfiles(ctx.getRuleContext(), RunfilesProvider.DATA_RUNFILES);
      }
      if (EvalUtils.toBoolean(collectDefault)) {
        builder.addRunfiles(ctx.getRuleContext(), RunfilesProvider.DEFAULT_RUNFILES);
      }
      if (!files.isEmpty()) {
        builder.addArtifacts(files.getContents(Artifact.class, "files"));
      }
      if (transitiveFiles != Runtime.NONE) {
        builder.addTransitiveArtifacts(((SkylarkNestedSet) transitiveFiles).getSet(Artifact.class));
      }
      if (!symlinks.isEmpty()) {
        // If Skylark code directly manipulates symlinks, activate more stringent validity checking.
        checkConflicts = true;
        for (Map.Entry<String, Artifact> entry : symlinks.getContents(
            String.class, Artifact.class, "symlinks").entrySet()) {
          builder.addSymlink(PathFragment.create(entry.getKey()), entry.getValue());
        }
      }
      if (!rootSymlinks.isEmpty()) {
        checkConflicts = true;
        for (Map.Entry<String, Artifact> entry : rootSymlinks.getContents(
            String.class, Artifact.class, "root_symlinks").entrySet()) {
          builder.addRootSymlink(PathFragment.create(entry.getKey()), entry.getValue());
        }
      }
      Runfiles runfiles = builder.build();
      if (checkConflicts) {
        runfiles.setConflictPolicy(Runfiles.ConflictPolicy.ERROR);
      }
      return runfiles;
    }
  };

  /**
   * Ensures the given {@link Map} has keys that have {@link Label} type and values that have either
   * {@link Iterable} or {@link SkylarkNestedSet} type, and raises {@link EvalException} otherwise.
   * Returns a corresponding map where any sets are replaced by iterables.
   */
  // TODO(bazel-team): find a better way to typecheck this argument.
  @SuppressWarnings("unchecked")
  private static Map<Label, Iterable<Artifact>> checkLabelDict(
      Map<?, ?> labelDict, Location loc, Environment env) throws EvalException {
    Map<Label, Iterable<Artifact>> convertedMap = new HashMap<>();
    for (Map.Entry<?, ?> entry : labelDict.entrySet()) {
      Object key = entry.getKey();
      if (!(key instanceof Label)) {
        throw new EvalException(
            loc, Printer.format("invalid key %r in 'label_dict'", key));
      }
      ImmutableList.Builder<Artifact> files = ImmutableList.builder();
      Object val = entry.getValue();
      Iterable<?> valIter;
      try {
        valIter = EvalUtils.toIterableStrict(val, loc, env);
      } catch (EvalException ex) {
        // EvalException is thrown only if the type is wrong.
        throw new EvalException(
            loc, Printer.format("invalid value %r in 'label_dict': " + ex, val));
      }
      for (Object file : valIter) {
        if (!(file instanceof Artifact)) {
          throw new EvalException(
              loc, Printer.format("invalid value %r in 'label_dict'", val));
        }
        files.add((Artifact) file);
      }
      convertedMap.put((Label) key, files.build());
    }
    return convertedMap;
  }

  /** suffix of script to be used in case the command is too long to fit on a single line */
  private static final String SCRIPT_SUFFIX = ".script.sh";

  @SkylarkSignature(
    name = "resolve_command",
    // TODO(bazel-team): The naming here isn't entirely accurate (input_manifests is no longer
    // manifests), but this is experimental/should be opaque to the end user.
    doc =
        "<i>(Experimental)</i> "
            + "Returns a tuple <code>(inputs, command, input_manifests)</code> of the list of "
            + "resolved inputs, the argv list for the resolved command, and the runfiles metadata"
            + "required to run the command, all of them suitable for passing as the same-named "
            + "arguments of the <code>ctx.action</code> method.",
    objectType = SkylarkRuleContext.class,
    returnType = Tuple.class,
    parameters = {
      @Param(name = "self", type = SkylarkRuleContext.class, doc = "this RuleContext."),
      @Param(
        name = "command",
        type = String.class, // string
        defaultValue = "''",
        named = true,
        positional = false,
        doc = "command to resolve."
      ),
      @Param(
        name = "attribute",
        type = String.class, // string
        defaultValue = "None",
        noneable = true,
        named = true,
        positional = false,
        doc = "name of the associated attribute for which to issue an error, or None."
      ),
      @Param(
        name = "expand_locations",
        type = Boolean.class,
        defaultValue = "False",
        named = true,
        positional = false,
        doc =
            "shall we expand $(location) variables? "
                + "See <a href=\"#expand_location\">ctx.expand_location()</a> for more details."
      ),
      @Param(
        name = "make_variables",
        type = SkylarkDict.class, // dict(string, string)
        noneable = true,
        defaultValue = "None",
        named = true,
        positional = false,
        doc = "make variables to expand, or None."
      ),
      @Param(
        name = "tools",
        defaultValue = "[]",
        type = SkylarkList.class,
        generic1 = TransitiveInfoCollection.class,
        named = true,
        positional = false,
        doc = "list of tools (list of targets)."
      ),
      @Param(
        name = "label_dict",
        type = SkylarkDict.class,
        defaultValue = "{}",
        named = true,
        positional = false,
        doc =
            "dictionary of resolved labels and the corresponding list of Files "
                + "(a dict of Label : list of Files)"
      ),
      @Param(
        name = "execution_requirements",
        type = SkylarkDict.class,
        defaultValue = "{}",
        named = true,
        positional = false,
        doc =
            "information for scheduling the action to resolve this command. See "
                + "<a href=\"/docs/be/common-definitions.html#common.tags\">tags</a> "
                + "for useful keys."
      ),
    },
    useLocation = true,
    useEnvironment = true
  )
  private static final BuiltinFunction resolveCommand =
      new BuiltinFunction("resolve_command") {
        @SuppressWarnings("unchecked")
        public Tuple<Object> invoke(
            SkylarkRuleContext ctx,
            String command,
            Object attributeUnchecked,
            Boolean expandLocations,
            Object makeVariablesUnchecked,
            SkylarkList tools,
            SkylarkDict<?, ?> labelDictUnchecked,
            SkylarkDict<?, ?> executionRequirementsUnchecked,
            Location loc,
            Environment env)
            throws ConversionException, EvalException {
          ctx.checkMutable("resolve_command");
          Label ruleLabel = ctx.getLabel();
          Map<Label, Iterable<Artifact>> labelDict = checkLabelDict(labelDictUnchecked, loc, env);
          // The best way to fix this probably is to convert CommandHelper to Skylark.
          CommandHelper helper =
              new CommandHelper(
                  ctx.getRuleContext(),
                  tools.getContents(TransitiveInfoCollection.class, "tools"),
                  ImmutableMap.copyOf(labelDict));
          String attribute =
              Type.STRING.convertOptional(attributeUnchecked, "attribute", ruleLabel);
          TemplateContext templateContext = TemplateContext.EMPTY;
          if (!EvalUtils.isNullOrNone(makeVariablesUnchecked)) {
            Map<String, String> makeVariables =
                Type.STRING_DICT.convert(makeVariablesUnchecked, "make_variables", ruleLabel);
            templateContext = ctx.getConfigurationMakeVariableContext(makeVariables);
          }
          command = helper.expandForSkylark(command, attribute, templateContext, expandLocations);
          List<Artifact> inputs = new ArrayList<>();
          inputs.addAll(helper.getResolvedTools());

          ImmutableMap<String, String> executionRequirements =
              ImmutableMap.copyOf(
                  SkylarkDict.castSkylarkDictOrNoneToDict(
                      executionRequirementsUnchecked,
                      String.class,
                      String.class,
                      "execution_requirements"));
          List<String> argv =
              helper.buildCommandLine(command, inputs, SCRIPT_SUFFIX, executionRequirements);
          return Tuple.<Object>of(
              MutableList.copyOf(env, inputs),
              MutableList.copyOf(env, argv),
              helper.getToolsRunfilesSuppliers());
        }
      };

  static {
    SkylarkSignatureProcessor.configureSkylarkFunctions(SkylarkRuleImplementationFunctions.class);
  }
}
