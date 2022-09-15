package com.google.devtools.build.lib.bazel.rules.java;

import com.google.devtools.build.lib.analysis.config.StarlarkDefinedConfigTransition;
import com.google.devtools.build.lib.analysis.starlark.StarlarkAttributeTransitionProvider;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.StructImpl;
import net.starlark.java.eval.*;
import net.starlark.java.syntax.Location;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.Type.STRING;

public class BazelJavaCurrentRepositorySupport {
  private static final String CURRENT_REPOSITORY_SETTING = "@bazel_tools//tools/java/runfiles:current_repository";
  private static final Label RUNFILES_CONSTANTS_RULE = Label.parseCanonicalUnchecked("@@bazel_tools//tools/java/runfiles:java_current_repository");

  static final Attribute CURRENT_REPOSITORY_ATTRIBUTE = attr("_current_repository", STRING)
    .value(new Attribute.ComputedDefault() {
             @Override
             public String getDefault(AttributeMap rule) {
               return rule.getLabel().getRepository().getName();
             }
           }
    )
    .build();

  static final Attribute RUNFILES_CONSTANTS_ATTRIBUTE = attr("$runfiles_constants", LABEL)
    .cfg(new StarlarkAttributeTransitionProvider(makeRunfilesConstantsTransition()))
    .value(RUNFILES_CONSTANTS_RULE)
    .build();

  private static StarlarkDefinedConfigTransition makeRunfilesConstantsTransition() {
    try {
      return StarlarkDefinedConfigTransition.newRegularTransition(
        new StarlarkCallable() {
          @Override
          public Object fastcall(StarlarkThread thread, Object[] positional, Object[] named) throws EvalException {
            String currentRepository = (String) ((StructImpl) positional[1]).getValue("_current_repository");
            return Dict.builder().put(CURRENT_REPOSITORY_SETTING, currentRepository).buildImmutable();
          }

          @Override
          public String getName() {
            return "_current_repository_transition_impl";
          }
        },
        StarlarkList.empty(),
        StarlarkList.immutableOf(CURRENT_REPOSITORY_SETTING),
        StarlarkSemantics.DEFAULT,
        Label.parseCanonicalUnchecked("@@_builtins//:common/java/java_common.bzl"),
        Location.BUILTIN,
        RepositoryMapping.ALWAYS_FALLBACK);
    } catch (EvalException e) {
      throw new RuntimeException(e);
    }
  }
}
