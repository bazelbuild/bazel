// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.skylarkinterface.SkylarkValue;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Printer;
import com.google.devtools.build.lib.syntax.Runtime;
import com.google.devtools.build.lib.vfs.PathFragment;

/**
 * Provides a Skylark interface for all action creation needs.
 */
@SkylarkModule(
    name = "actions",
    category = SkylarkModuleCategory.BUILTIN,
    doc = "Module providing functions to create actions."
)

public class SkylarkActionFactory implements SkylarkValue {
  private final SkylarkRuleContext context;
  private RuleContext ruleContext;


  public SkylarkActionFactory(SkylarkRuleContext context, RuleContext ruleContext) {
    this.context = context;
    this.ruleContext = ruleContext;
  }

  Root newFileRoot() throws EvalException {
    return context.isForAspect()
        ? ruleContext.getConfiguration().getBinDirectory(ruleContext.getRule().getRepository())
        : ruleContext.getBinOrGenfilesDirectory();
  }

  @SkylarkCallable(
      name = "declare_file",
      doc =
          "Declares that rule or aspect creates a file with the given filename. "
              + "If <code>sibling</code> is not specified, file name is relative to "
              + "package directory, otherwise the file is in the same directory as "
              + "<code>sibling</code>. "
              + "You must create an action that generates the file. <br>"
              + "Files that are specified in rule's outputs do not need to be declared and are "
              + "available through <a href=\"ctx.html#outputs\">ctx.outputs</a>.",
      parameters = {
          @Param(
              name = "filename",
              type = String.class,
              doc =
                  "If no 'sibling' provided, path of the new file, relative "
                  + "to the current package. Otherwise a base name for a file "
                   + "('sibling' determines a directory)."
          ),
          @Param(
              name = "sibling",
              doc = "A file that lives in the same directory as the newly created file.",
              type = Artifact.class,
              noneable = true,
              positional = false,
              named = true,
              defaultValue = "None"
          )
      }
  )
  public Artifact declareFile(String filename, Object sibling) throws EvalException {
    context.checkMutable("actions.declare_file");
    if (Runtime.NONE.equals(sibling)) {
      return ruleContext.getPackageRelativeArtifact(filename, newFileRoot());
    } else {
      PathFragment original = ((Artifact) sibling).getRootRelativePath();
      PathFragment fragment = original.replaceName(filename);
      return ruleContext.getDerivedArtifact(fragment, newFileRoot());
    }
  }

  @SkylarkCallable(
      name = "declare_directory",
      doc =
          "Declares that rule or aspect create a directory with the given name, in the "
              + "current package. You must create an action that generates the file. <br>"
              + "Files that are specified in rule's outputs do not need to be declared and are "
              + "available through  <a href=\"ctx.html#outputs\">ctx.outputs</a>.",
      parameters = {
          @Param(
              name = "filename",
              type = String.class,
              doc =
                  "If no 'sibling' provided, path of the new directory, relative "
                      + "to the current package. Otherwise a base name for a file "
                      + "('sibling' defines a directory)."
          ),
          @Param(
              name = "sibling",
              doc = "A file that lives in the same directory as the newly declared directory.",
              type = Artifact.class,
              noneable = true,
              positional = false,
              named = true,
              defaultValue = "None"
          )
      }
  )
  public Artifact declareDirectory(String filename, Object sibling) throws EvalException {
    context.checkMutable("actions.declare_directory");
    if (Runtime.NONE.equals(sibling)) {
      return ruleContext.getPackageRelativeTreeArtifact(
          PathFragment.create(filename), newFileRoot());
    } else {
      PathFragment original = ((Artifact) sibling).getRootRelativePath();
      PathFragment fragment = original.replaceName(filename);
      return ruleContext.getTreeArtifact(fragment, newFileRoot());
    }
  }


    @Override
  public boolean isImmutable() {
    return context.isImmutable();
  }

  @Override
  public void write(Appendable buffer, char quotationMark) {
    Printer.append(buffer, "actions for");
    context.write(buffer, quotationMark);
  }

  void nullify() {
    ruleContext = null;
  }
}
