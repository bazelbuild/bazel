// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.docgen;

import com.google.devtools.build.docgen.builtin.BuiltinProtos.Builtins;
import com.google.devtools.build.docgen.builtin.BuiltinProtos.Callable;
import com.google.devtools.build.docgen.builtin.BuiltinProtos.Param;
import com.google.devtools.build.docgen.builtin.BuiltinProtos.Type;
import com.google.devtools.build.docgen.builtin.BuiltinProtos.Value;
import com.google.devtools.build.docgen.skylark.SkylarkConstructorMethodDoc;
import com.google.devtools.build.docgen.skylark.SkylarkMethodDoc;
import com.google.devtools.build.docgen.skylark.SkylarkModuleDoc;
import com.google.devtools.build.docgen.skylark.SkylarkParamDoc;
import com.google.devtools.build.lib.util.Classpath.ClassPathException;
import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Map;

/** The main class for the Skylark documentation generator. */
public class ApiExporter {

  private static void appendBuiltins(String builtinsFile) {
    try (BufferedOutputStream out = new BufferedOutputStream(new FileOutputStream(builtinsFile))) {
      Builtins.Builder builtins = Builtins.newBuilder();

      Map<String, SkylarkModuleDoc> allTypes = SkylarkDocumentationCollector.collectModules();

      // Add all global variables and functions in Builtins as Values.
      SkylarkModuleDoc topLevelModule =
          allTypes.remove(SkylarkDocumentationCollector.getTopLevelModule().name());
      for (SkylarkMethodDoc meth : topLevelModule.getMethods()) {
        builtins.addGlobal(collectFieldInfo(meth));
      }
      for (Map.Entry<String, SkylarkModuleDoc> modEntry : allTypes.entrySet()) {
        SkylarkModuleDoc mod = modEntry.getValue();

        // Include SkylarkModuleDoc in Builtins as a Type.
        Type.Builder type = Type.newBuilder();
        type.setName(mod.getName());
        type.setDoc(mod.getDocumentation());
        for (SkylarkMethodDoc meth : mod.getJavaMethods()) {
          // Constructors should be exported as globals.
          if (meth instanceof SkylarkConstructorMethodDoc) {
            builtins.addGlobal(collectFieldInfo(meth));
          } else {
            type.addField(collectFieldInfo(meth));
          }
        }
        builtins.addType(type);

        // Include SkylarkModuleDoc in Builtins as a Value.
        Value.Builder value = Value.newBuilder();
        value.setName(mod.getName());
        value.setType(mod.getName());
        value.setDoc(mod.getDocumentation());
        builtins.addGlobal(value);
      }
      Builtins build = builtins.build();
      build.writeTo(out);
    } catch (IOException | ClassPathException e) {
      System.err.println(e);
    }
  }

  private static Value.Builder collectFieldInfo(SkylarkMethodDoc meth) {
    Value.Builder field = Value.newBuilder();
    field.setName(meth.getShortName());
    field.setDoc(meth.getDocumentation());
    if (meth.isCallable()) {
      Callable.Builder callable = Callable.newBuilder();
      for (SkylarkParamDoc par : meth.getParams()) {
        Param.Builder param = Param.newBuilder();
        param.setName(par.getName());
        param.setType(par.getType());
        param.setDoc(par.getDocumentation());
        param.setDefaultValue(par.getDefaultValue());
        callable.addParam(param);
      }
      callable.setReturnType(meth.getReturnType());
      field.setCallable(callable);
    } else {
      field.setType(meth.getReturnType());
    }
    return field;
  }

  public static void main(String[] args) {
    if (args.length != 1) {
      throw new IllegalArgumentException(
          "Expected one argument. Usage:\n" + "{api_exporter_bin} {builtin_output_file}");
    }

    String builtinsProtoFile = args[0];

    appendBuiltins(builtinsProtoFile);
  }
}

