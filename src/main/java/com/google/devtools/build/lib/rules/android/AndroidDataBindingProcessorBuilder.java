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
package com.google.devtools.build.lib.rules.android;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.vfs.PathFragment;

/** Builder for creating databinding processing action. */
public class AndroidDataBindingProcessorBuilder {

  /**
   * Creates and registers an action to strip databinding from layout xml and generate the layout
   * info file.
   *
   * @param dataContext The android data context.
   * @param androidResources The resources to process.
   * @param appId The app id (the app's java package).
   * @param dataBindingLayoutInfoOut The output layout info file to write.
   * @return The new AndroidResources that has been processed by databinding.
   */
  public static AndroidResources create(
      AndroidDataContext dataContext,
      AndroidResources androidResources,
      String appId,
      Artifact dataBindingLayoutInfoOut) {

    ImmutableList.Builder<Artifact> databindingProcessedResourcesBuilder = ImmutableList.builder();
    for (Artifact resource : androidResources.getResources()) {

      // Create resources that will be processed by databinding under paths that look like:
      //
      // <bazel-pkg>/databinding-processed-resources/<rule-name>/<bazal-pkg>/<resource-dir>

      Artifact databindingProcessedResource =
          dataContext.getUniqueDirectoryArtifact("databinding-processed-resources",
              resource.getRootRelativePath());

      databindingProcessedResourcesBuilder.add(databindingProcessedResource);
    }
    ImmutableList<Artifact> databindingProcessedResources =
        databindingProcessedResourcesBuilder.build();

    BusyBoxActionBuilder builder = BusyBoxActionBuilder.create(dataContext, "PROCESS_DATABINDING");

    // Create output resource roots that correspond to the paths of the resources created above:
    //
    //   <bazel-pkg>/databinding-processed-resources/<rule-name>/<resource-root>
    //
    // AndroidDataBindingProcessingAction will append each value of --resource_root to its
    // corresponding --output_resource_root, so the only part that needs to be constructed here is
    //
    //   <bazel-pkg>/databinding-processed-resources/<rule-name>
    ArtifactRoot binOrGenfiles = dataContext.getBinOrGenfilesDirectory();
    PathFragment uniqueDir =
        dataContext.getUniqueDirectory(PathFragment.create("databinding-processed-resources"));
    PathFragment outputResourceRoot = binOrGenfiles.getExecPath().getRelative(uniqueDir);

    ImmutableList.Builder<PathFragment> outputResourceRootsBuilder = ImmutableList.builder();
    for (PathFragment resourceRoot : androidResources.getResourceRoots()) {

      outputResourceRootsBuilder.add(outputResourceRoot);

      // The order of these matter, the input root and the output root have to be matched up
      // because the resource processor will iterate over them in pairs.
      builder.addFlag("--resource_root", resourceRoot.toString());
      builder.addFlag("--output_resource_root", outputResourceRoot.toString());
    }

    // Even though the databinding processor really only cares about layout files, we send
    // all the resources so that the new resource root that is created for databinding processing
    // can be used for later processing (e.g. aapt). It would be nice to send only the layout
    // files, but then we'd have to mix roots and rely on sandboxing to "hide" the
    // old unprocessed files, which might not work if, for example, the actions run locally.
    builder.addInputs(androidResources.getResources());

    builder.addOutputs(databindingProcessedResources);

    builder.addOutput("--dataBindingInfoOut", dataBindingLayoutInfoOut);
    builder.addFlag("--appId", appId);

    builder.buildAndRegister("Processing data binding", "ProcessDatabinding");

    return new AndroidResources(databindingProcessedResources, outputResourceRootsBuilder.build());
  }
}
