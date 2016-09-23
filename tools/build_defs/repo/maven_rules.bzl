# Copyright 2016 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Implementations of Maven rules in Skylark:
# 1) maven_jar(name, artifact, repository, sha1)
#    The API of this is largely the same as the native maven_jar rule,
#    except for the server attribute, which is not implemented.
# 2) maven_dependency_plugin()
#    This rule downloads the maven-dependency-plugin used internally
#    for testing and the implementation for the fetching of artifacts.

# Installation requirements prior to using this rule:
# 1) Maven binary: `mvn`
# 2) Maven plugin: `maven-dependency-plugin:2.10`
# Get it: $ mvn org.apache.maven.plugins:maven-dependency-plugin:2.10:get \
#    -Dartifact=org.apache.maven.plugins:maven-dependency-plugin:2.10 \
#    -Dmaven.repo.local=$HOME/.m2/repository # or specify your own local repository

"""Rules for retrieving Maven dependencies (experimental)"""

MAVEN_CENTRAL_URL = "https://repo1.maven.org/maven2"

# Binary dependencies needed for running the bash commands
DEPS = ["mvn", "openssl", "awk"]

MVN_PLUGIN = "org.apache.maven.plugins:maven-dependency-plugin:2.10"


def _execute(ctx, command):
  return ctx.execute(["bash", "-c", """
set -ex
%s""" % command])


# Fail fast
def _check_dependencies(ctx):
  for dep in DEPS:
    if ctx.which(dep) == None:
      fail("maven_jar requires %s as a dependency. Please check your PATH." % dep)


def _validate_attr(ctx):
  if (ctx.attr.server != None):
    fail("%s specifies a 'server' attribute which is currently not supported." % ctx.name)


def _artifact_dir(coordinates):
  return "/".join(coordinates.group_id.split(".") +
                  [coordinates.artifact_id, coordinates.version])


# Creates a struct containing the different parts of an artifact's FQN
def _create_coordinates(fully_qualified_name):
  parts = fully_qualified_name.split(":")
  packaging = None
  classifier = None

  if len(parts) == 3:
    group_id, artifact_id, version = parts
  elif len(parts) == 4:
    group_id, artifact_id, packaging, version = parts
  elif len(parts) == 5:
    group_id, artifact_id, packaging, classifier, version = parts
  else:
    fail("Invalid fully qualified name for artifact: %s" % fully_qualified_name)

  return struct(
      fully_qualified_name = fully_qualified_name,
      group_id = group_id,
      artifact_id = artifact_id,
      packaging = packaging,
      classifier = classifier,
      version = version,
  )


# NOTE: Please use this method to define ALL paths that the maven_jar
# rule uses. Doing otherwise will lead to inconsistencies and/or errors.
#
# CONVENTION: *_path refers to files, *_dir refers to directories.
def _create_paths(ctx, coordinates):
  """Creates a struct that contains the paths to create the cache WORKSPACE"""

  # e.g. guava-18.0.jar
  # TODO(jingwen): Make the filename conditional on package type (jar, war, etc.)
  jar_filename = "%s-%s.jar" % (coordinates.artifact_id, coordinates.version)
  sha1_filename = "%s.sha1" % jar_filename

  # e.g. com/google/guava/guava/18.0
  relative_jar_dir = _artifact_dir(coordinates)

  # The symlink to the actual .jar is stored in this dir, along with the
  # BUILD file.
  symlink_dir = "jar"

  m2 = ".m2"
  m2_repo = "/".join([m2, "repository"]) # .m2/repository

  m2_plugin_coordinates = _create_coordinates(MVN_PLUGIN)
  m2_plugin_filename = "%s-%s.jar" % (m2_plugin_coordinates.artifact_id,
                                      m2_plugin_coordinates.version)
  m2_plugin_dir = "/".join([m2_repo, _artifact_dir(m2_plugin_coordinates)])

  if (ctx.attr.local_repository):
    bazel_m2_dir = ctx.path("%s" % (ctx.path(ctx.attr.local_repository).dirname))
  else:
    bazel_m2_dir = None

  return struct(
      jar_filename = jar_filename,
      sha1_filename = sha1_filename,

      symlink_dir = ctx.path(symlink_dir),

      # e.g. external/com_google_guava_guava/ \
      #        .m2/repository/com/google/guava/guava/18.0/guava-18.0.jar
      jar_path = ctx.path("/".join([m2_repo, relative_jar_dir, jar_filename])),
      jar_dir = ctx.path("/".join([m2_repo, relative_jar_dir])),

      sha1_path = ctx.path("/".join([m2_repo, relative_jar_dir, sha1_filename])),

      # e.g. external/com_google_guava_guava/jar/guava-18.0.jar
      symlink_jar_path = ctx.path("/".join([symlink_dir, jar_filename])),

      # maven directories and filepaths
      m2_dir = ctx.path(m2),
      m2_repo_dir = ctx.path(m2_repo),
      m2_settings_path = ctx.path("settings.xml"),
      m2_plugin_dir = ctx.path(m2_plugin_dir),
      m2_plugin_path = ctx.path("/".join([m2_plugin_dir, m2_plugin_filename])),

      bazel_m2_dir = bazel_m2_dir,
  )


# Provides the syntax "@jar_name//jar" for dependencies
def _generate_build_file(ctx, paths):
  contents = """
# DO NOT EDIT: automatically generated BUILD file for maven_jar rule {rule_name}

java_import(
    name = 'jar',
    jars = ['{jar_filename}'],
    visibility = ['//visibility:public']
)

filegroup(
    name = 'file',
    srcs = ['{jar_filename}'],
    visibility = ['//visibility:public']
)\n""".format(rule_name = ctx.name, jar_filename = paths.jar_filename)
  ctx.file('%s/BUILD' % paths.symlink_dir, contents, False)


# Used for integration tests within bazel.
def _mvn_init(ctx, paths, repository):
  if repository == "":
    repository = MAVEN_CENTRAL_URL

  # Symlink the m2 cache to the local m2 in //external
  ctx.symlink(paths.bazel_m2_dir, paths.m2_dir)

  # Having a custom settings.xml and setting a mirror is the only way to
  # force the Maven binary to download from the specified repository
  # directly, and skip the default configured repositories.
  _SETTINGS_XML = """
<!-- # DO NOT EDIT: automatically generated settings.xml for maven_jar rule {rule_name} -->
<settings xmlns="http://maven.apache.org/SETTINGS/1.0.0"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:schemaLocation="http://maven.apache.org/SETTINGS/1.0.0
    https://maven.apache.org/xsd/settings-1.0.0.xsd">
  <localRepository>{localRepository}</localRepository>
  <mirrors>
    <mirror>
      <id>central</id>
      <url>{mirror}</url>
      <mirrorOf>*,default</mirrorOf>
    </mirror>
  </mirrors>
</settings>
""".format(
    rule_name = ctx.name,
    mirror = repository, # Required because the Bazel test environment has no internet access
    localRepository = paths.m2_repo_dir,
  )

  # Overwrite default settings file
  ctx.file("%s" % paths.m2_settings_path, _SETTINGS_XML)


def _file_exists(ctx, filename):
  return _execute(ctx, "[[ -f %s ]] && exit 0 || exit 1" % filename).return_code == 0


# Constructs the maven command to retrieve the dependencies from remote
# repositories using the dependency plugin, and executes it.
def _mvn_download(ctx, paths, fully_qualified_name):
  repository = ctx.attr.repository
  if repository == "":
    repository = MAVEN_CENTRAL_URL

  # If a custom settings file exists, we'll use that. If not, Maven will use the default settings.
  mvn_flags = ""
  if _file_exists(ctx, paths.m2_settings_path):
    mvn_flags += "-s %s" % paths.m2_settings_path

  # dependency:get step. Downloads the artifact into the local repository.
  mvn_get = MVN_PLUGIN + ":get"
  mvn_artifact = "-Dartifact=%s" % fully_qualified_name
  mvn_transitive = "-Dtransitive=false"
  mvn_remote_repo = "-Dmaven.repo.remote=%s" % repository
  command = " ".join(["mvn", mvn_flags, mvn_get, mvn_transitive, mvn_remote_repo, mvn_artifact])
  exec_result = _execute(ctx, command)
  if exec_result.return_code != 0:
    fail("%s\n%s\nFailed to fetch Maven dependency" % (exec_result.stdout, exec_result.stderr))

  # dependency:copy step. Moves the artifact from the local repository into //external.
  mvn_copy = MVN_PLUGIN + ":copy"
  mvn_output_dir = "-DoutputDirectory=%s" % paths.jar_dir
  command = " ".join(["mvn", mvn_flags, mvn_copy, mvn_artifact, mvn_output_dir])
  exec_result = _execute(ctx, command)
  if exec_result.return_code != 0:
    fail("%s\n%s\nFailed to fetch Maven dependency" % (exec_result.stdout, exec_result.stderr))


def _check_sha1(ctx, paths, sha1):
  actual_sha1 = _execute(ctx, "openssl sha1 %s | awk '{printf $2}'" % paths.jar_path).stdout

  if sha1.lower() != actual_sha1.lower():
    fail(("{rule_name} has SHA-1 of {actual_sha1}, " +
          "does not match expected SHA-1 ({expected_sha1})").format(
              rule_name = ctx.name,
              expected_sha1 = sha1,
              actual_sha1 = actual_sha1))
  else:
    _execute(ctx, "echo %s %s > %s" % (sha1, paths.jar_path, paths.sha1_path))


def _maven_jar_impl(ctx):
  # Ensure that we have all of the dependencies installed
  _check_dependencies(ctx)

  # Provide warnings and errors about attributes
  _validate_attr(ctx)

  # Create a struct to contain the different parts of the artifact FQN
  coordinates = _create_coordinates(ctx.attr.artifact)

  # Create a struct to store the relative and absolute paths needed for this rule
  paths = _create_paths(ctx, coordinates)

  _generate_build_file(
      ctx = ctx,
      paths = paths,
  )

  # Initialize local settings.xml files and symlink the dependency plugin
  # artifact to the local repository
  if ctx.attr.local_repository:
    _mvn_init(
        ctx = ctx,
        paths = paths,
        repository = ctx.attr.repository
    )

  if _execute(ctx, "mkdir -p %s" % paths.symlink_dir).return_code != 0:
    fail("%s: Failed to create dirs in execution root.\n" % ctx.name)

  # Download the artifact
  _mvn_download(
      ctx = ctx,
      paths = paths,
      fully_qualified_name = coordinates.fully_qualified_name
  )

  if (ctx.attr.sha1 != ""):
    _check_sha1(
        ctx = ctx,
        paths = paths,
        sha1 = ctx.attr.sha1,
    )

  ctx.symlink(paths.jar_path, paths.symlink_jar_path)


_maven_jar_attrs = {
    "artifact": attr.string(
        default = "",
        mandatory = True,
    ),
    "repository": attr.string(default = ""),
    "server": attr.label(default = None),
    "sha1": attr.string(default = ""),
    "local_repository": attr.label(
        default = None,
        allow_single_file = True,
    )
}


maven_jar = repository_rule(
    implementation=_maven_jar_impl,
    attrs=_maven_jar_attrs,
    local=False,
)


def _maven_dependency_plugin_impl(ctx):
  _BUILD_FILE = """
# DO NOT EDIT: automatically generated BUILD file for maven_dependency_plugin

filegroup(
    name = 'files',
    srcs = glob(['**']),
    visibility = ['//visibility:public']
)
"""
  ctx.file("BUILD", _BUILD_FILE, False)

  _SETTINGS_XML = """
<!-- # DO NOT EDIT: automatically generated settings.xml for maven_dependency_plugin -->
<settings xmlns="http://maven.apache.org/SETTINGS/1.0.0"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:schemaLocation="http://maven.apache.org/SETTINGS/1.0.0
    https://maven.apache.org/xsd/settings-1.0.0.xsd">
  <localRepository>{localRepository}</localRepository>
  <mirrors>
    <mirror>
      <id>central</id>
      <url>{mirror}</url>
      <mirrorOf>*,default</mirrorOf>
    </mirror>
  </mirrors>
</settings>
""".format(
    localRepository = ctx.path("repository"),
    mirror = MAVEN_CENTRAL_URL,
  )
  settings_path = ctx.path("settings.xml")
  ctx.file("%s" % settings_path, _SETTINGS_XML, False)

  # Download the plugin with transitive dependencies
  mvn_flags = "-s %s" % settings_path
  mvn_get = MVN_PLUGIN + ":get"
  mvn_artifact = "-Dartifact=%s" % MVN_PLUGIN
  command = " ".join(["mvn", mvn_flags, mvn_get, mvn_artifact])

  exec_result = _execute(ctx, command)
  if exec_result.return_code != 0:
    fail("%s\nFailed to fetch Maven dependency" % exec_result.stderr)


_maven_dependency_plugin = repository_rule(
    implementation=_maven_dependency_plugin_impl,
)


def maven_dependency_plugin():
  _maven_dependency_plugin(name = "m2")
