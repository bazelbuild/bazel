---
layout: documentation
title: Building JavaScript Outputs
---

# Building JavaScript Outputs

Bazel supports an incremental and customizable means of building and testing
JavaScript outputs from JavaScript and TypeScript sources.

It can also support build steps needed for frameworks like Angular.

**Note:** This document describes Bazel features and workflows that are useful,
but the Bazel team has not fully verified and does not officially support
these features and workflows.

## Contents

*  [Overview](#overview)
*  [Setting up your environment](#setting-up-your-environment)
   *  [Step 1: Installing Bazel](#step-1-installing-bazel)
   *  [Step 2: Installing iBazel](#step-2-installing-ibazel)
   *  [Step 3: Configuring the `bazel.rc` file](#step-3-configuring-the-bazel-rc-file)
   *  [Step 4: Setup linting (optional)](#step-4-linting)
*  [Building JavaScript inputs](#building-javascript)
*  [Building TypeScript inputs](#building-typescript)
   *  [Compiling TypeScript inputs (`ts_library`)](#compiling-typescript-inputs-ts_library)
   *  [Running a development server (`ts_devserver`)](#running-a-development-server-ts_devserver)
   *  [Testing TypeScript code (`ts_web_test`)](#testing-typescript-code-ts_web_test)
*  [Building Angular inputs](#building-angular-inputs)

## Overview

Bazel rules for building JavaScript outputs are split into three layers, since
you can use JavaScript without TypeScript, and TypeScript without Angular.
This document assumes you are already familiar with Bazel and uses the
[Angular for Bazel sample project](https://github.com/angular/angular-bazel-example)
to illustrate the recommended configuration. You can use the sample project as a
starting point and add your own code to it to start building with Bazel.

If you're new to Bazel, take a look at the ["Getting
Started"](getting-started.html) material before proceeding.

## Setting up your environment

To set up your environment for building JavaScript outputs with Bazel, do the
following:

### Step 1: Installing Bazel

You can either [Install Bazel](install.html) following the same steps that you
would for backend development, or you can install NodeJS with npm and run
`npm install -g @bazel/bazel`.

### Step 2: Installing iBazel

iBazel, or iterative Bazel, is a "watchdog" version of Bazel that automatically
runs whenever your source files change. Use it to auto-run your tests and
auto-refresh the code served by the development server.

[Install iBazel](https://github.com/bazelbuild/bazel-watcher) globally using
your package manager of choice. The global installation is required so that
iBazel is in your PATH variable. Also install a specific version of iBazel into
your project so that your whole team updates together. For example:

```
npm install --save-dev @bazel/ibazel
npm install --global @bazel/ibazel
```
or

```
yarn add -D @bazel/ibazel
yarn global add @bazel/ibazel
```

To use `ibazel`, simply replace `bazel` with `ibazel` in your Bazel commands.

### Step 3: Configuring the `bazel.rc` file

Any Bazel build flag or option that can be placed on the command line can also
be set in the project's [`bazel.rc` file](https://docs.bazel.build/guide.html#bazelrc)
so that it is applied every time Bazel builds or tests the project.

Based on how you want to share Bazel settings across your project and team(s),
you can use any combination of the following techniques:

*   **To use the same Bazel settings for the project**, create a `tools/bazel.rc`
    file at the root of the Bazel workspace. Adding it to the workspace will
    check the file into version control and propagate it to others working on
    the project as well as the CI system.

*   **To personalize Bazel settings for the project but not share them,**
    create a `.bazel.rc` file at the root of the project and add the file to
    your `.gitignore` list.

*   **To personalize Bazel settings for all of your projects on your
    local machine,** create a `.bazel.rc` file in your home directory.

Here's an example `tools/bazel.rc` file to share with your team. Modify this
template as needed.

```
###############################
# Directory structure         #
###############################

# Artifacts are typically placed in a directory called "dist"
# Be aware that this setup will still create a bazel-out symlink in
# your project directory, which you must exclude from version control and your
# editor's search path.
build --symlink_prefix=dist/

###############################
# Output                      #
###############################

# A more useful default output mode for bazel query, which
# prints "ng_module rule //foo:bar" instead of just "//foo:bar".
query --output=label_kind

# By default, failing tests don't print any output, it's logged to a
# file instead.
test --test_output=errors

# Show which actions are running under which workers and print all
# the actions running in parallel. This shows that Bazel runs on all
# cores of a CPU.
build --experimental_ui
test --experimental_ui

###############################
# Typescript / Angular / Sass #
###############################
# Make TypeScript and Angular compilation fast, by keeping a few
# copies of the compiler running as daemons, and cache SourceFile
# ASTs to reduce parse time.
build --strategy=TypeScriptCompile=worker --strategy=AngularTemplateCompile=worker

# Enable debugging tests with --config=debug
test:debug --test_arg=--node_options=--inspect-brk --test_output=streamed --test_strategy=exclusive --test_timeout=9999 --nocache_test_results
```

### Step 4: Linting

Add the `buildifier` dependency to your project:

```sh
npm install --save-dev @bazel/buildifier
```

or if you prefer Yarn,

```sh
yarn add -D @bazel/buildifier
```

then add scripts to your `package.json` that run Buildifier.
We've selected a set of enabled warnings here, you could add and remove from
this list.

```json
"scripts": {
        "bazel:format": "find . -type f \\( -name \"*.bzl\" -or -name WORKSPACE -or -name BUILD -or -name BUILD.bazel \\) ! -path \"*/node_modules/*\" | xargs buildifier -v --warnings=attr-cfg,attr-license,attr-non-empty,attr-output-default,attr-single-file,constant-glob,ctx-actions,ctx-args,depset-iteration,depset-union,dict-concatenation,duplicated-name,filetype,git-repository,http-archive,integer-division,load,load-on-top,native-build,native-package,out-of-order-load,output-group,package-name,package-on-top,positional-args,redefined-variable,repository-name,same-origin-load,string-iteration,unsorted-dict-items,unused-variable",
        "bazel:lint": "yarn bazel:format --lint=warn",
        "bazel:lint-fix": "yarn bazel:format --lint=fix"
}
```

> The Bazel team is aware that this configuration is not ergonomic. Follow https://github.com/bazelbuild/buildtools/issues/479

> Also the Buildifier tool is not available on Windows. Follow https://github.com/bazelbuild/buildtools/issues/375

## Building JavaScript

Use the [`rules_nodejs`](https://github.com/bazelbuild/rules_nodejs)
rules to build NodeJS applications and execute JavaScript code within Bazel. You
can execute JavaScript tools in the Bazel toolchain, binary programs, or tests.
The NodeJS rules add the NodeJS runtime to your Bazel project.

See the documentation on that site for setup steps and to configure targets.

## Building TypeScript

See the https://www.npmjs.com/package/@bazel/typescript package.


### Compiling TypeScript inputs (`ts_library`)

The `ts_library` rule compiles one package of TypeScript code at a time. Each
library compiles independently using the `.d.ts` declaration files from its
dependencies. Thus, Bazel will only rebuild a package if the API the package
depends on changes.

The `ts_library `rule, by default, outputs a `.d.ts` file for each `.ts` source
file input into it, plus an ES5 (devmode) `.js` file to be used as inputs for
rule targets that depend on the current target, including transitively.

**Tip:** You can try out the `ts_library` rule by running bazel build src in
the [sample project](https://github.com/alexeagle/angular-bazel-example/wiki).

**Note:** We recommend standardizing your TypeScript settings into a single
`tsconfig.json` file or as few `tsconfig.json` files as possible.

Note the following:

*   Bazel controls parts of the `tsconfig.json `file  that define locations of
    input and output files, manage dependencies on typings, and produce
    JavaScript output that's readable by downstream tooling. Currently, this
    format is unbundled UMD modules, wrapping noth named (non-anonymous) AMD
    modules and `commonjs` modules.

*   Bazel may introduce new requirements for your TypeScript code. For example,
    Bazel uses the `-declarations` flag to produce `.d.ts` outputs required by
    dependent rule targets; your code may require adjustment to not produce
    errors when the `-declarations` flag is in use.

*   If your TypeScript builds are slow, consider granularizing the affected rule
    target(s) into smaller sub-targets and declaring dependencies between them
    appropriately.

### Running a development server (`ts_devserver`)

The `ts_devserver` rule brings up a development server from your application
sources. It's intended for use with the `ibazel run` command so that the server
picks up your code changes immediately. The rule injects a `livereload` script
into the browser, which causes the page to auto-refresh with the latest changes
at the completion of each build.

**Tip:** You can test-drive the development server feature by running
`ibazel run src: devserver` on the [sample project](https://github.com/alexeagle/angular-bazel-example/wiki).


### Testing TypeScript code (`ts_web_test`)

Use the `ts_web_test` rule to execute the Karma test runner. This rule works
best with ibazel so that both the test runner and the browser pick up your
changes at the completion of each build. For faster builds, Bazel bundles your
code and its dependencies into a single JavaScript file delivered to the browser
when the test runner executes.

If you need to match lots of tests with a target pattern such as bazel test //…
or using CI, run the `ts_web_test` rule with the regular `bazel test` command
instead. Bazel will then launch a headless Chrome instance and exit after a
single run.

**Tip:** You can test-drive the `ts_web_test` rule by running `ibazel run` or
`bazel run` on the `src/hello-world:test` target in the [sample project](https://github.com/alexeagle/angular-bazel-example/wiki).


## Building Angular inputs

Bazel can build JavaScript outputs from Angular. For instructions, see [Building Angular with Bazel](https://github.com/alexeagle/angular-bazel-example/wiki/Angular-rules).

