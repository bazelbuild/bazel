Intro
=====

This is a minimal viable set of C# bindings for building csharp code with
mono. It's still pretty rough but it works as a proof of concept that could
grow into something more. If windows support ever happens for Bazel then this
might become especially valuable.

Rules
=====

* csharp\_library

        csharp_library(
            name="MyLib",
            srcs=["MyLib.cs"],
            deps=["//my/dependency:SomeLib"],
        )

* csharp\_binary

        csharp_binary(
            name="MyApp",
            main="MyApp", # optional name of the main class.
            srcs=["MyApp.cs"],
            deps=["//my/dependency:MyLib"],
        )

* csharp\_nunit\_test

        csharp_nunit_test(
            name="MyApp",
            srcs=["MyApp.cs"],
            deps=["//my/dependency:MyLib"],
        )

Shared attributes for all csharp rules
--------------------------------------

<table class=table table-condensed table-bordered table-params">
<colgroup>
  <col class="col-param" />
  <col class="param-description" />
</colgroup>
<thead>
  <tr>
    <th colspan="2">Attributes</th>
  </tr>
</thead>
<tbody>
<tr><td>name</td>string<td><td>Unique name for this rule</td></td>Required</td></tr>
<tr><td>srcs</td>List of Labels<td><td>Csharp .cs or .resx files.</td></td>Required</td></tr>
<tr><td>deps</td>List of Labels<td><td>Dependencies for this rule.</td></td>Optional</td></tr>
<tr><td>warn</td>Int<td><td>Compiler warn level for this library. (Defaults to 4.)</td></td>optional</td></tr>
<tr><td>csc</td>string<td><td>Override the default csharp compiler.</td></td>Optional</td></tr>
</tbody>
</table>

Usage
=====

Copy the contents of the dotnet.WORKSPACE file into your WORKSPACE file.

Things still missing:
=====================

- Handle .resx files correctly.
- .Net Modules
- building documentation.
- Pulling Mono in through a mono.WORKSPACE file.

Future nice to haves:
=====================

- building csproj and sln files for VS and MonoDevelop.
- Nuget Packaging
- Windows .Net framwork support
