:: Copyright 2020 The Bazel Authors. All rights reserved.
::
:: Licensed under the Apache License, Version 2.0 (the "License");
:: you may not use this file except in compliance with the License.
:: You may obtain a copy of the License at
::
::    http://www.apache.org/licenses/LICENSE-2.0
::
:: Unless required by applicable law or agreed to in writing, software
:: distributed under the License is distributed on an "AS IS" BASIS,
:: WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
:: See the License for the specific language governing permissions and
:: limitations under the License.

if not defined RELEASE_NAME (
  set RELEASE_NAME=unknown
)

mkdir T:\tmp\tool
set BAZELISK=T:\tmp\tool\bazelisk.exe
powershell /c "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; (New-Object Net.WebClient).DownloadFile('https://github.com/bazelbuild/bazelisk/releases/download/v1.2.1/bazelisk-windows-amd64.exe', '%BAZELISK%')"

set PATH=C:\python37;%PATH%

%BAZELISK% build //src:bazel.exe
mkdir output
copy bazel-bin\src\bazel.exe output\bazel.exe

output\bazel build -c opt --copt=-w --host_copt=-w --stamp --embed_label %RELEASE_NAME% src/bazel scripts/packages/bazel.zip

mkdir artifacts
move bazel-bin\src\bazel artifacts\bazel-%RELEASE_NAME%-windows-x86_64.exe
move bazel-bin\scripts\packages\bazel.zip artifacts\bazel-%RELEASE_NAME%-windows-x86_64.zip
