:: Copyright 2016 The Bazel Authors. All rights reserved.
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

:: Invoke the python script under pydir with the same basename
@echo OFF
set arg0=%1
for %%F in ("%arg0%") do set DRIVER_BIN=%%~dpF

for /F %%i in ("%arg0%") do set TOOLNAME=%%~ni

set PYDIR="%DRIVER_BIN%pydir"

if not defined MSVCPYTHON set MSVCPYTHON=%{python_binary}
%MSVCPYTHON% -B "%PYDIR%\%TOOLNAME%.py" %*
