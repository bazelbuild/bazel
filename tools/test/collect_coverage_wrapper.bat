:: ==============================================================================
:: Coverage Collection Wrapper for Windows
:: ==============================================================================
:: 
:: This batch file serves as a Windows wrapper to execute the collect_coverage.sh
:: shell script using bash. Since the coverage collection script is written in
:: bash/shell script format, it cannot run directly on Windows. This wrapper
:: bridges that gap by invoking the script through the bash shell.
::
:: PREREQUISITES:
:: 1. Set the BAZEL_SH environment variable to point to your bash executable
::    Example: set BAZEL_SH=C:\msys64\usr\bin\bash.exe
::
:: 2. When running Bazel tests with coverage, use the following flag:
::    bazel test --test_env=BAZEL_SH <your_test_targets>
::
:: 3. Follow the official Bazel Windows installation guide:
::    https://bazel.build/install/windows
::    This includes installing MSYS2 or another compatible bash environment.
::
:: USAGE:
:: This wrapper is typically invoked automatically by Bazel during test coverage
:: collection. It passes all command-line arguments (%*) to the underlying
:: collect_coverage.sh script.
:: ==============================================================================
@echo off
setlocal
%BAZEL_SH% "%~dp0collect_coverage.sh" %*
endlocal