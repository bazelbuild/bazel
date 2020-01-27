write-host "Attempting to stop any running bazel processes, to allow upgrade"
try
{
  $running = get-process bazel
}
catch
{
  write-host "No running bazel processes to stop"
  $running = @()
}

if ($running)
{
  write-host "Stopping bazel processes"
  foreach($p in $running)
  {
    stop-process $p
    write-verbose "Stopped $($p.ProcessName) $($p.Id)"
  }
}
