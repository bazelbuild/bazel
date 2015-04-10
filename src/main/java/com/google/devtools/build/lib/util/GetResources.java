package com.google.devtools.build.lib.util;

public class GetResources {
  
  static {
    System.loadLibrary("GetResources");
  }

  private native int getNumProcessors();

  private native long getMemoryAvailable();

  public static int NumProcessors(){
    int numProc = new GetResources().getNumProcessors();
    return numProc;
  }

  public static long MemoryAvailable(){
    long memoryAvail= new GetResources().getMemoryAvailable();
    return memoryAvail;
  }

}
