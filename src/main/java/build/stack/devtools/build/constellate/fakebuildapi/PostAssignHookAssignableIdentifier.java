package build.stack.devtools.build.constellate.fakebuildapi;

/**
* PostAssignHookAssignableIdentifier implementations get a callback with the
* PostAssignHook to receive the name of the variable they are being assigned as. 
*/
public interface PostAssignHookAssignableIdentifier {
    void setAssignedName(String assignedName);
    String getAssignedName();
}
