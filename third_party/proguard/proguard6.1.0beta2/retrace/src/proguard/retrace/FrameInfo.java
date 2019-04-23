/*
 * ProGuard -- shrinking, optimization, obfuscation, and preverification
 *             of Java bytecode.
 *
 * Copyright (c) 2002-2019 Guardsquare NV
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option)
 * any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 */
package proguard.retrace;

/**
 * This class represents the class name, field name, method name, etc.
 * possibly found in a stack frame. Values that are not defined are null.
 */
public class FrameInfo
{
    private final String className;
    private final String sourceFile;
    private final int    lineNumber;
    private final String type;
    private final String fieldName;
    private final String methodName;
    private final String arguments;


    /**
     * Creates a new FrameInfo with the given information.
     * Any undefined values can be null.
     */
    public FrameInfo(String className,
                     String sourceFile,
                     int    lineNumber,
                     String type,
                     String fieldName,
                     String methodName,
                     String arguments)
    {
        this.className  = className;
        this.sourceFile = sourceFile;
        this.lineNumber = lineNumber;
        this.type       = type;
        this.fieldName  = fieldName;
        this.methodName = methodName;
        this.arguments  = arguments;
    }


    public String getClassName()
    {
        return className;
    }


    public String getSourceFile()
    {
        return sourceFile;
    }


    public int getLineNumber()
    {
        return lineNumber;
    }


    public String getType()
    {
        return type;
    }


    public String getFieldName()
    {
        return fieldName;
    }


    public String getMethodName()
    {
        return methodName;
    }


    public String getArguments()
    {
        return arguments;
    }


    // Implementations for Object.

    public String toString()
    {
        return FrameInfo.class.getName() + "(class=["+className+"], line=["+lineNumber+"], type=["+type+"], field=["+fieldName+"], method=["+methodName+"], arguments=["+arguments+"]";
    }
}
